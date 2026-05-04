import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import A3TGCN, TGCN, GConvGRU, GConvLSTM


def make_batched_edge_index(edge_index: torch.Tensor, num_nodes: int, batch_size: int, device):
    # edge_index: (2, E)
    edge_count = edge_index.size(1)
    batched = edge_index.repeat(1, batch_size).to(device)
    offsets = (torch.arange(batch_size, device=device) * num_nodes).repeat_interleave(edge_count)
    batched[0] += offsets
    batched[1] += offsets
    return batched  # (2, E * B)


class StaticEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GNN_Temporal_Model(nn.Module):
    def __init__(
        self,
        edge_index,
        static_dim,
        dynamic_dim,
        future_dim,
        input_seq_len,
        forecast_horizon,
        model_type="GConvGRU",
        include_velocity_target: bool = False,
    ):
        super().__init__()
        self.edge_index = None
        self.static_encoder = StaticEncoder(static_dim)

        if model_type == "GConvGRU":
            self.recurrent = GConvGRU(32 + dynamic_dim, 32, K=2)
            self.future_recurrent = GConvGRU(32 + future_dim, 32, K=2)
        elif model_type == "TGCN":
            self.recurrent = TGCN(32 + dynamic_dim, 32)
            self.future_recurrent = TGCN(32 + future_dim, 32)
        elif model_type == "GConvLSTM":
            self.recurrent = GConvLSTM(32 + dynamic_dim, 32, K=2)
            self.future_recurrent = GConvLSTM(32 + future_dim, 32, K=2)
        elif model_type == "A3TGCN":
            # A3TGCN is used as a batched encoder over the full past sequence.
            # For future rollout we keep a recurrent decoder compatible with the
            # existing "known future covariates + hidden state" step-wise setup.
            self.recurrent = A3TGCN(
                in_channels=32 + dynamic_dim,
                out_channels=32,
                periods=input_seq_len,
            )
            self.future_recurrent = TGCN(32 + future_dim, 32)
        else:
            raise ValueError("Unknown model type")

        output_dim = 2 if include_velocity_target else 1
        self.linear = nn.Linear(32, output_dim)
        self.model_type = model_type

    def forward(
        self, edge_index, static_inputs, dynamic_seq_inputs, future_known_inputs
    ):
        # edge_index: (2, E), static_inputs: (N, Fstat)
        # dynamic_seq_inputs: (B, Tin, N, Fdyn), future_known_inputs: (B, H, N, Ff)
        B, Tin, N, Fdyn = dynamic_seq_inputs.shape
        H = future_known_inputs.shape[1]
        device = static_inputs.device

        # Precompute batched graph once
        if (
            self.edge_index is None
            or self.edge_index.shape[1] != edge_index.shape[1] * B
        ):
            self.edge_index = make_batched_edge_index(edge_index, N, B, device)

        # Static embedding, then tile across batch
        static_embed = self.static_encoder(static_inputs)  # (N, 32)
        static_embed = static_embed.unsqueeze(0).expand(B, N, 32)  # (B,N,32)

        # Hidden state (B,N,32) -> flatten to (B*N, 32)
        h = torch.zeros(B, N, 32, device=device)
        h_flat = h.reshape(B * N, 32)
        c_flat = None
        if self.model_type == "GConvLSTM":
            c_flat = h_flat.clone()

        # ---- Past sequence ----
        if self.model_type != "A3TGCN":
            for t in range(Tin):
                x_t = dynamic_seq_inputs[:, t]  # (B,N,Fdyn)
                combined = torch.cat([static_embed, x_t], -1)  # (B,N,32+Fdyn)
                x_flat = combined.reshape(B * N, 32 + Fdyn)  # (B*N, in)
                if self.model_type == "GConvLSTM":
                    h_flat, c_flat = self.recurrent(
                        x_flat, self.edge_index, H=h_flat, C=c_flat
                    )  # (B*N,32)
                else:
                    h_flat = self.recurrent(x_flat, self.edge_index, H=h_flat)  # (B*N,32)
        else:
            # A3TGCN expects node features for T periods. We encode each batch
            # item independently with the shared static graph, then switch back
            # to the existing batched step-wise decoder for future rollout.
            seq = []
            for b in range(B):
                combined = torch.cat([static_embed[b].unsqueeze(0).expand(Tin, N, 32), dynamic_seq_inputs[b]], dim=-1)
                x_b = combined.permute(1, 2, 0).contiguous()  # (N, in_channels, Tin)
                seq.append(self.recurrent(x_b, edge_index))
            h_flat = torch.stack(seq, dim=0).reshape(B * N, 32)

        # ---- Future steps ----
        outs = []
        for t in range(H):
            f_t = future_known_inputs[:, t]  # (B,N,Ff)
            combined_f = torch.cat([h_flat.view(B, N, 32), f_t], -1)  # (B,N,32+Ff)
            x_flat = combined_f.reshape(B * N, 32 + f_t.size(-1))
            if self.model_type == "GConvLSTM":
                h_flat, c_flat = self.future_recurrent(
                    x_flat, self.edge_index, H=h_flat, C=c_flat
                )
            else:
                h_flat = self.future_recurrent(x_flat, self.edge_index, H=h_flat)
            out_dim = self.linear.out_features
            out_t = self.linear(h_flat).view(B, N, out_dim)  # (B,N,out_dim)
            outs.append(out_t)

        return torch.stack(outs, dim=2)  # (B, N, H, out_dim)
