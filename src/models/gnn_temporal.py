
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, GConvLSTM, A3TGCN2

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
    def __init__(self, edge_index, static_dim, dynamic_dim, future_dim,
                 input_seq_len, forecast_horizon, model_type="GConvGRU"):
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
            self.recurrent = A3TGCN2(in_channels=32 + dynamic_dim,
                                     out_channels=32,
                                     periods=input_seq_len,
                                     kernel=2)
            self.future_recurrent = A3TGCN2(in_channels=32 + future_dim,
                                            out_channels=32,
                                            periods=forecast_horizon,
                                            kernel=2)
        else:
            raise ValueError("Unknown model type")

        self.linear = nn.Linear(32, 1)
        self.model_type = model_type

    def forward(self, edge_index, static_inputs, dynamic_seq_inputs, future_known_inputs):
        # edge_index: (2, E), static_inputs: (N, Fstat)
        # dynamic_seq_inputs: (B, Tin, N, Fdyn), future_known_inputs: (B, H, N, Ff)
        B, Tin, N, Fdyn = dynamic_seq_inputs.shape
        H = future_known_inputs.shape[1]
        device = static_inputs.device

        # Precompute batched graph once
        if (self.edge_index is None or
            self.edge_index.shape[1] != edge_index.shape[1] * B):
            self.edge_index = make_batched_edge_index(edge_index, N, B, device)

        # Static embedding, then tile across batch
        static_embed = self.static_encoder(static_inputs)        # (N, 32)
        static_embed = static_embed.unsqueeze(0).expand(B, N, 32)  # (B,N,32)

        # Hidden state (B,N,32) -> flatten to (B*N, 32)
        h = torch.zeros(B, N, 32, device=device)
        h_flat = h.reshape(B*N, 32)

        # ---- Past sequence ----
        if self.model_type != "A3TGCN":
            for t in range(Tin):
                x_t = dynamic_seq_inputs[:, t]                  # (B,N,Fdyn)
                combined = torch.cat([static_embed, x_t], -1)   # (B,N,32+Fdyn)
                x_flat = combined.reshape(B*N, 32+Fdyn)         # (B*N, in)
                h_flat = self.recurrent(x_flat, self.edge_index, H=h_flat)  # (B*N,32)
        else:
            # Build full (Tin, B*N, in) then reshape to (B*N, Tin, in) if your cell expects that
            seq = []
            for t in range(Tin):
                x_t = dynamic_seq_inputs[:, t]
                combined = torch.cat([static_embed, x_t], -1)   # (B,N,in)
                seq.append(combined.reshape(B*N, -1).unsqueeze(1))
            full_seq = torch.cat(seq, dim=1)                    # (B*N, Tin, in)
            h_flat = self.recurrent(full_seq, self.edge_index)  # -> (B*N,32) or similar

        # ---- Future steps ----
        outs = []
        for t in range(H):
            f_t = future_known_inputs[:, t]                     # (B,N,Ff)
            combined_f = torch.cat([h_flat.view(B, N, 32), f_t], -1)  # (B,N,32+Ff)
            x_flat = combined_f.reshape(B*N, 32 + f_t.size(-1))
            h_flat = self.future_recurrent(x_flat, self.edge_index, H=h_flat)
            out_t = self.linear(h_flat).view(B, N, 1)           # (B,N,1)
            outs.append(out_t)

        return torch.cat(outs, dim=-1)  # (B, N, H)                      # (B, N, H)
