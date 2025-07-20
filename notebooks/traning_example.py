# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: congestion-bogota
#     language: python
#     name: python3
# ---

# %%
# cd ..

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, GConvLSTM, A3TGCN2
import os
from src.dataloader import GraphTimeSeriesDataset, GraphTimeSeriesLoader


# %%
# === Static encoder ===
class StaticEncoder(nn.Module):
    def __init__(self, static_dim):
        super().__init__()
        self.fc = nn.Linear(static_dim, 32)

    def forward(self, static_feat):
        return F.relu(self.fc(static_feat))


# === General model framework ===
class GNN_Temporal_Model(nn.Module):
    def __init__(self, edge_index, static_dim, dynamic_dim, future_dim,
                 input_seq_len, forecast_horizon, model_type="GConvGRU"):
        super().__init__()
        self.edge_index = edge_index
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

    def forward(self, static_inputs, dynamic_seq_inputs, future_known_inputs):
        B, T_in, N, D = dynamic_seq_inputs.shape
        H = future_known_inputs.shape[1]  # forecast horizon

        # Encode static features
        static_embed = self.static_encoder(static_inputs)  # [N, 32]
        static_embed = static_embed.unsqueeze(0).repeat(B, 1, 1)  # [B, N, 32]

        # Initialize hidden state
        h = torch.zeros(B, N, 32, device=static_inputs.device)

        # === Past sequence processing ===
        if self.model_type != "A3TGCN":
            for t in range(T_in):
                x_t = dynamic_seq_inputs[:, t]  # [B, N, D]
                combined = torch.cat([static_embed, x_t], dim=-1)  # [B, N, 32+D]
                for b in range(B):
                    h[b] = self.recurrent(combined[b], self.edge_index, H=h[b])
        else:
            full_seq = []
            for t in range(T_in):
                x_t = dynamic_seq_inputs[:, t]
                combined = torch.cat([static_embed, x_t], dim=-1)
                full_seq.append(combined.unsqueeze(1))
            full_seq = torch.cat(full_seq, dim=1)
            for b in range(B):
                h[b] = self.recurrent(full_seq[b], self.edge_index)

        # === Future steps recurrent modeling ===
        outputs = []
        for t in range(H):
            f_t = future_known_inputs[:, t]  # [B, N, future_dim]
            combined_f = torch.cat([h, f_t], dim=-1)  # [B, N, 32+future_dim]
            for b in range(B):
                h[b] = self.future_recurrent(combined_f[b], self.edge_index, H=h[b])
            out_t = self.linear(h)  # [B, N, 1]
            outputs.append(out_t)

        return torch.cat(outputs, dim=-1)  # [B, N, H]


# %%
import os
processed_data_dir = "data/datos_procesados"
loader = GraphTimeSeriesLoader(
    graph_pkl_filename=f"{processed_data_dir}/conectivity_graph/road_graph.pickle",
    events_csv_filename=f"{processed_data_dir}/events/events_dataset.csv",
    incidents_csv_filename=f"{processed_data_dir}/incidents/incidents_dataset.csv",
    road_state_parquet_filename=f"{processed_data_dir}/road_state/road_state_dateset.parquet",
    irregularities_filenames=[f"{processed_data_dir}/irregularidades_procesadas/" + f for f in os.listdir(f"{processed_data_dir}/irregularidades_procesadas/") if f.endswith('.parquet')],
    coi_points_filenames=[f"{processed_data_dir}/pmts_processed/coi_puntos/" + f for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_puntos/") if f.endswith('.parquet')],
    coi_tracts_filenames=[f"{processed_data_dir}/pmts_processed/coi_tramos/" + f for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_tramos/") if f.endswith('.parquet')],
    default_speeds_parquet_filename=f"{processed_data_dir}/neighbor_features.parquet",
    dynamic_columns=f"{processed_data_dir}/columns_dict.pkl"
)

# %%
all_nodes = list(loader.base_graph.nodes())
# crea un mapping de tus node_ids a índices
edge_index_full = from_networkx(loader.base_graph).edge_index

# %%
nodes_to_load = all_nodes[:500]
nodes_to_load = [str(n) for n in nodes_to_load]
time_windows_to_load = [
    ('2024-09-29 23:00:00', 96, 288),
    ('2024-08-24 15:30:00', 96, 288),
    ('2024-02-24 01:30:00', 96, 288)
]

# Map original node ids to compact indices
node_to_idx = {nid: i for i, nid in enumerate(nodes_to_load)}

# Build mapping from NetworkX enumeration (used by from_networkx) to original node IDs
ordered_nodes = list(loader.base_graph.nodes())
nx_index_to_node = {i: nid for i, nid in enumerate(ordered_nodes)}

src = []
dst = []

for u, v in zip(edge_index_full[0], edge_index_full[1]):
    u_id = nx_index_to_node[u.item()]
    v_id = nx_index_to_node[v.item()]
    if u_id in node_to_idx and v_id in node_to_idx:
        src.append(node_to_idx[u_id])  # remap to compact index
        dst.append(node_to_idx[v_id])  # remap to compact index

edge_index_filtered = torch.tensor([src, dst], dtype=torch.long)

# %%
print("edge_index_filtered max index:", edge_index_filtered.max().item())
print("edge_index_filtered shape:", edge_index_filtered.shape)
print("Max edge_index_filtered:", edge_index_filtered.max().item())
print("Num nodes (N):", len(nodes_to_load))

# %%
sample_data = GraphTimeSeriesDataset(
    loader=loader,
    node_ids=nodes_to_load,
    time_windows=time_windows_to_load
)

# %%
# Infer dimensions from one batch
static_tensor, past_inputs, future_inputs, target = sample_data[0]
static_dim = static_tensor.shape[1]
dynamic_dim = past_inputs.shape[-1]
future_dim = future_inputs.shape[-1]

input_seq_len = 288
forecast_horizon = 96

# Reload dataset with batch loader
train_dataloader = DataLoader(sample_data, batch_size=2, shuffle=False)

# %%
# for model_type in ["GConvGRU", "TGCN", "GConvLSTM", "A3TGCN"]:
model_type = "GConvGRU"
print(f"\nTraining model: {model_type}")
model = GNN_Temporal_Model(
    edge_index=edge_index_filtered,
    model_type=model_type,
    static_dim=static_dim,
    dynamic_dim=dynamic_dim,
    future_dim=future_dim,
    input_seq_len=input_seq_len,
    forecast_horizon=forecast_horizon
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(5):
    total_loss = 0
    for static_inputs, dynamic_seq_inputs, future_inputs, target in train_dataloader:
        print(f"    Static data shape: {static_inputs.shape}")
        print(f"    Past data shape:   {dynamic_seq_inputs.shape}")
        print(f"    Future data shape: {future_inputs.shape}")
        print(f"    Target data shape: {target.shape}")
        print("-" * 20)
        static_batch = static_inputs[0].squeeze(0)
        optimizer.zero_grad()
        predictions = model(static_batch, dynamic_seq_inputs, future_inputs)
        loss = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader):.4f}")

# %%
static_batch.shape
dynamic_seq_inputs.shape
future_inputs.shape

# %%

# %%
