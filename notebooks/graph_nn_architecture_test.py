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
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# %% [markdown]
# #### Prueba 1 - No Usar

# %%

# 1️⃣ Create your directed graph in NetworkX
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # simple cycle

# You can also add node features directly into the NetworkX nodes
for node in G.nodes():
    G.nodes[node]['x'] = torch.tensor([float(node), 1.0, 0.5])  # 3 features

# 2️⃣ Convert to torch_geometric.data.Data
data = from_networkx(G)

# If you want all node features stacked:
data.x = torch.stack([data.x[i] for i in range(data.num_nodes)])

# 3️⃣ Define a simple GNN model
class SimpleGRNN(torch.nn.Module):
    def __init__(self):
        super(SimpleGRNN, self).__init__()
        self.conv1 = GCNConv(3, 4)  # input dim = 3 (your node feature size)
        self.conv2 = GCNConv(4, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 4️⃣ Train or test the model
model = SimpleGRNN()
output = model(data)

print(output)

# %% [markdown]
# #### Prueba 2

# %%
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# Create directed graph
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

# Suppose we have 4 nodes
num_nodes = 4
static_features_dim = 20
dynamic_features_dim = 20

# Generate random static features (fixed per node)
static_features = {node: np.random.rand(static_features_dim) for node in G.nodes()}

# Generate random dynamic features at one time step (example t=0)
time_step = 0
dynamic_features = {node: np.random.rand(dynamic_features_dim) for node in G.nodes()}

# Combine both into node attributes for NetworkX
for node in G.nodes():
    node_features = np.concatenate([static_features[node], dynamic_features[node]])
    G.nodes[node]['x'] = torch.tensor(node_features, dtype=torch.float)

# Convert to PyTorch Geometric format
data = from_networkx(G)
data.x = torch.stack([data.x[i] for i in range(data.num_nodes)])

# Model: simple GCN
class GRNN(torch.nn.Module):
    def __init__(self):
        super(GRNN, self).__init__()
        self.conv1 = GCNConv(40, 64)  # 40 input features
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 2)  # output dimension 2 (for classification, etc.)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GRNN()
output = model(data)

print(output)

# %% [markdown]
# #### Prueba 3

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv

# === Problem parameters ===
num_nodes = 4            # number of nodes in the graph
static_dim = 20          # number of static features per node (do not change over time)
dynamic_dim = 20         # number of dynamic features per node per timestep (changing over time)
input_seq_len = 5        # how many historical time steps we use as input
forecast_horizon = 3     # how many future time steps we want to predict

# === Create graph topology (static graph structure) ===
G = nx.DiGraph()
G.add_edges_from([(0,1), (1,2), (2,3), (3,0)])  # simple 4-node cycle

# === Generate random static features for each node ===
# This simulates real static attributes (e.g. location, type, capacity, etc.)
static_features = {node: np.random.rand(static_dim) for node in G.nodes()}

# === Generate random dynamic historical features ===
# Shape: (input_seq_len, num_nodes, dynamic_dim)
# This simulates past measurements (e.g. traffic, power, temperature, etc.)
history_dynamic = np.random.rand(input_seq_len, num_nodes, dynamic_dim)

# === Model definitions ===

# --- Encoder for static features ---
class StaticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(static_dim, 32)  # compress static_dim → 32 dimensional embedding
    def forward(self, static_feat):
        return F.relu(self.fc(static_feat))

# --- Encoder for graph (spatial structure) ---
class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(32 + dynamic_dim, 64)  # input: concatenated static+dynamic features
        self.conv2 = GCNConv(64, 32)                # output: spatial embedding per node
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))       # first GCN layer
        x = F.relu(self.conv2(x, edge_index))       # second GCN layer
        return x  # output: (num_nodes, 32)

# --- Full model combining GNN (spatial) + GRU (temporal) ---
class GNN_GRU_Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_encoder = StaticEncoder()       # encoder for static features
        self.gnn = GNNEncoder()                     # GNN to encode spatial relations
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)  # GRU for temporal sequence modeling
        self.linear = nn.Linear(64, forecast_horizon)  # final output layer → 1 scalar per future timestep

    def forward(self, static_inputs, seq_dynamic_inputs, graph):
        # 1️⃣ First, compute static embeddings once (no need to recompute for every timestep)
        static_embeddings = {}
        for node in graph.nodes():
            static_tensor = torch.tensor(static_inputs[node], dtype=torch.float)
            static_embeddings[node] = self.static_encoder(static_tensor)

        # 2️⃣ Process each time step of dynamic data
        node_embeddings = []
        for t in range(seq_dynamic_inputs.shape[0]):  # for each timestep in the input sequence
            for node in graph.nodes():
                dynamic_tensor = torch.tensor(seq_dynamic_inputs[t, node], dtype=torch.float)
                # Concatenate static embedding + current dynamic features
                combined = torch.cat([static_embeddings[node], dynamic_tensor], dim=0)
                graph.nodes[node]['x'] = combined  # assign node features for this timestep

            # Convert updated NetworkX graph into torch_geometric format
            data = from_networkx(graph)
            data.x = torch.stack([data.x[i] for i in range(data.num_nodes)])

            # Pass through GNN to get node embeddings for this timestep
            node_embed = self.gnn(data)  # (num_nodes, 32)
            node_embeddings.append(node_embed.unsqueeze(0))  # (1, num_nodes, 32)

        # 3️⃣ After all timesteps processed, stack into a temporal sequence tensor
        node_embeddings = torch.cat(node_embeddings, dim=0)  # (input_seq_len, num_nodes, 32)
        node_embeddings = node_embeddings.permute(1, 0, 2)   # (num_nodes, input_seq_len, 32)

        # 4️⃣ For each node, process its temporal sequence independently through the GRU
        outputs = []
        for node_seq in node_embeddings:  # node_seq: (input_seq_len, 32)
            gru_out, _ = self.gru(node_seq.unsqueeze(0))  # (1, input_seq_len, 64)
            pred = self.linear(gru_out[:, -1, :])         # take last hidden state and predict future horizon
            outputs.append(pred)                          # (1, forecast_horizon)

        # 5️⃣ Stack predictions for all nodes
        outputs = torch.stack(outputs, dim=0)  # (num_nodes, 1, forecast_horizon)
        outputs = outputs.squeeze(1)           # (num_nodes, forecast_horizon)
        return outputs

# === Instantiate and run the model ===

model = GNN_GRU_Predictor()

# Call model with:
# - static_features: dictionary {node: static array}
# - history_dynamic: numpy array of shape (input_seq_len, num_nodes, dynamic_dim)
# - G: the NetworkX graph
forecast = model(static_features, history_dynamic, G)

print(forecast.shape)
# Output shape: (num_nodes, forecast_horizon)
# => for each node, predict 'forecast_horizon' future steps

# %%
loss_fn = nn.MSELoss()
loss = loss_fn(predictions, targets)

# %%
for epoch in range(num_epochs):
    for static_inputs, dynamic_seq_inputs, targets in dataloader:
        optimizer.zero_grad()
        forecast = model(static_inputs, dynamic_seq_inputs, graph)
        loss = loss_fn(forecast, targets)
        loss.backward()
        optimizer.step()


# %%
def custom_loss(predictions, targets, threshold=0.1, alpha=1.0):
    # Base MSE loss
    base_loss = torch.mean((predictions - targets)**2)

    # Compute per-node average error
    abs_error = torch.abs(predictions - targets)
    node_error = torch.mean(abs_error, dim=1)  # [num_nodes]

    # Violation count
    violations = (node_error > threshold).float()
    violation_ratio = torch.mean(violations)

    # Total loss
    total_loss = base_loss + alpha * violation_ratio
    return total_loss


# %% [markdown]
# ##### Version torch geometric temporal manual

# %% [markdown]
# #### Entranamiento Arquitectura intercambiable entre tipos, arquitectura final para realizar pruebas de entrenamiento de los 4 tipos que se utilizaran en el proyecto

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, GConvLSTM, A3TGCN2

# === Parameters ===
num_nodes = 4  # number of nodes in the graph
static_dim = 20  # static features per node
dynamic_dim = 20  # dynamic features per node per timestep
input_seq_len = 5  # historical time steps used as input
forecast_horizon = 3  # future time steps to predict

# === Build static graph ===
G = nx.DiGraph()  # create directed graph
G.add_edges_from([(0,1), (1,2), (2,3), (3,0)])  # define edges
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()  # convert edges to PyTorch Geometric format

# === Simulate full time-series data ===
total_timesteps = 100  # total simulated time steps
static_features = np.random.rand(num_nodes, static_dim).astype(np.float32)  # static node features
dynamic_features = np.random.rand(total_timesteps, num_nodes, dynamic_dim).astype(np.float32)  # dynamic time series features
target_series = np.random.rand(total_timesteps, num_nodes).astype(np.float32)  # target values to predict

# === Dataset class ===
class TemporalGraphDataset(Dataset):
    def __init__(self, static_features, dynamic_features, target_series, input_seq_len, forecast_horizon):
        super().__init__()
        self.static_features = torch.tensor(static_features)
        self.dynamic_features = torch.tensor(dynamic_features)
        self.target_series = torch.tensor(target_series)
        self.input_seq_len = input_seq_len
        self.forecast_horizon = forecast_horizon
        self.length = total_timesteps - input_seq_len - forecast_horizon + 1  # number of samples

    def len(self):
        return self.length

    def get(self, idx):
        dynamic_seq = self.dynamic_features[idx:idx+self.input_seq_len]  # dynamic input sequence
        target = self.target_series[idx+self.input_seq_len : idx+self.input_seq_len+self.forecast_horizon]  # corresponding target
        target = target.permute(1, 0)  # rearrange to (num_nodes, forecast_horizon)
        return self.static_features, dynamic_seq, target

# Create dataset and dataloader
dataset = TemporalGraphDataset(static_features, dynamic_features, target_series, input_seq_len, forecast_horizon)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# === Static encoder ===
class StaticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(static_dim, 32)  # map static features to 32-dim embedding
    def forward(self, static_feat):
        return F.relu(self.fc(static_feat))

# === General model framework ===
class GNN_Temporal_Model(nn.Module):
    def __init__(self, edge_index, model_type="GConvGRU"):
        super().__init__()
        self.edge_index = edge_index
        self.static_encoder = StaticEncoder()

        # Choose recurrent model type
        if model_type == "GConvGRU":
            self.recurrent = GConvGRU(32 + dynamic_dim, 32, K=2)
        elif model_type == "TGCN":
            self.recurrent = TGCN(32 + dynamic_dim, 32)
        elif model_type == "GConvLSTM":
            self.recurrent = GConvLSTM(32 + dynamic_dim, 32, K=2)
        elif model_type == "A3TGCN":
            self.recurrent = A3TGCN2(in_channels=32 + dynamic_dim, out_channels=32, periods=input_seq_len, kernel=2)
        else:
            raise ValueError("Unknown model type")

        self.linear = nn.Linear(32, forecast_horizon)  # output prediction layer
        self.model_type = model_type

    def forward(self, static_inputs, dynamic_seq_inputs):
        batch_size = dynamic_seq_inputs.shape[0]  # batch size
        num_nodes = static_inputs.shape[0]  # number of nodes

        # Encode static features
        static_embed = self.static_encoder(static_inputs)
        static_embed = static_embed.unsqueeze(0).repeat(batch_size, 1, 1)  # repeat for batch

        h = torch.zeros(batch_size, num_nodes, 32)  # initialize hidden states

        if self.model_type != "A3TGCN":  # recurrent models (GConvGRU, TGCN, GConvLSTM)
            for t in range(dynamic_seq_inputs.shape[1]):
                dynamic_t = dynamic_seq_inputs[:, t]
                combined = torch.cat([static_embed[:, :, :], dynamic_t], dim=-1)  # concatenate static+dynamic features
                for b in range(batch_size):
                    h[b] = self.recurrent(combined[b], self.edge_index, h[b])  # recurrent update
        else:  # A3TGCN processes full sequence at once
            full_seq = []
            for t in range(dynamic_seq_inputs.shape[1]):
                dynamic_t = dynamic_seq_inputs[:, t]
                combined = torch.cat([static_embed[:, :, :], dynamic_t], dim=-1)
                full_seq.append(combined.unsqueeze(1))
            full_seq = torch.cat(full_seq, dim=1)
            for b in range(batch_size):
                h[b] = self.recurrent(full_seq[b], self.edge_index)

        out = self.linear(h)  # final linear projection to predict horizon
        return out

# === Training loop ===
for model_type in ["GConvGRU", "TGCN", "GConvLSTM", "A3TGCN"]:
    print(f"\nTraining model: {model_type}")
    model = GNN_Temporal_Model(edge_index, model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        total_loss = 0
        for static_inputs, dynamic_seq_inputs, target in dataloader:
            optimizer.zero_grad()
            predictions = model(static_inputs, dynamic_seq_inputs)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, GConvLSTM, A3TGCN2

# === Static encoder ===
class StaticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(static_dim, 32)  # map static features to 32-dim embedding
    def forward(self, static_feat):
        return F.relu(self.fc(static_feat))

# === General model framework ===
class GNN_Temporal_Model(nn.Module):
    def __init__(self, edge_index, model_type="GConvGRU"):
        super().__init__()
        self.edge_index = edge_index
        self.static_encoder = StaticEncoder()

        # Choose recurrent model type
        if model_type == "GConvGRU":
            self.recurrent = GConvGRU(32 + dynamic_dim, 32, K=2)
        elif model_type == "TGCN":
            self.recurrent = TGCN(32 + dynamic_dim, 32)
        elif model_type == "GConvLSTM":
            self.recurrent = GConvLSTM(32 + dynamic_dim, 32, K=2)
        elif model_type == "A3TGCN":
            self.recurrent = A3TGCN2(in_channels=32 + dynamic_dim, out_channels=32, periods=input_seq_len, kernel=2)
        else:
            raise ValueError("Unknown model type")

        self.linear = nn.Linear(32, forecast_horizon)  # output prediction layer
        self.model_type = model_type

    def forward(self, static_inputs, dynamic_seq_inputs):
        batch_size = dynamic_seq_inputs.shape[0]  # batch size
        num_nodes = static_inputs.shape[0]  # number of nodes

        # Encode static features
        static_embed = self.static_encoder(static_inputs)
        static_embed = static_embed.unsqueeze(0).repeat(batch_size, 1, 1)  # repeat for batch

        h = torch.zeros(batch_size, num_nodes, 32)  # initialize hidden states

        if self.model_type != "A3TGCN":  # recurrent models (GConvGRU, TGCN, GConvLSTM)
            for t in range(dynamic_seq_inputs.shape[1]):
                dynamic_t = dynamic_seq_inputs[:, t]
                combined = torch.cat([static_embed[:, :, :], dynamic_t], dim=-1)  # concatenate static+dynamic features
                for b in range(batch_size):
                    h[b] = self.recurrent(combined[b], self.edge_index, h[b])  # recurrent update
        else:  # A3TGCN processes full sequence at once
            full_seq = []
            for t in range(dynamic_seq_inputs.shape[1]):
                dynamic_t = dynamic_seq_inputs[:, t]
                combined = torch.cat([static_embed[:, :, :], dynamic_t], dim=-1)
                full_seq.append(combined.unsqueeze(1))
            full_seq = torch.cat(full_seq, dim=1)
            for b in range(batch_size):
                h[b] = self.recurrent(full_seq[b], self.edge_index)

        out = self.linear(h)  # final linear projection to predict horizon
        return out

# === Training loop ===
for model_type in ["GConvGRU", "TGCN", "GConvLSTM", "A3TGCN"]:
    print(f"\nTraining model: {model_type}")
    model = GNN_Temporal_Model(edge_index, model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        total_loss = 0
        for static_inputs, dynamic_seq_inputs, target in dataloader:
            optimizer.zero_grad()
            predictions = model(static_inputs, dynamic_seq_inputs)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, GConvLSTM, A3TGCN2

# === Static encoder ===
class StaticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(static_dim, 32)

    def forward(self, static_feat):
        return F.relu(self.fc(static_feat))

# === General model framework ===
class GNN_Temporal_Model(nn.Module):
    def __init__(self, edge_index, model_type="GConvGRU"):
        super().__init__()
        self.edge_index = edge_index
        self.static_encoder = StaticEncoder()

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
            self.recurrent = A3TGCN2(in_channels=32 + dynamic_dim, out_channels=32, periods=input_seq_len, kernel=2)
            self.future_recurrent = A3TGCN2(in_channels=32 + future_dim, out_channels=32, periods=forecast_horizon, kernel=2)
        else:
            raise ValueError("Unknown model type")

        self.linear = nn.Linear(32, 1)
        self.model_type = model_type

    def forward(self, static_inputs, dynamic_seq_inputs, future_known_inputs):
        B, T_in, N, D = dynamic_seq_inputs.shape
        H = future_known_inputs.shape[1]  # forecast horizon

        static_embed = self.static_encoder(static_inputs)  # [N, 32]
        static_embed = static_embed.unsqueeze(0).repeat(B, 1, 1)  # [B, N, 32]

        h = torch.zeros(B, N, 32, device=static_inputs.device)

        if self.model_type != "A3TGCN":
            for t in range(T_in):
                x_t = dynamic_seq_inputs[:, t]  # [B, N, D]
                combined = torch.cat([static_embed, x_t], dim=-1)  # [B, N, 32+D]
                for b in range(B):
                    h[b] = self.recurrent(combined[b], self.edge_index, h[b])
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
                h[b] = self.future_recurrent(combined_f[b], self.edge_index, h[b])
            out_t = self.linear(h)  # [B, N, 1]
            outputs.append(out_t)

        return torch.cat(outputs, dim=-1)  # [B, N, H]

# === Training loop ===
for model_type in ["GConvGRU", "TGCN", "GConvLSTM", "A3TGCN"]:
    print(f"\nTraining model: {model_type}")
    model = GNN_Temporal_Model(edge_index, model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        total_loss = 0
        for static_inputs, dynamic_seq_inputs, future_inputs, target in dataloader:
            optimizer.zero_grad()
            predictions = model(static_inputs, dynamic_seq_inputs, future_inputs)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")


# %%

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, GConvLSTM, A3TGCN2

# === Static encoder ===
class StaticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(static_dim, 32)

    def forward(self, static_feat):
        return F.relu(self.fc(static_feat))

# === General model framework ===
class GNN_Temporal_Model(nn.Module):
    def __init__(self, edge_index, model_type="GConvGRU"):
        super().__init__()
        self.edge_index = edge_index
        self.static_encoder = StaticEncoder()

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
            self.recurrent = A3TGCN2(in_channels=32 + dynamic_dim, out_channels=32, periods=input_seq_len, kernel=2)
            self.future_recurrent = A3TGCN2(in_channels=32 + future_dim, out_channels=32, periods=forecast_horizon, kernel=2)
        else:
            raise ValueError("Unknown model type")

        self.linear = nn.Linear(32, 1)
        self.model_type = model_type

    def forward(self, static_inputs, dynamic_seq_inputs, future_known_inputs):
        B, T_in, N, D = dynamic_seq_inputs.shape
        H = future_known_inputs.shape[1]  # forecast horizon

        static_embed = self.static_encoder(static_inputs)  # [N, 32]
        static_embed = static_embed.unsqueeze(0).repeat(B, 1, 1)  # [B, N, 32]

        h = torch.zeros(B, N, 32, device=static_inputs.device)

        if self.model_type != "A3TGCN":
            for t in range(T_in):
                x_t = dynamic_seq_inputs[:, t]  # [B, N, D]
                combined = torch.cat([static_embed, x_t], dim=-1)  # [B, N, 32+D]
                for b in range(B):
                    h[b] = self.recurrent(combined[b], self.edge_index, h[b])
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
                h[b] = self.future_recurrent(combined_f[b], self.edge_index, h[b])
            out_t = self.linear(h)  # [B, N, 1]
            outputs.append(out_t)

        return torch.cat(outputs, dim=-1)  # [B, N, H]

# === Training loop ===
for model_type in ["GConvGRU", "TGCN", "GConvLSTM", "A3TGCN"]:
    print(f"\nTraining model: {model_type}")
    model = GNN_Temporal_Model(edge_index, model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        total_loss = 0
        for static_inputs, dynamic_seq_inputs, future_inputs, target in dataloader:
            optimizer.zero_grad()
            predictions = model(static_inputs, dynamic_seq_inputs, future_inputs)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

