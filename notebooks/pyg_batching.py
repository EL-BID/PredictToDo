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
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.nn.recurrent import A3TGCN2
from torch_geometric_temporal.signal import temporal_signal_split

shuffle = True
batch_size = 32
DEVICE = "cpu"
num_timesteps_in = 10
num_timesteps_out = 3
loader = METRLADatasetLoader(raw_data_dir=os.path.join(os.getcwd(), "notebooks/data"))

# %load_ext autoreload
# %autoreload 2

# %%
dataset = loader.get_dataset(
    num_timesteps_in=num_timesteps_in, num_timesteps_out=num_timesteps_out
)

# %%
# self.edge_index = edge_index
# self.edge_weight = edge_weight
# self.features = features
# self.targets = targets
# self.additional_feature_keys

# %%
dataset.edge_index.max()

# %%
print("Shape of edge_index:", np.array(dataset.edge_index).shape)
# print("Shape of edge_weight:", np.array(dataset.edge_weight).shape)
print("Shape of features:", np.array(dataset.features).shape)
print("Shape of targets:", np.array(dataset.targets).shape)
print("Additional feature keys:", dataset.additional_feature_keys)

# %%
print("Shape of edge_index:", np.array(dataset.edge_index).shape)
# print("Shape of edge_weight:", np.array(dataset.edge_weight).shape)
print("Shape of features:", np.array(dataset.features).shape)
print("Shape of targets:", np.array(dataset.targets).shape)
print("Additional feature keys:", dataset.additional_feature_keys)

# %%
# Visualize traffic over time
sensor_number = 1
hours = 24
sensor_labels = [bucket.y[sensor_number][0].item() for bucket in list(dataset)[:hours]]
plt.plot(sensor_labels)
plt.title(f"Traffic at Sensor {sensor_number} Over Time")
plt.xlabel("Time (hours)")
plt.ylabel("Traffic Volume")
plt.show()

# %%
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

print("Number of train buckets: ", len(list(train_dataset)))
print("Number of test buckets: ", len(list(test_dataset)))

# %%
train_input = np.array(
    train_dataset.features
)  # (num_train_buckets, num_nodes, num_features_per_node, num_timesteps_in)
train_target = np.array(
    train_dataset.targets
)  # (num_test_buckets, num_nodes, num_timesteps_out)
train_x_tensor = (
    torch.from_numpy(train_input).type(torch.FloatTensor).to(DEVICE)
)  # (B, N, F, T)
train_target_tensor = (
    torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)
)  # (B, N, T)
train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True
)

# %%
test_input = np.array(
    test_dataset.features
)  # (num_test_buckets, num_nodes, num_features_per_node, num_timesteps_out)
test_target = np.array(
    test_dataset.targets
)  # (num_test_buckets, num_nodes, num_timesteps_out)
test_x_tensor = (
    torch.from_numpy(test_input).type(torch.FloatTensor).to(DEVICE)
)  # (B, N, F, T)
test_target_tensor = (
    torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)
)  # (B, N, T)
test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(
    test_dataset_new, batch_size=batch_size, shuffle=shuffle, drop_last=True
)


# %%
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN2(
            in_channels=node_features,
            out_channels=32,
            periods=periods,
            batch_size=batch_size,
        )  # periods = T_out
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index)  # x [B, N, F, T_in]  returns h [B, N, T_out]
        h = F.relu(h)
        h = self.linear(h)
        return h


# %%
# Create model and optimizers
model = TemporalGNN(
    node_features=2, periods=num_timesteps_out, batch_size=batch_size
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

print("Net's state_dict:")
total_param = 0
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    total_param += np.prod(model.state_dict()[param_tensor].size())
print("Net's total params:", total_param)
# --------------------------------------------------
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

# %%
for snapshot in train_dataset:
    static_edge_index = snapshot.edge_index.to(DEVICE)
    break
print(snapshot)

# %%
model.train()

for epoch in range(1):
    step = 0
    loss_list = []
    for encoder_inputs, labels in train_loader:
        y_hat = model(encoder_inputs, static_edge_index)  # Get model predictions
        loss = loss_fn(
            y_hat, labels
        )  # Mean squared error #loss = torch.mean((y_hat-labels)**2)  sqrt to change it to rmse
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step = step + 1
        loss_list.append(loss.item())
        if step % 100 == 0:
            print(sum(loss_list) / len(loss_list))
    print("Epoch {} train RMSE: {:.4f}".format(epoch, sum(loss_list) / len(loss_list)))

# %%
model.eval()

step = 0
# Store for analysis
total_loss = []
for encoder_inputs, labels in test_loader:
    # Get model predictions
    y_hat = model(encoder_inputs, static_edge_index)
    # Mean squared error
    loss = loss_fn(y_hat, labels)
    total_loss.append(loss.item())
    # Store for analysis below
    # test_labels.append(labels)
    # predictions.append(y_hat)


print("Test MSE: {:.4f}".format(sum(total_loss) / len(total_loss)))

# %%
sensor = 123
timestep = 5
preds = np.asarray([pred[sensor][timestep].detach().cpu().numpy() for pred in y_hat])
labs = np.asarray([label[sensor][timestep].cpu().numpy() for label in labels])
print("Data points:,", preds.shape)

plt.figure(figsize=(20, 5))
sns.lineplot(data=preds, label="Predicted")
sns.lineplot(data=labs, label="True")
plt.show()

# %% [markdown]
# ## Neighbor Batching

# %%
import os

import numpy as np
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.loader import NeighborLoader, DataLoader
from torch_geometric_temporal.nn.recurrent import GConvGRU, TGCN, GConvLSTM, A3TGCN2

from src.dataloader import GraphTimeSeriesDataset, GraphTimeSeriesLoader

processed_data_dir = "data/datos_procesados"
roads_filename = "data/2_Estructura_via/estructura_via_completa/MVI.geojson"

# Download the world map data for visualization
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)

# %%
# Load the roads geometry for plotting
roads_gdf = gpd.read_file(roads_filename)
roads_gdf = roads_gdf.to_crs(epsg=4326)
roads_gdf["id_mvi"] = (
    roads_gdf["MVICIV"].astype(str) + "_" + roads_gdf["MVISVIA"].str.replace(" ", "")
)
roads_gdf["contador"] = roads_gdf.groupby("id_mvi").cumcount() + 1
roads_gdf["id_mvi"] = roads_gdf["id_mvi"] + "." + roads_gdf["contador"].astype(str)
roads_gdf.set_index("id_mvi", inplace=True)
road_geom = roads_gdf.geometry
del roads_gdf  # We don't need the full GeoDataFrame anymore

# %%
ts_loader = GraphTimeSeriesLoader(
    graph_pkl_filename=f"{processed_data_dir}/conectivity_graph/road_graph.pickle",
    events_csv_filename=f"{processed_data_dir}/events/events_dataset.csv",
    incidents_csv_filename=f"{processed_data_dir}/incidents/incidents_dataset.csv",
    road_state_parquet_filename=f"{processed_data_dir}/road_state/road_state_dateset.parquet",
    irregularities_filenames=[
        f"{processed_data_dir}/irregularidades_procesadas/" + f
        for f in os.listdir(f"{processed_data_dir}/irregularidades_procesadas/")
        if f.endswith(".parquet")
    ],
    coi_points_filenames=[
        f"{processed_data_dir}/pmts_processed/coi_puntos/" + f
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_puntos/")
        if f.endswith(".parquet")
    ],
    coi_tracts_filenames=[
        f"{processed_data_dir}/pmts_processed/coi_tramos/" + f
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_tramos/")
        if f.endswith(".parquet")
    ],
    default_speeds_parquet_filename=f"{processed_data_dir}/neighbor_features.parquet",
    jams_parquet_filenames=[
        f"{processed_data_dir}/jams_procesados/" + f
        for f in os.listdir(f"{processed_data_dir}/jams_procesados/")
        if f.endswith(".parquet")
    ],
)

# %%
pyg_static_data = from_networkx(
    ts_loader.base_graph, group_node_attrs="all", group_edge_attrs="all"
)

# %%
k_hops = 20
neighborhoods_per_batch = 1

neighbor_loader = NeighborLoader(
    pyg_static_data,
    # Sample all neighbors for each node for k iterations
    # This will return all nodes at k-hops distance
    num_neighbors=[-1] * k_hops,
    batch_size=neighborhoods_per_batch,
    replace=False,
)

all_nodes = np.array(ts_loader.base_graph.nodes())
loader_iterator = iter(neighbor_loader)

# %%
# Get the first 3 batches for visualization
sampled_nodes = []
for _ in range(3):
    sampled_data = next(loader_iterator)
    sampled_node_idxs = sampled_data.n_id.numpy()
    sampled_nodes.append(all_nodes[sampled_node_idxs])

# Plot the sampled nodes on the world map
m = road_geom.loc[sampled_nodes[0]].explore(
    style_kwds={"weight": 2.5, "color": "red"}, height=1500, width=1500
)
road_geom.loc[sampled_nodes[1]].explore(
    m=m, style_kwds={"weight": 2.5, "color": "green"}, height=1500, width=1500
)
road_geom.loc[sampled_nodes[2]].explore(
    m=m, style_kwds={"weight": 2.5, "color": "blue"}, height=1500, width=1500
)
m

# %%
sampled_data = next(iter(neighbor_loader))

sampled_node_idxs = sampled_data.n_id.numpy()
sampled_nodes = all_nodes[sampled_node_idxs]

# Plot the sampled nodes on the world map
m = road_geom.loc[sampled_nodes].explore(
    style_kwds={"weight": 2.5, "color": "red"}, height=900, width=1500
)
m
