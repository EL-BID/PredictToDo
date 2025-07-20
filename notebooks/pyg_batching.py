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
roads_gdf = roads_gdf.to_crs(epsg = 4326)
roads_gdf["id_mvi"] = roads_gdf["MVICIV"].astype(str) + "_" + roads_gdf["MVISVIA"].str.replace(" ", "")
roads_gdf['contador'] = roads_gdf.groupby('id_mvi').cumcount() + 1
roads_gdf['id_mvi'] = roads_gdf['id_mvi'] + '.' + roads_gdf['contador'].astype(str)
roads_gdf.set_index('id_mvi', inplace=True)
road_geom = roads_gdf.geometry
del roads_gdf  # We don't need the full GeoDataFrame anymore

# %%
ts_loader = GraphTimeSeriesLoader(
    graph_pkl_filename=f"{processed_data_dir}/conectivity_graph/road_graph.pickle",
    events_csv_filename=f"{processed_data_dir}/events/events_dataset.csv",
    incidents_csv_filename=f"{processed_data_dir}/incidents/incidents_dataset.csv",
    road_state_parquet_filename=f"{processed_data_dir}/road_state/road_state_dateset.parquet",
    irregularities_filenames=[f"{processed_data_dir}/irregularidades_procesadas/" + f for f in os.listdir(f"{processed_data_dir}/irregularidades_procesadas/") if f.endswith('.parquet')],
    coi_points_filenames=[f"{processed_data_dir}/pmts_processed/coi_puntos/" + f for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_puntos/") if f.endswith('.parquet')],
    coi_tracts_filenames=[f"{processed_data_dir}/pmts_processed/coi_tramos/" + f for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_tramos/") if f.endswith('.parquet')],
    default_speeds_parquet_filename=f"{processed_data_dir}/neighbor_features.parquet",
    jams_parquet_filenames=[f"{processed_data_dir}/jams_procesados/" + f for f in os.listdir(f"{processed_data_dir}/jams_procesados/") if f.endswith('.parquet')],
)

# %%
pyg_static_data = from_networkx(ts_loader.base_graph, group_node_attrs="all", group_edge_attrs="all")

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
m = road_geom.loc[sampled_nodes[0]].explore(style_kwds={"weight": 2.5, "color": "red"}, height=1500, width=1500)
road_geom.loc[sampled_nodes[1]].explore(m=m, style_kwds={"weight": 2.5, "color": "green"}, height=1500, width=1500)
road_geom.loc[sampled_nodes[2]].explore(m=m, style_kwds={"weight": 2.5, "color": "blue"}, height=1500, width=1500)
m

# %%
sampled_data = next(iter(neighbor_loader))

sampled_node_idxs = sampled_data.n_id.numpy()
sampled_nodes = all_nodes[sampled_node_idxs]

# Plot the sampled nodes on the world map
m = road_geom.loc[sampled_nodes].explore(style_kwds={"weight": 2.5, "color": "red"}, height=900, width=1500)
m
