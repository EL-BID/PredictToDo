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
import pandas as pd
import networkx as nx
import numpy as np
import pickle as pkl
import geopandas as gpd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2

# %%
processed_data_dir = "../data/datos_procesados"
roads_filename = "../data/2_Estructura_via/estructura_via_completa/MVI.geojson"

irregularities_files = [
    f"{processed_data_dir}/irregularidades_procesadas/" + f
    for f in os.listdir(f"{processed_data_dir}/irregularidades_procesadas/")
    if f.endswith(".parquet")
]
jams_files = [
    f"{processed_data_dir}/jams_procesados/" + f
    for f in os.listdir(f"{processed_data_dir}/jams_procesados/")
    if f.endswith(".parquet")
]
graph_filename = f"{processed_data_dir}/conectivity_graph/road_graph.pickle"

# Download the world map data for visualization
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)

# Load the base graph
with open(graph_filename, "rb") as f:
    base_graph = pkl.load(f)
node_ids = list(base_graph.nodes())

# Load the roads GeoDataFrame
roads_gdf = gpd.read_file(roads_filename)
roads_gdf = roads_gdf.to_crs(epsg=4326)
roads_gdf["id_mvi"] = (
    roads_gdf["MVICIV"].astype(str) + "_" + roads_gdf["MVISVIA"].str.replace(" ", "")
)
roads_gdf["contador"] = roads_gdf.groupby("id_mvi").cumcount() + 1
roads_gdf["id_mvi"] = roads_gdf["id_mvi"] + "." + roads_gdf["contador"].astype(str)
roads_gdf = roads_gdf.explode(ignore_index=True).dissolve(by="id_mvi").reset_index()
roads_gdf["road_id"] = roads_gdf["id_mvi"]
roads_gdf["buffer"] = roads_gdf.geometry.buffer(0.01)
roads_gdf.set_index("id_mvi", inplace=True)
road_geom = roads_gdf.geometry
del roads_gdf  # We don't need the full GeoDataFrame anymore

# Load all the irregularities data
use_cols = ["id_mvi", "timeframe", "trend", "speed", "regularSpeed"]
irregularities_df = pd.concat(
    [pd.read_parquet(f) for f in irregularities_files], ignore_index=True
)
jams_df = pd.concat([pd.read_parquet(f) for f in jams_files], ignore_index=True)
# Compute the mean features for each road segment
mean_features = irregularities_df.groupby("id_mvi")[["trend", "regularSpeed"]].mean()

# set(mean_features.index.tolist()) < set(base_graph.nodes())

# %%
# Compute the disconnected nodes
components = [
    (len(c), base_graph.subgraph(c).copy())
    for c in nx.connected_components(base_graph.to_undirected())
]
components.sort(key=lambda x: x[0], reverse=True)
disconnected_nodes = set(
    [node for component in components[1:] for node in component[1].nodes()]
)

# %%
irregularities_df[["id_mvi", "timeframe", "jamLevel", "severity"]]

# %%
all_jams_df = (
    irregularities_df[["id_mvi", "timeframe", "jamLevel"]]
    .rename(columns={"id_mvi": "road_id"})
    .merge(
        jams_df[["road_id", "timeframe", "level_mean"]],
        on=["road_id", "timeframe"],
        how="outer",
    )
)

# %%
print(
    "Both jamLevel and level_mean are not NA:",
    round(
        (all_jams_df["jamLevel"].notna() & all_jams_df["level_mean"].notna()).mean()
        * 100,
        2,
    ),
    "%",
)
print(
    "jamLevel is not NA:", round(all_jams_df["jamLevel"].notna().mean() * 100, 2), "%"
)
print(
    "level_mean is not NA:",
    round(all_jams_df["level_mean"].notna().mean() * 100, 2),
    "%",
)

# %% [markdown]
# ## Evaluate node removal

# %%
# df_to_use = irregularities_df
# col_to_use = "jamLevel"
# id_name = "id_mvi"

df_to_use = jams_df
col_to_use = "level_mean"
id_name = "road_id"

df_to_use["timeframe_rounded"] = df_to_use["timeframe"].dt.round("d")
daily_jams_df = df_to_use.groupby([id_name, "timeframe_rounded"])[col_to_use].mean()
jams_per_node = daily_jams_df.groupby(id_name).count()
main_component_nodes = components[0][1].nodes()
jams_per_node = jams_per_node[jams_per_node.index.isin(list(main_component_nodes))]
jams_per_node = jams_per_node.reindex(list(main_component_nodes), fill_value=0)

hours_of_traffic_per_node = df_to_use.groupby(id_name)[col_to_use].count() * 0.25
hours_of_traffic_per_node = hours_of_traffic_per_node.reindex(
    list(main_component_nodes), fill_value=0
)

# %%
thresholds = np.arange(0, hours_of_traffic_per_node.max(), 10)
# thresholds = np.arange(0, 1500, 10)
percent_above_hours = [
    (hours_of_traffic_per_node > x).sum() / len(hours_of_traffic_per_node) * 100
    for x in thresholds
]

days_of_traffic_per_node = jams_per_node
thresholds_days = np.arange(0, days_of_traffic_per_node.max(), 10)
percent_above_days = [
    (days_of_traffic_per_node > x).sum() / len(days_of_traffic_per_node) * 100
    for x in thresholds_days
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(thresholds, percent_above_hours, linewidth=2, color="blue")
axes[0].set_xlabel("Threshold X (hours of traffic)")
axes[0].set_ylabel("Percentage of nodes")
axes[0].set_title("Nodes with > X hours of traffic")
axes[0].grid(True)

axes[1].plot(thresholds_days, percent_above_days, linewidth=2, color="orange")
axes[1].set_xlabel("Threshold X (days with traffic)")
axes[1].set_ylabel("Percentage of nodes")
axes[1].set_title("Nodes with > X days with traffic")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# %%
jams_per_node.describe()

# %%
min_days_jam = 5

nodes_with_many_jams = jams_per_node[jams_per_node >= min_days_jam].index.tolist()
nodes_with_few_jams = jams_per_node[jams_per_node < min_days_jam].index.tolist()
nodes_with_any_jam = jams_per_node[jams_per_node > 0].index.tolist()
nodes_with_no_jams = jams_per_node[jams_per_node == 0].index.tolist()

print("Number of nodes with many jams:", len(nodes_with_many_jams))
print("Number of dropped nodes:", len(nodes_with_few_jams))
print(
    "Percentage of remaining nodes:",
    round(len(nodes_with_many_jams) / len(jams_per_node) * 100, 2),
    "%",
)
print(
    "Percentage of nodes with any jam:",
    round(len(nodes_with_any_jam) / len(jams_per_node) * 100, 2),
    "%",
)
print(
    "Percentage of nodes with no jams:",
    round(len(nodes_with_no_jams) / len(jams_per_node) * 100, 2),
    "%",
)

# m = road_geom.loc[nodes_with_many_jams].explore(style_kwds={"weight": 2.5, "color": "blue"}, height=900, width=1500)
# road_geom.loc[nodes_with_few_jams].explore(m=m, style_kwds={"weight": 2.5, "color": "red"}, height=900, width=1500)
# m

m = road_geom.loc[nodes_with_any_jam].explore(
    style_kwds={"weight": 2.5, "color": "red"}, height=900, width=1500
)
road_geom.loc[nodes_with_no_jams].explore(
    m=m, style_kwds={"weight": 2.5, "color": "green"}, height=900, width=1500
)
m

# %% [markdown]
# ## Plot disconnected nodes

# %%
m = road_geom.loc[list(disconnected_nodes)].explore(
    style_kwds={"weight": 3}, height=900, width=1500
)
m

# %% [markdown]
# ## KNN Imputation

# %%
# Type of street for each street segment (node in the graph)
index = nx.get_node_attributes(base_graph, "MVITIPO").keys()
values = nx.get_node_attributes(base_graph, "MVITIPO").values()
street_types = (
    pd.DataFrame({"node_id": index, "street_type": values})
    .replace({" ": "Null"})
    .set_index("node_id")
)

k_hops = 10
# This dictionary will store the final results
# Keys: Node ID
# Values: A dictionary with the mean trend and regulgarSpeed of its neighbors
neighbor_features: dict[str, dict[str, float]] = {}
neighbor_nodes_with_data: dict[str, list[str]] = {}

# Iterate over each node in the graph
for node in tqdm(base_graph.nodes()):
    if node in disconnected_nodes:
        # If the node is disconnected, we skip it
        neighbor_features[node] = {"mean_trend": np.nan, "mean_regularSpeed": np.nan}
        continue
    k_hop_subgraph = nx.ego_graph(base_graph, node, radius=k_hops)
    neighbors = list(k_hop_subgraph.nodes())
    # Filter neighbors to only those that have the same stree type as the node
    node_street_type = street_types.loc[node, "street_type"]
    neighbors = [
        n
        for n in neighbors
        if street_types.loc[n, "street_type"] == node_street_type
        and n in mean_features.index
    ]
    neighbor_nodes_with_data[node] = neighbors
    # Check if the node has any neighbors
    if neighbors:
        neighbor_df = mean_features.loc[neighbors]

        neighbor_features[node] = {
            "mean_trend": neighbor_df["trend"].mean(),
            "mean_regularSpeed": neighbor_df["regularSpeed"].mean(),
        }
    else:
        # We assign NaN if the node has no neighbors with data
        neighbor_features[node] = {"mean_trend": np.nan, "mean_regularSpeed": np.nan}

neighbor_features_df = pd.DataFrame.from_dict(neighbor_features, orient="index")
print(
    "Proportion without imputable values:",
    neighbor_features_df.isna().sum().iloc[0] / len(neighbor_features_df),
)

# %%
# Type of street for each street segment (node in the graph)
index = nx.get_node_attributes(base_graph, "MVITIPO").keys()
values = nx.get_node_attributes(base_graph, "MVITIPO").values()
street_types = (
    pd.DataFrame({"node_id": index, "street_type": values})
    .replace({" ": "Null"})
    .set_index("node_id")
)

# %%
mean_features

# %%
mean_features_per_street_type = (
    mean_features.merge(street_types, how="left", left_index=True, right_index=True)
    .groupby("street_type")
    .mean()
)
mean_features_per_street_type

# %%
neighbor_features_filled_df = neighbor_features_df.merge(
    street_types, how="left", left_index=True, right_index=True
).merge(
    mean_features_per_street_type, how="left", left_on="street_type", right_index=True
)
neighbor_features_filled_df["mean_trend"] = neighbor_features_filled_df[
    "mean_trend"
].fillna(neighbor_features_filled_df["trend"])
neighbor_features_filled_df["mean_regularSpeed"] = neighbor_features_filled_df[
    "regularSpeed"
].fillna(neighbor_features_filled_df["trend"])
neighbor_features_filled_df.drop(
    columns=["trend", "regularSpeed", "street_type"], inplace=True
)
neighbor_features_filled_df

# %%
# Export pickle files with the neighbor features and nodes with data
with open(os.path.join(processed_data_dir, "neighbor_features.pkl"), "rb") as f:
    neighbor_features_df = pkl.load(f)

# %%
# Export pickle files with the neighbor features and nodes with data
with open(f"neighbor_features.pickle", "wb") as f:
    pkl.dump(neighbor_features_df, f)
# Export the nodes with data
with open(f"neighbor_nodes_with_data.pickle", "wb") as f:
    pkl.dump(neighbor_nodes_with_data, f)

# %%
import random

main_node = random.choice(list(neighbor_nodes_with_data.keys()))
main_node_neighbors = neighbor_nodes_with_data[main_node]
m = road_geom.loc[[main_node]].explore(
    style_kwds={"weight": 7, "color": "green"}, height=900, width=1500
)
road_geom.loc[main_node_neighbors].explore(
    m=m, style_kwds={"weight": 7, "color": "red"}, height=900, width=1500
)
m
