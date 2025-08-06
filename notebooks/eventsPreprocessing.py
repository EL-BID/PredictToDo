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
#     display_name: jose310
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Polygon, Point
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
import os
import datetime as dt
import networkx as nx
import pickle
from itertools import product

# %%
tqdm.pandas()


# %%
def get_roads_within_distance(R, seed_road, max_distance_m):
    """
    Returns all roads reachable from seed_road within max_distance_m meters.

    R                : networkx.Graph where nodes have attribute 'length_m'
    seed_road        : node ID (road_id) from which to start
    max_distance_m   : maximum cumulative distance in meters

    Returns:
      distances : dict {road_id: cumulative_distance}
      df_edges  : pandas.DataFrame with edges between reachable roads and their attributes
    """
    # 1. Ensure each edge has a 'weight' attribute: avg of node lengths
    for u, v, data in R.edges(data=True):
        if "weight" not in data:
            data["weight"] = (R.nodes[u]["length_m"] + R.nodes[v]["length_m"]) / 2.0

    # 2. Run Dijkstra up to cutoff
    distances = nx.single_source_dijkstra_path_length(
        R, seed_road, cutoff=max_distance_m, weight="weight"
    )
    reachable = set(distances.keys())

    return list(reachable)


# %%
def quarter_slices(start, end):
    """
    Given two pandas Timestamps (start < end), return a list of (slice_start, slice_end)
    pairs, each of which lies wholly within one 15-minute bucket (minute marks 00,15,30,45).
    """
    out = []
    current = start

    while current < end:
        # Find “floor” of current to the nearest 15 minutes, then add 15 min
        floor_q = current.floor("15T")
        next_bound = floor_q + pd.Timedelta(minutes=15)

        # If current is exactly on a quarter, jump to the next one
        if current == floor_q:
            next_bound = current + pd.Timedelta(minutes=15)

        # End of this chunk is either the next boundary or the original end
        chunk_end = min(next_bound, end)

        out.append(current)
        current = chunk_end
    return out


def get_date(x):
    try:
        return dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    except:
        return dt.datetime.strptime(x, "%d/%m//%Y")


# %%
roads_gdf = gpd.read_file(
    "../data/2_Estructura_via/estructura_via_completa/MVI.geojson"
)
roads_gdf = roads_gdf.to_crs(epsg=4326)
roads_gdf["id_mvi"] = (
    roads_gdf["MVICIV"].astype(str) + "_" + roads_gdf["MVISVIA"].str.replace(" ", "")
)
roads_gdf["contador"] = roads_gdf.groupby("id_mvi").cumcount() + 1
roads_gdf["id_mvi"] = roads_gdf["id_mvi"] + "." + roads_gdf["contador"].astype(str)
roads_gdf = (
    roads_gdf.explode(ignore_index=True)  # if shapefile has split segments
    .dissolve(by="id_mvi")  # merge all segments of same road
    .reset_index()
)
roads_gdf["centroid"] = list(map(lambda x: x.centroid, roads_gdf.geometry))
roads_gdf["road_id"] = roads_gdf["id_mvi"]
roads_gdf["buffer"] = roads_gdf.geometry.buffer(0.0027)
roads_buf = roads_gdf.set_geometry("buffer")[["road_id", "buffer", "centroid"]]

# %%
events = pd.read_csv(
    "../data/8_eventos/events_dataset_preprocessing.csv", encoding="latin-1"
)
events = events.loc[pd.notnull(events.Longitude)]
events["geometry"] = list(
    map(lambda x, y: Point(x, y), events["Longitude"], events["Latutude"])
)
events = gpd.GeoDataFrame(events).set_crs(epsg=4326)
events["id_event"] = events.index
events["fecha_inicio"] = list(
    tqdm(map(lambda start: get_date(start), events["fecha_inicio"]))
)
events["fecha_final"] = list(
    tqdm(map(lambda end: get_date(end), events["fecha_final"]))
)

# %%
joined = gpd.sjoin(events, roads_buf, how="left", predicate="intersects")

# %%
joined = joined.drop(["index_right"], axis=1)
joined["geometry"] = joined["geometry"].apply(
    lambda x: Point(x) if not isinstance(x, Point) else x
)
joined["centroid"] = joined["centroid"].apply(
    lambda x: Point(x) if not isinstance(x, Point) else x
)

# Compute distance
joined["distance"] = joined.apply(
    lambda row: row["geometry"].distance(row["centroid"]), axis=1
)

# Rank by distance per id_event
joined["rank_distance"] = joined.groupby("id_event")["distance"].rank(method="dense")
joined = joined.loc[joined.rank_distance == 1].reset_index(drop=True)

# %%
file_input_name = "../datos_procesados/conectivity_graph/road_graph.pickle"

with open(file_input_name, "rb") as f:
    R = pickle.load(f)

# %%
hop_distances = {"ALTA": 500, "MEDIA": 300, "BAJA": 200, "SIN DEFINIR": 500}
joined["hop_distance"] = list(
    tqdm(map(lambda x: hop_distances.get(x, 500), joined.Complejidad))
)
joined["reachable_nodes"] = list(
    tqdm(
        map(
            lambda road_id, hop_distance: get_roads_within_distance(
                R, road_id, hop_distance
            )
            if pd.notnull(road_id)
            else [],
            joined["road_id"],
            joined["hop_distance"],
        )
    )
)
joined = joined.drop(["road_id"], axis=1)
joined["timeframe_set"] = list(
    tqdm(
        map(
            lambda start, end: quarter_slices(start, end),
            joined["fecha_inicio"],
            joined["fecha_final"],
        )
    )
)
joined["reachable_pairs"] = joined.progress_apply(
    lambda row: list(product(row["reachable_nodes"], row["timeframe_set"])), axis=1
)

# %%
exploded_df = joined.explode("reachable_pairs").reset_index(drop=True)
exploded_df[["road_id", "timeframe"]] = pd.DataFrame(
    exploded_df["reachable_pairs"].tolist(), index=exploded_df.index
)
exploded_df = exploded_df.drop(
    columns=[
        "reachable_pairs",
        "reachable_nodes",
        "timeframe_set",
        "centroid",
        "geometry",
    ]
)
exploded_df["total"] = 1
events_dataset = (
    exploded_df.groupby(["road_id", "timeframe", "tipo_actividad"])
    .aggregate({"total": "sum"})
    .reset_index()
    .pivot_table(
        index=["road_id", "timeframe"], columns=["tipo_actividad"], values="total"
    )
    .reset_index()
    .fillna(0)
)

# %%
file_output_name = "../data/datos_procesados/events/"
os.makedirs(file_output_name, exist_ok=True)
events_dataset.to_csv(f"{file_output_name}events_dataset.csv", index=False)
