# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %%
# cd ..

# %%
import os
import pandas as pd
import pickle as pkl


# %load_ext autoreload
# %autoreload 2

# %%
processed_data_dir = "data/datos_procesados"
dataset_filenames = {
    "buildings": f"{processed_data_dir}/buildings/roads_buildings.csv",
    "bus_stops": f"{processed_data_dir}/bus_stops/bus_stops_per_road.parquet",
    "events": f"{processed_data_dir}/events/events_dataset.csv",
    "incidents": f"{processed_data_dir}/incidents/incidents_dataset.csv",
    "irregularities": [
        f"{processed_data_dir}/irregularidades_procesadas/" + f
        for f in os.listdir(f"{processed_data_dir}/irregularidades_procesadas/")
        if f.endswith(".parquet")
    ],
    "coos_points": f"{processed_data_dir}/pmts_processed/coos_punto_dataset.parquet",
    "coi_parks": f"{processed_data_dir}/pmts_processed/coi_parques_dataset.parquet",
    "coi_points": [
        f"{processed_data_dir}/pmts_processed/coi_puntos/" + f
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_puntos/")
        if f.endswith(".parquet")
    ],
    "coi_tracts": [
        f"{processed_data_dir}/pmts_processed/coi_tramos/" + f
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_tramos/")
        if f.endswith(".parquet")
    ],
    "road_state": f"{processed_data_dir}/road_state/road_state_dateset.parquet",  # TODO: Should be 'dataset' no 'dateset'
    "jams": [
        f"{processed_data_dir}/jams_procesados/" + f
        for f in os.listdir(f"{processed_data_dir}/jams_procesados/")
        if f.endswith(".parquet")
    ],
}

graph_filename = f"{processed_data_dir}/conectivity_graph/road_graph.pickle"
imputation_parquet_filename = f"{processed_data_dir}/neighbor_features.parquet"

# %%
with open(graph_filename, "rb") as f:
    base_graph = pkl.load(f)
all_nodes = list(base_graph.nodes())

road_state_df = (
    pd.read_parquet(dataset_filenames["road_state"])
    .set_index("road_id")
    .drop(columns=["timeframe_start", "timeframe_end"])
)

events_df = (
    pd.read_csv(dataset_filenames["events"])
    .set_index(["road_id", "timeframe"])
    .astype("Int8")
)
events_df.index = events_df.index.set_levels(
    [events_df.index.levels[0], pd.to_datetime(events_df.index.levels[1])], level=[0, 1]
)

incidents_df = (
    pd.read_csv(dataset_filenames["incidents"])
    .set_index(["road_id", "timeframe"])
    .astype("Int8")
)
incidents_df.index = incidents_df.index.set_levels(
    [incidents_df.index.levels[0], pd.to_datetime(incidents_df.index.levels[1])],
    level=[0, 1],
)

irregularities_df = pd.concat(
    [
        pd.read_parquet(f)
        .rename(columns={"id_mvi": "road_id"})
        .set_index(["road_id", "timeframe"])
        for f in dataset_filenames["irregularities"]
    ],
    ignore_index=False,
).fillna(0)
irregularities_df.index = irregularities_df.index.set_levels(
    [
        irregularities_df.index.levels[0],
        pd.to_datetime(irregularities_df.index.levels[1]),
    ],
    level=[0, 1],
)

# TODO: The irregularities should be filled with the default_speeds
default_speeds_df = pd.read_parquet(imputation_parquet_filename)
default_speeds_df.index.name = "road_id"

coi_points = (
    pd.concat(
        [
            pd.read_parquet(f).set_index(["road_id", "timeframe"])
            for f in dataset_filenames["coi_points"]
        ],
        ignore_index=False,
    )
    .fillna(0)
    .astype("Int8")
)
coi_points.index = coi_points.index.set_levels(
    [coi_points.index.levels[0], pd.to_datetime(coi_points.index.levels[1])],
    level=[0, 1],
)

coi_tracts = (
    pd.concat(
        [
            pd.read_parquet(f).set_index(["road_id", "timeframe"])
            for f in dataset_filenames["coi_tracts"]
        ],
        ignore_index=False,
    )
    .fillna(0)
    .astype("Int8")
)
coi_tracts.index = coi_tracts.index.set_levels(
    [coi_tracts.index.levels[0], pd.to_datetime(coi_tracts.index.levels[1])],
    level=[0, 1],
)

all_timeframes = pd.date_range(
    start=pd.to_datetime("2024-01-01 00:00:00"),
    end=pd.to_datetime("2024-12-31 23:45:00"),
    freq="15min",
).to_list()

all_columns = (
    events_df.columns.tolist()
    + incidents_df.columns.tolist()
    + road_state_df.columns.tolist()
    + irregularities_df.columns.tolist()
    + coi_points.columns.tolist()
    + coi_tracts.columns.tolist()
)
all_columns.remove("timeframe_end")
all_columns.remove("timeframe_start")
all_columns.remove("road_id")

# %%
dfs = [
    events_df,
    incidents_df,
    irregularities_df,
    coi_points,
    coi_tracts,
]
min_timeframes, max_timeframes = [], []
for df in dfs:
    min_timeframes.append(df.index.get_level_values("timeframe").min())
    max_timeframes.append(df.index.get_level_values("timeframe").max())

print(f"Min timeframes: {min_timeframes}")
print(f"Max timeframes: {max_timeframes}")

all_timeframes = pd.date_range(
    start=pd.to_datetime("2024-01-01 00:00:00"),
    end=pd.to_datetime("2024-12-31 23:45:00"),
    freq="15min",
).to_list()

all_columns = (
    events_df.columns.tolist()
    + incidents_df.columns.tolist()
    + road_state_df.columns.tolist()
    + irregularities_df.columns.tolist()
    + coi_points.columns.tolist()
    + coi_tracts.columns.tolist()
)
all_columns.remove("timeframe_end")
all_columns.remove("timeframe_start")
all_columns.remove("road_id")

# %%
non_zero_temporal_values = (
    (events_df != 0).sum().sum()
    + (incidents_df != 0).sum().sum()
    + (irregularities_df != 0).sum().sum()
    + (coi_points != 0).sum().sum()
    + (coi_tracts != 0).sum().sum()
)
total_values = len(all_timeframes) * len(all_nodes) * len(all_columns)
print(f"Total values: {total_values}")
print(f"Non-zero temporal values: {non_zero_temporal_values}")
print(
    f"Percentage of non-zero values: {non_zero_temporal_values / total_values * 100:.2f}%"
)

size_full = (
    total_values * 8
)  # Size in bytes of the dataset (assuming float32 for all features)
size_full_gb = size_full / (1024**3)  # Convert to GB
print(f"Size of the full dataset: {size_full_gb:.2f} GB")

size_sparse = (
    non_zero_temporal_values * 12
)  # Size in bytes of the dataset (assuming float32 for all features)
size_sparse_gb = size_sparse / (1024**3)  # Convert to GB
print(f"Size of the sparse dataset: {size_sparse_gb:.2f} GB")
