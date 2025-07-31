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
import pickle as pkl

import pandas as pd
from tqdm.auto import tqdm

from src.dataloader import GraphTimeSeriesLoader
from src.preprocessing.merging import SparseDatasetGenerator

# %load_ext autoreload
# %autoreload 2

# %%
processed_data_dir = "data/datos_procesados"

with open(f"{processed_data_dir}/conectivity_graph/road_graph.pickle", "rb") as f:
    base_graph = pkl.load(f)

# Clean the base graph node attributes
base_graph = GraphTimeSeriesLoader._clean_node_attributes(base_graph)
all_road_ids = list(base_graph.nodes())

dataset_generator = SparseDatasetGenerator(
    node_ids=all_road_ids,
    min_timeframe=pd.to_datetime("2024-01-01 00:00:00"),
    max_timeframe=pd.to_datetime("2024-12-31 23:45:00"),
    freq="15min",
)

# %%
static_node_data = pd.DataFrame.from_dict(
    dict(base_graph.nodes(data=True)), orient="index"
)
all_road_ids = static_node_data.index.tolist()

road_state_df = pd.read_parquet(
    f"{processed_data_dir}/road_state/road_state_dateset.parquet"
)
road_state_df = road_state_df.drop(
    columns=["timeframe_start", "timeframe_end"]
).set_index("road_id")

static_df = static_node_data.join(road_state_df, how="left")
static_df.index.name = "road_id"
del base_graph, road_state_df, static_node_data  # Clean up memory

# %%
dataset_generator.static_dataframe_to_coo(
    static_df,
    output_path=f"{processed_data_dir}/sparse_datasets/static_dataset.h5",
    has_index=True,
    return_df=False,
)

# %%
pd.read_csv(f"{processed_data_dir}/incidents/incidents_dataset.csv")

# %%
processed_data_dir = "data/datos_procesados"
dataset_filenames = {
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
jams_df = pd.concat(
    [pd.read_parquet(f) for f in dataset_filenames["jams"]],
    ignore_index=False,
).fillna(0)

# %%
with open(graph_filename, "rb") as f:
    base_graph = pkl.load(f)
all_nodes = list(base_graph.nodes())

# road_state_df = (
#     pd.read_parquet(dataset_filenames["road_state"])
#     .set_index("road_id")
#     .drop(columns=["timeframe_start", "timeframe_end"])
# )

# events_df = (
#     pd.read_csv(dataset_filenames["events"])
#     .set_index(["road_id", "timeframe"])
#     .astype("Int8")
# )
# events_df.index = events_df.index.set_levels(
#     [events_df.index.levels[0], pd.to_datetime(events_df.index.levels[1])], level=[0, 1]
# )

# incidents_df = (
#     pd.read_csv(dataset_filenames["incidents"])
#     .set_index(["road_id", "timeframe"])
#     .astype("Int8")
# )
# incidents_df.index = incidents_df.index.set_levels(
#     [incidents_df.index.levels[0], pd.to_datetime(incidents_df.index.levels[1])],
#     level=[0, 1],
# )

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

# # TODO: The irregularities should be filled with the default_speeds
# default_speeds_df = pd.read_parquet(imputation_parquet_filename)
# default_speeds_df.index.name = "road_id"

# coi_points = (
#     pd.concat(
#         [
#             pd.read_parquet(f).set_index(["road_id", "timeframe"])
#             for f in dataset_filenames["coi_points"]
#         ],
#         ignore_index=False,
#     )
#     .fillna(0)
#     .astype("Int8")
# )
# coi_points.index = coi_points.index.set_levels(
#     [coi_points.index.levels[0], pd.to_datetime(coi_points.index.levels[1])],
#     level=[0, 1],
# )

# coi_tracts = (
#     pd.concat(
#         [
#             pd.read_parquet(f).set_index(["road_id", "timeframe"])
#             for f in dataset_filenames["coi_tracts"]
#         ],
#         ignore_index=False,
#     )
#     .fillna(0)
#     .astype("Int8")
# )
# coi_tracts.index = coi_tracts.index.set_levels(
#     [coi_tracts.index.levels[0], pd.to_datetime(coi_tracts.index.levels[1])],
#     level=[0, 1],
# )

# %%
all_timeframes = pd.date_range(
    start=pd.to_datetime("2024-01-01 00:00:00"),
    end=pd.to_datetime("2024-12-31 23:45:00"),
    freq="15min",
).to_list()

# all_columns = (
#     events_df.columns.tolist()
#     + incidents_df.columns.tolist()
#     + road_state_df.columns.tolist()
#     + irregularities_df.columns.tolist()
#     + coi_points.columns.tolist()
#     + coi_tracts.columns.tolist()
# )
# all_columns.remove("timeframe_end")
# all_columns.remove("timeframe_start")
# all_columns.remove("road_id")


# %%
def dataframe_to_coo(all_timeframes, all_nodes, df):
    time_idx_map = {ts: i for i, ts in enumerate(all_timeframes)}
    node_idx_map = {node: i for i, node in enumerate(all_nodes)}

    # Feature mapping: feature name -> integer index
    # all_features = sorted(list(set(col for df in dataframes for col in df.columns)))
    all_features = sorted(list(set(df.columns)))
    feature_idx_map = {feature: i for i, feature in enumerate(all_features)}

    # Reset index to turn (road_id, timeframe) into columns
    df = df.reset_index()

    feature_long_dfs = []
    # Iterate over each feature column to process it individually, avoiding `melt`
    for feature_name in tqdm(df.columns):
        # Get the series for the current feature
        series = df[feature_name]
        # Filter out zero values immediately to keep the series small
        non_zero_series = series[series != 0]

        if not non_zero_series.empty:
            # Convert the sparse series to a DataFrame
            feature_df = non_zero_series.reset_index()
            # Rename the column containing values to a generic 'value'
            feature_df.rename(columns={feature_name: "value"}, inplace=True)
            # Add a column to identify which feature these values belong to
            feature_df["feature_name"] = feature_name
            feature_long_dfs.append(feature_df)

    # If any non-zero values were found in this DataFrame, combine them
    if feature_long_dfs:
        long_df = pd.concat(feature_long_dfs, ignore_index=True)
        # Apply the integer mappings
        long_df["node_idx"] = long_df["road_id"].map(node_idx_map)
        long_df["time_idx"] = long_df["timeframe"].map(time_idx_map)
        long_df["feature_idx"] = long_df["feature_name"].map(feature_idx_map)

        # Select and reorder columns, and drop temporary ones
        final_cols = ["node_idx", "time_idx", "feature_idx", "value"]
        long_df = long_df[final_cols]
        long_df["node_idx"] = long_df["node_idx"].astype(int)
        long_df["time_idx"] = long_df["time_idx"].astype(int)
        long_df["feature_idx"] = long_df["feature_idx"].astype(int)

    print(f"    - Found and processed {len(long_df)} non-zero entries.")
    return long_df


# --- 4. Final Combination and Saving ---
# print("\nStep 4: Combining data and saving to disk...")

# # Concatenate all processed DataFrames into one
# final_df = pd.concat(processed_dfs, ignore_index=True)

# # Drop any potential duplicates that might arise from overlapping data
# final_df.drop_duplicates(subset=['node_id', 'time_idx', 'feature_idx'], keep='first', inplace=True)

# # Sort for efficiency. This can make slicing by time_idx faster later.
# final_df.sort_values(by=['time_idx', 'node_id'], inplace=True)

# # Save to a CSV file. Using Parquet ('temporal_features.parquet') is
# # often faster and more space-efficient for large datasets.
# output_filename = 'temporal_features_sparse.csv'
# final_df.to_csv(output_filename, index=False)


# %%
irregularities_coo = dataframe_to_coo(all_timeframes, all_nodes, irregularities_df)

# %%
irregularities_coo

# %%
serie = irregularities_df.xs("0_FT.1", level="road_id").ACCIDENT_MAJOR_count
serie[serie != 0]

# %%
processed_df = processed_df.set_index(["road_id", "time_idx", "feature_idx"])

# %%
size_mb = irregularities_df.memory_usage(deep=True).sum() / (1024**2)
print(f"Size of irregularities_df: {size_mb:.2f} MB")

size_mb = processed_df.memory_usage(deep=True).sum() / (1024**2)
print(f"Size of processed_df: {size_mb:.2f} MB")

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

# %%
