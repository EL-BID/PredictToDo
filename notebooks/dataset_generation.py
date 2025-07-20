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
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from src.dataloader import GraphTimeSeriesLoader, GraphTimeSeriesDataset
import re
from collections import defaultdict
# %load_ext autoreload
# %autoreload 2

# %%
# === Utility for snake_case conversion ===
import pickle


def to_snake_case(string: str) -> str:
    string = string.replace("mvi", "mvi_").replace("MVI", "MVI_")
    string = re.sub(r"[^a-zA-Z0-9]+", "_", string)
    string = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string)
    string = string.lower()
    string = re.sub(r"_+", "_", string)
    return string.strip("_")


# === Base path ===
processed_data_dir = "data/datos_procesados"

# === Define dynamic datasets only ===
dynamic_datasets = {
    "events": [f"{processed_data_dir}/events/events_dataset.csv"],
    "incidents": [f"{processed_data_dir}/incidents/incidents_dataset.csv"],
    "road_state": [f"{processed_data_dir}/road_state/road_state_dateset.parquet"],
    "irregularities": [
        os.path.join(processed_data_dir, "irregularidades_procesadas", f)
        for f in os.listdir(f"{processed_data_dir}/irregularidades_procesadas/")
        if f.endswith(".parquet")
    ]
    if os.path.exists(f"{processed_data_dir}/irregularidades_procesadas/")
    else [],
    "coi_points": [
        os.path.join(processed_data_dir, "pmts_processed/coi_puntos", f)
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_puntos/")
        if f.endswith(".parquet")
    ]
    if os.path.exists(f"{processed_data_dir}/pmts_processed/coi_puntos/")
    else [],
    "coi_tracts": [
        os.path.join(processed_data_dir, "pmts_processed/coi_tramos", f)
        for f in os.listdir(f"{processed_data_dir}/pmts_processed/coi_tramos/")
        if f.endswith(".parquet")
    ]
    if os.path.exists(f"{processed_data_dir}/pmts_processed/coi_tramos/")
    else [],
}

# === Track source columns ===
column_sources = defaultdict(set)

# === Skip columns you want to exclude globally ===
EXCLUDE_COLUMNS = {"road_id", "timeframe", "timeframe_start", "timeframe_end"}

# === Scan columns per dataset ===
for dataset_name in tqdm(dynamic_datasets, desc="Scanning datasets"):
    for path in tqdm(
        dynamic_datasets[dataset_name], desc=f"{dataset_name}", leave=False
    ):
        if not os.path.exists(path):
            print(f"[Missing] {path}")
            continue
        try:
            df = (
                pd.read_parquet(path)
                if path.endswith(".parquet")
                else pd.read_csv(path)
            )

            # Standardize
            if "id_mvi" in df.columns:
                df = df.rename(columns={"id_mvi": "road_id"})

            for col in df.columns:
                if col in EXCLUDE_COLUMNS:
                    continue
                clean_col = to_snake_case(col)
                column_sources[clean_col].add(dataset_name)

        except Exception as e:
            print(f"[Error] {dataset_name} – {path}: {e}")

# === Final cleaned dictionary object ===
DYNAMIC_FEATURE_COLUMNS = {
    col: sorted(list(sources)) for col, sources in column_sources.items()
}

processed_data_dir = "data/datos_procesados"
with open(f"{processed_data_dir}/columns_dict.pkl", "wb") as f_pkl:
    pickle.dump(DYNAMIC_FEATURE_COLUMNS, f_pkl)

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
dynamic_columns = f"{processed_data_dir}/columns_dict.pkl"


# Load one file from each dataset to see the column names and their types
def load_dataset_columns(dataset_filenames):
    columns: dict[str, str] = {}
    for key, filename in dataset_filenames.items():
        if isinstance(filename, list):
            # If it's a list, load the first file
            filename = filename[0]
        df = (
            pd.read_parquet(filename)
            if filename.endswith(".parquet")
            else pd.read_csv(filename)
        )
        columns.update(
            {f"{key}_{col}": str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
        )
    return columns


dataset_columns = load_dataset_columns(dataset_filenames)
dataset_columns_df = (
    pd.DataFrame(dataset_columns, index=["dtype"])
    .T.reset_index()
    .rename(columns={"index": "column_name"})
)
object_columns = dataset_columns_df[
    ~dataset_columns_df["dtype"].isin(["float64", "int64", "int32"])
].column_name

assert all(
    "timeframe" in col or "road_id" in col or "id_mvi" in col for col in object_columns
)

# %%
loader = GraphTimeSeriesLoader(
    graph_pkl_filename=graph_filename,
    events_csv_filename=dataset_filenames["events"],
    incidents_csv_filename=dataset_filenames["incidents"],
    road_state_parquet_filename=dataset_filenames["road_state"],
    irregularities_filenames=dataset_filenames["irregularities"],
    coi_points_filenames=dataset_filenames["coi_points"],
    coi_tracts_filenames=dataset_filenames["coi_tracts"],
    default_speeds_parquet_filename=imputation_parquet_filename,
    jams_parquet_filenames=dataset_filenames["jams"],
)

# %%
# incident_set = set([tuple(l) for l in loader.incidents_df[["road_id", "timeframe"]].values])
# event_set = set([tuple(l) for l in loader.events_df[["road_id", "timeframe"]].values])
# common_tuples = incident_set.intersection(event_set)
# list(common_tuples)[:5]

# %%
# Example usage of the loader to load data for specific nodes and time windows
nodes_to_load = ["9004322_TF.1", "50008506_TF.1", "12003262_FT.1"]
time_windows_to_load = [
    ("2024-09-29 23:00:00", 96, 288),
    ("2024-08-24 15:30:00", 96, 288),
    ("2024-02-24 01:30:00", 96, 288),
]

# Load the data
(static_features, temporal_features), target = loader.load_data(
    nodes_to_load, time_windows_to_load
)

# %%
temporal_features

# %%
# Create an instance of the custom dataset
train_dataset = GraphTimeSeriesDataset(
    loader=loader, node_ids=nodes_to_load, time_windows=time_windows_to_load
)

# Create the DataLoader
# This will handle batching, shuffling, and parallel data loading.
batch_size = 2
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle the data at the beginning of each epoch
    num_workers=0,  # Set to > 0 to use multiprocessing for data loading
)

# Iterate over the DataLoader in the training loop
num_epochs = 3
for epoch in range(num_epochs):
    print(f"--- Epoch {epoch + 1}/{num_epochs} ---")
    for i, (static_batch, past_batch, future_batch, target_batch) in enumerate(
        train_dataloader
    ):
        static_batch = static_batch[0].squeeze(0)
        print(f"  Batch {i + 1}:")
        print(f"    Static data shape: {static_batch.shape}")
        print(f"    Past data shape:   {past_batch.shape}")
        print(f"    Future data shape: {future_batch.shape}")
        print(f"    Target data shape: {target_batch.shape}")
        print("-" * 20)

        # Here the batches should be fed to the GNN model for training
        # model.train()
        # optimizer.zero_grad()
        # output = model(static_batch, temporal_batch)
        # loss = criterion(output, target_batch)
        # loss.backward()
        # optimizer.step()

# %%
32 * 135000

# %%
static_batch.shape
