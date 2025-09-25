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
from torch_geometric.loader import DataLoader

from src.sparse_dataloader import TrafficDataset

# %load_ext autoreload
# %autoreload 2

# %load_ext line_profiler

# %%
processed_data_dir = "data/datos_procesados"
sparse_data_dir = f"{processed_data_dir}/sparse_datasets"
graph_filename = f"{processed_data_dir}/conectivity_graph/road_graph.pickle"
static_data_filename = f"{sparse_data_dir}/static_dataset.h5"
dynamic_data_filenames = [
    f"{sparse_data_dir}/coi_points_dataset.h5",
    f"{sparse_data_dir}/coi_tracts_dataset.h5",
    f"{sparse_data_dir}/irregularities_dataset.h5",
    f"{sparse_data_dir}/events_dataset.h5",
    f"{sparse_data_dir}/incidents_dataset.h5",
]
metadata_filenames = [
    f"{sparse_data_dir}/coi_points_dataset_metadata.pkl",
    f"{sparse_data_dir}/coi_tracts_dataset_metadata.pkl",
    f"{sparse_data_dir}/irregularities_dataset_metadata.pkl",
    f"{sparse_data_dir}/events_dataset_metadata.pkl",
    f"{sparse_data_dir}/incidents_dataset_metadata.pkl",
]
target_data_filename = f"{sparse_data_dir}/jams_dataset.h5"
in_timesteps = 6
out_timesteps = 3

traffic_dataset = TrafficDataset(
    graph_path=graph_filename,
    static_data_path=static_data_filename,
    dynamic_data_paths=dynamic_data_filenames,
    metadata_paths=metadata_filenames,
    target_data_path=target_data_filename,
    in_timesteps=in_timesteps,
    out_timesteps=out_timesteps,
    future_features=[
        "coi_puntos",
        "coi_tramo",
        "AGLOMERACION",
        "EMERGENCIA",
        "MANIFESTACION",
    ],
    node_batch_size=1024,
)

# %%
batch_size = 3

traffic_dataloader = DataLoader(
    dataset=traffic_dataset,
    batch_size=batch_size,  # This now batches the Data objects
    shuffle=True,
    num_workers=0,
)

# %%
print(f"\nDataset size: {len(traffic_dataset)} samples")
print(f"Number of batches: {len(traffic_dataloader)}")
print("-" * 40)
print("-" * 40)
# Get a single batch
sample_batch = next(iter(traffic_dataloader))

# Inspect the batch
print(f"Edge index shape (batch_size, 2, num_edges):")
print(f"{sample_batch['edge_index'].shape}")
print()
print(f"Static features shape (batch_size, num_nodes, num_static_features):")
print(f"{sample_batch['x_static'].shape}")
print()
print(
    f"Past dynamic features shape (batch_size, num_nodes, in_timesteps, num_dynamic_features):"
)
print(f"{sample_batch['x_dynamic_past'].shape}")
print()
print(
    f"Future dynamic features shape (batch_size, num_nodes, out_timesteps, num_dynamic_features):"
)
print(f"{sample_batch['x_dynamic_future'].shape}")
print()
print(f"Target shape (batch_size, num_nodes, out_timesteps):")
print(f"{sample_batch['y'].shape}")

# %%
# %lprun -f TrafficDataset.__getitem__ next(iter(traffic_dataloader))

# %% [markdown]
# ## Compare sorted search vs. naive approach

# %%
import h5py
import numpy as np

file_path = traffic_dataset.dynamic_data_paths[2]
idx = 100


file_indices = np.zeros((traffic_dataset.total_timesteps_in_data - 1, 2), dtype=int)
with h5py.File(file_path, "r") as f:
    # Reading the entire time_idx column can be slow
    time_column = f["data"][:, 1]
    file_indices[:, 0] = np.searchsorted(
        time_column, np.arange(traffic_dataset.total_timesteps_in_data - 1), side="left"
    )
    file_indices[:, 1] = np.searchsorted(
        time_column,
        np.arange(traffic_dataset.total_timesteps_in_data - 1),
        side="right",
    )


def demo_no_sorting(idx):
    input_start_time = idx
    input_end_time = input_start_time + 50
    with h5py.File(file_path, "r") as f:
        # Reading the entire time_idx column can be slow
        all_time_idx = f["data"][:, 1]
        # Find the rows that fall within our time window
        # TODO: If the data is sorted by time, np.searchsorted can make this much faster
        rows_in_window_mask = (all_time_idx >= input_start_time) & (
            all_time_idx < input_end_time
        )
        relevant_data = f["data"][rows_in_window_mask]
        del all_time_idx, rows_in_window_mask  # Free memory
    return relevant_data


def demo_sorted(idx):
    input_start_time = idx
    input_end_time = input_start_time + 50
    with h5py.File(file_path, "r") as f:
        start_index = file_indices[input_start_time, 0]
        end_index = file_indices[input_end_time, 1]
        relevant_data = f["data"][start_index:end_index]
    return relevant_data


relevant_data_no_sorting = demo_no_sorting(idx)
relevant_data_sorted = demo_sorted(idx)

assert (relevant_data_sorted == relevant_data_no_sorting).mean() == 1.0, (
    "Data mismatch between sorted and unsorted methods"
)

# %%
# %lprun -f demo_no_sorting demo_no_sorting(idx)

# %%
# %lprun -f demo_sorted demo_sorted(idx)

# %%
print("Improvement of", round(100 - (0.000815 / 2.35875) * 100, 2), "%")

# %% [markdown]
# ## How does sparsity work?

# %%
import pandas as pd
import numpy as np

rows, cols = 100, 50
data = np.zeros((rows, cols), dtype=int)

# Randomly assign some nonzero values
np.random.seed(42)
num_nonzero = 50
row_idx = np.random.randint(0, rows, num_nonzero)
col_idx = np.random.randint(0, cols, num_nonzero)
values = np.random.randint(1, 100, num_nonzero)
data[row_idx, col_idx] = values

df = pd.DataFrame(data, columns=range(1, cols+1))

df.head(14).style.map(lambda x: 'color: red' if x != 0 else '', subset=df.columns)

# %%
# Convert to long format for better visualization
df_long = df.reset_index().melt(id_vars='index', var_name='Feature', value_name='Value')
df_long = df_long[df_long['Value'] != 0]
df_long.sort_values(by=['index', 'Feature'], inplace=True)
df_long.head().style.map(lambda x: 'color: red' if x != 0 else '', subset=['Value'])

# %%
# Size of df in KB
df_size_kb = df.memory_usage(deep=True).sum() / 1024
print(f"DataFrame size: {df_size_kb:.2f} KB")

# Size of df_long in KB
df_long_size_kb = df_long.memory_usage(deep=True).sum() / 1024
print(f"Long format DataFrame size: {df_long_size_kb:.2f} KB")
