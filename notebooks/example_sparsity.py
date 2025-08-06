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
from torch.utils.data import DataLoader

from src.sparse_dataloader import TrafficDataset

# %load_ext autoreload
# %autoreload 2

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
target_data_filename = f"{sparse_data_dir}/jams_dataset.h5"
in_timesteps = 6
out_timesteps = 3

traffic_dataset = TrafficDataset(
    graph_path=graph_filename,
    static_data_path=static_data_filename,
    dynamic_data_paths=dynamic_data_filenames,
    target_data_path=target_data_filename,
    in_timesteps=in_timesteps,
    out_timesteps=out_timesteps,
)

# %%
batch_size = 2

# IMPORTANT: With this lazy-loading approach, using multiple workers (num_workers > 0)
# can be tricky due to how file handles are passed to subprocesses
# If multiprocessing is needed, we may need to handle file opening within each worker process
traffic_dataloader = DataLoader(
    traffic_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

# %%
print(f"\nDataset size: {len(traffic_dataset)} samples")
print(f"Number of batches: {len(traffic_dataloader)}")
print("-" * 40)
print("-" * 40)
# Get a single batch
sample_batch = next(iter(traffic_dataloader))

# %%
# Inspect the batch
print(f"Edge index shape (batch_size, 2, num_edges):")
print(f"{sample_batch['edge_index'].shape}")
print()
print(f"Static features shape (batch_size, num_nodes, num_static_features):")
print(f"{sample_batch['x_static'].shape}")
print()
print(
    f"Dynamic features shape (batch_size, num_nodes, in_timesteps, num_dynamic_features):"
)
print(f"{sample_batch['x_dynamic'].shape}")
print()
print(f"Target shape (batch_size, num_nodes, out_timesteps):")
print(f"{sample_batch['y'].shape}")

# %%
# Flatten the (node, time) multi-index into a single row index
# row_idx = node_idx * num_timesteps + time_idx

# dynamic_features = SparseTensor(
#     row=Tensor(row_idx).long(),
#     col=Tensor(feature_idx).long(),
#     value=Tensor(values),
#     sparse_sizes=(num_nodes, num_features),
# )

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

df = pd.DataFrame(data, columns=range(1, cols + 1))

df.head(14).style.map(lambda x: "color: red" if x != 0 else "", subset=df.columns)

# %%
# Convert to long format for better visualization
df_long = df.reset_index().melt(id_vars="index", var_name="Feature", value_name="Value")
df_long = df_long[df_long["Value"] != 0]
df_long.sort_values(by=["index", "Feature"], inplace=True)
df_long.head().style.map(lambda x: "color: red" if x != 0 else "", subset=["Value"])

# %%
# Size of df in KB
df_size_kb = df.memory_usage(deep=True).sum() / 1024
print(f"DataFrame size: {df_size_kb:.2f} KB")

# Size of df_long in KB
df_long_size_kb = df_long.memory_usage(deep=True).sum() / 1024
print(f"Long format DataFrame size: {df_long_size_kb:.2f} KB")
