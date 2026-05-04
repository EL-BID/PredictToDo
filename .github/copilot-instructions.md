# Copilot Instructions for congestion-bogota

## Project Overview
Traffic forecasting system for Bogotá using Graph Neural Networks (GNNs) with temporal dynamics. The system predicts road congestion via spatio-temporal modeling of road networks as directed graphs with rich node features (infrastructure, demographics) and dynamic features (jams, irregularities, weather).

## Architecture & Data Flow

### The Two Pipelines

**1. Training Pipeline** (`prepare_dataset.py` → `training_entrypoint.py`)
- **Feature Generation**: 10 processors transform raw data (roads, buildings, events, jams, weather) into HDF5 matrices
- **Data Merging**: Features combined with temporal alignment (15-minute intervals)
- **Dataset Creation**: `TrafficDataset` lazy-loads static features + HDF5 dynamic data
- **Model Training**: `GNN_Temporal_Model` learns node embeddings via graph recurrent cells

**2. Inference Pipeline** (`inference_production.py`)
- Loads trained model + pre-computed normalization constants
- Accepts new dynamic feature CSVs (jams, weather, events)
- Returns per-road predictions with timestamps

### Critical Design Pattern: Node Batching via Seed Reassignment
The `TrafficDataset` uses a **unique non-standard batching strategy**:
- Each sample = neighborhood of nodes via BFS from a seed node
- `__len__()` returns time windows for the **current seed only** (not total samples)
- Training loop must call `dataset.reassign_seed(new_seed)` after each dataloader completion
- `EpochUniquePicker` manages seed traversal to ensure each seed visited once per epoch
- Use `dataset.total_training_samples` property for true epoch workload

**Why?** Memory efficiency for large graphs: loads full graph structure once, dynamically selects neighborhoods.

See [src/sparse_dataloader.py#L1-L35](src/sparse_dataloader.py#L1-L35) for docstring.

## Essential File Structure

| Path | Purpose |
|------|---------|
| `src/prepare_dataset.py` | Orchestrates all 10 feature processors in dependency order |
| `src/sparse_dataloader.py` | `TrafficDataset` class with seed-based batching (1331 lines) |
| `src/training/loop.py` | `train_one_epoch()` with picker integration + detailed timing profiling |
| `src/models/gnn_temporal.py` | 4 GNN cell types (GConvGRU, TGCN, GConvLSTM, A3TGCN2) with static encoder |
| `src/training_utils.py` | `EpochUniquePicker` (seed ordering), `pad_collate()` (batching), node pickers |
| `src/preprocessing/features/*.py` | 10 independent processors for building graph features |
| `src/inference_production.py` | Production serving with error handling + performance monitoring |

## Feature Processing Dependencies & Order
**Must run in this order**:
1. `BuildingsProcessor` → `roads_buildings.csv` (extracts OSM footprints)
2. `BusStopsProcessor` → `bus_stops_per_road.parquet` (SITP data)
3. `RoadGraphConstructor` (depends on 1 & 2) → `road_graph.pickle`
4. `EventsProcessor`, `IncidentsProcessor`, `JamsProcessor`, etc. (all parallel, depend on graph)

## Key Conventions & Patterns

### Naming & IDs
- **Node IDs**: String format `"0_FT.5"` (segment_road_direction), defined in road network GeoJSON
- **Timestamps**: 15-minute intervals stored as `pd.Timestamp` objects, mapped to HDF5 indices
- **CRS**: Primary = `EPSG:3116` (Colombia projected); road network = `EPSG:21899`

### Data Storage
- **Graph**: NetworkX pickled + GraphML + JSON (multi-format for compatibility)
- **Features**: HDF5 with layout `/node_id/feature_name` (lazy-loaded per sample)
- **Normalization**: JSON with per-feature min/max/mean/std (required for inference)

### Model Inputs/Outputs (Batch Processing)
```
x_static: (N, static_dim) - node features (buildings, population, road length)
x_dynamic_past: (B, Tin, N, dynamic_dim) - past jams/weather
x_dynamic_future: (B, H, N, future_dim) - known future exogenous features
y_target: (B, H, N, 1) - speed/congestion labels
edge_index: (2, E) - adjacency; batched internally to (2, E*B)
```

### Normalization Strategy
**Speed/Velocity IS normalized** in the training pipeline:
1. **Computed**: `src/preprocessing/normalize.py` computes percentile-based constants (99th percentile default)
2. **Stored**: Constants saved to `normalization_constants.pkl` with per-feature min/max
3. **Applied**: `TrafficDataset._apply_normalization()` divides target by `mean_jam_velocity` constant, then clamps to [-1, 1]
4. **Two target dimensions**:
   - `y[:, :, 0]` = jam_level (0-5 scale) normalized by `mean_jam_level`
   - `y[:, :, 1]` = velocity (km/h) normalized by `mean_jam_velocity`
5. **Required**: Pass `normalization_constants_path` to `TrafficDataset()` to enable normalization
6. **Inference**: Must apply same normalization constants during inference for consistency

### Testing & Validation
- **Unit tests**: `tests/` use `TestDataGenerator` fixture for synthetic graphs (paths, not real topologies)
- **Integration tests**: SageMaker notebook templates included (`notebooks/sagemaker_training_job.ipynb`)
- **No evaluation yet**: `notebooks/evaluation_example.ipynb` is placeholder

## Developer Workflows

### Setup & Testing
```bash
make format       # Ruff format + import sorting
make lint         # Check formatting + Ruff rules
make test         # pytest on tests/ (coverage available)
make spell-check  # Codespell for typos
```

### Training Loop Profiling
`train_one_epoch()` logs 8 timing categories: data_wait, to_device, reshape, forward, loss, backward+step, seed_pick, batch_total. Use for bottleneck analysis.

### Modern PyTorch GradScaler Usage
Use the new `torch.amp.GradScaler` API instead of deprecated `torch.cuda.amp.GradScaler`:
```python
# Old (deprecated):
from torch.cuda.amp import GradScaler
scaler = GradScaler(enabled=True)

# New (recommended):
from torch.amp import GradScaler
scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu', enabled=True)
```

### Preparing Data Locally
```python
from prepare_dataset import run_feature_generation, run_merging, run_normalization
run_feature_generation()  # Processes all raw data
run_merging()             # Aligns features temporally
run_normalization()       # Computes stats for inference
```

### Production Inference
Expects these CSV inputs: jams_path, irregularities_path, weather_path, pmt_puntos_path, pmt_tramos_path, events_path.
Returns predictions dict with per-road timestamps + values.

## External Dependencies & Integration
- **GNN Libraries**: torch-geometric, torch-geometric-temporal (temporal recurrent cells)
- **Geospatial**: geopandas, osmnx, shapely (road network, OSM data)
- **Distributed**: Ray (feature generation parallelization), Dask (large array ops)
- **Data**: h5py (lazy feature loading), networkx (graph ops)
- **Cloud**: AWS SageMaker integration (training_entrypoint.py parses SM environment variables)

## Common Pitfalls
1. **Forgetting seed reassignment**: DataLoader will exhaust same seed's time windows; manually call `dataset.reassign_seed()`
2. **Feature dependency order**: Building graph without processor outputs causes KeyError; always run builders first
3. **CRS mismatches**: Spatial joins fail silently if CRS not projected; ensure `EPSG:3116` or `EPSG:21899`
4. **HDF5 index mapping**: Node ID strings must match graph node names exactly; typos cause missing data
5. **Batch dimension confusion**: Model expects 4D dynamic inputs (B, T, N, F); 3D inputs silently broadcast incorrectly
6. **Output layer mismatch**: The linear output layer must match your target shape. If `y_target` has shape `(B, N, H, 2)`, ensure `nn.Linear(32, 2)` not `nn.Linear(32, 1)`. Reshape errors like `RuntimeError: shape '[B, N, 2]' is invalid for input of size B*N` indicate this mismatch.

## Where to Start
- **New feature?** Add processor to `src/preprocessing/features/` following `BuildingsProcessor` template
- **Modify GNN?** Edit `src/models/gnn_temporal.py` cell type or encoder
- **Fix inference?** Check normalization constants JSON matches training data
- **Debug data?** Use `notebooks/example_sparsity.ipynb` to visualize feature matrices
