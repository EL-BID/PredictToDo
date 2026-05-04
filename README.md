# Dataset Preparation Guide

This guide explains how to prepare the complete dataset for the traffic forecasting model using the automated pipeline.

## Table of Contents
- [Training Pipeline](#training-pipeline)
  - [Quick Start](#quick-start)
  - [Pipeline Overview](#pipeline-overview)
- [Inference Pipeline](#inference-pipeline)
  - [Quick Start](#inference-quick-start)
  - [Usage](#inference-usage)

## Training Pipeline

### Quick Start

To prepare the entire dataset, simply run:

```bash
uv run python prepare_dataset.py
```

This will execute all preprocessing steps in the correct order and produce a ready-to-use `TrafficDataset` object.

## Pipeline Overview

The script executes five main steps:

### Step 1: Feature Generation

Creates processed datasets from raw data sources:

0. **Buildings** (`buildings.py`)
   - Extracts building footprints from OpenStreetMap
   - Creates 100m buffers around road segments
   - Performs spatial join between buildings and roads
   - Aggregates building counts by type (residential, commercial, industrial, etc.)
   - Output: `roads_buildings.csv`
   - **Prerequisites**: None (must run first)
   - **Required by**: Road Network Graph

1. **Bus Stops** (`bus_stops.py`)
   - Loads SITP bus stop shapefile data
   - Assigns each bus stop to the nearest road segment
   - Calculates accurate distances in meters using projected CRS
   - Aggregates stop counts and statistics per road
   - Distinguishes between main stops (M) and secondary stops (S)
   - Output: `bus_stops_per_road.parquet`
   - **Prerequisites**: None (must run before graph)
   - **Required by**: Road Network Graph

2. **Road Network Graph** (`graph.py`)
   - Builds a directed graph of the road network
   - Adds node attributes (length, land use, population, buildings, bus stops)
   - Creates legal and illegal edges based on traffic flow rules
   - Output: `road_graph.pickle`, `road_graph.graphml`, `road_graph.json`
   - **Prerequisites**: Buildings, Bus Stops
   - **Required by**: Events, other processors

3. **Events** (`events.py`)
   - Processes public events data
   - Assigns events to road segments
   - Propagates impact through network based on complexity
   - Expands temporally to 15-minute intervals
   - Output: `events_dataset.csv`

4. **Incidents** (`incidents.py`)
   - Processes traffic incidents data
   - Assigns incidents to road segments via spatial join
   - Aggregates by incident type and case specifics
   - Output: `incidents_dataset.csv`

5. **Irregularities** (`irregularities.py`)
   - Processes Waze traffic irregularities
   - Applies dual weighting (temporal + spatial overlap)
   - Slices temporally and intersects spatially with roads
   - Output: Multiple parquet files in batches

6. **Jams** (`jams.py`)
   - Processes Waze traffic jam data
   - Calculates jam statistics (velocity, delay, level)
   - Assigns jams to road segments
   - Output: Multiple parquet files

7. **PMTs Points** (`pmts_points.py`)
   - Processes traffic management plans (point-based)
   - Handles COI parks and points
   - Expands temporally based on schedules
   - Output: Parquet files by month

8. **PMTs Tracts** (`pmts_tracts.py`)
   - Processes traffic management plans (tract-based)
   - Handles COI tramos and COOS points
   - Output: Parquet files by month

9. **Road State** (`road_state.py`)
   - Processes road condition data
   - Uses AASHTO classification system
   - Assigns conditions to road segments
   - Output: `road_state_dataset.parquet`

10. **Weather** (`weather.py`)
   - Processes weather station data
   - Assigns stations to road segments
   - Interpolates weather features
   - Output: Separate parquet files per weather variable

### Step 2: Imputation Data Generation

Generates neighbor-aggregated features for missing data imputation:

- Uses k-hop neighborhood search (default: 10 hops)
- Computes mean features from neighbors with same street type
- Provides fallback values for nodes without direct observations
- Output: `neighbor_features.parquet`

**Important:** This step must run AFTER feature generation (needs graph and irregularities) but BEFORE dataset instantiation.

### Step 3: Dataset Merging

Converts all datasets to efficient HDF5 sparse format:

- **Static Features**: Time-invariant node attributes (road characteristics)
- **Dynamic Features**: Time-varying features (traffic, weather, events)
- **Target Variable**: Jam levels for prediction

Uses COO (Coordinate) sparse matrix format to optimize storage and loading:
- Stores only non-zero values
- Reduces file size by 90%+
- Enables efficient time-range queries

Output files in `data/datos_procesados/sparse_datasets/`:
- `static_dataset.h5` + metadata
- `events_dataset.h5` + metadata
- `incidents_dataset.h5` + metadata
- `irregularities_dataset.h5` + metadata
- `coi_points_dataset.h5` + metadata
- `coi_tracts_dataset.h5` + metadata
- `jams_dataset.h5` + metadata (target variable)
- `jam_velocity_dataset.h5` + metadata
- `jam_delay_dataset.h5` + metadata
- `jam_level_count_dataset.h5` + metadata
- Weather data files

### Step 4: Normalization

Computes robust normalization constants for all features:

- **Percentile-based normalization**: Uses 99th percentile of absolute values to determine scale
- **Outlier handling**: Values beyond the percentile are clipped to prevent loss of resolution
- **Feature-wise constants**: Each feature gets its own normalization constant
- **Comprehensive coverage**: Normalizes static features, dynamic features, targets, and weather data

Output files in `data/datos_procesados/sparse_datasets/`:
- `normalization_constants.pkl` - Dictionary of normalization constants
- `normalization_constants_summary.txt` - Human-readable summary of all constants

**Important:** This step must run AFTER dataset merging (needs HDF5 files) but BEFORE dataset instantiation.

**Usage:**
```bash
python -m src.preprocessing.normalize
```

**How it works:**
- For each feature, computes the 99th percentile of absolute values
- This becomes the normalization constant for that feature
- During training, features are divided by their constant and clipped to [-1, 1]
- Ensures all features have similar scales without losing resolution due to outliers

### Step 5: Dataset Instantiation

Creates and pickles the final `TrafficDataset` object:

- Loads all HDF5 files
- Loads normalization constants
- Configures time windows (150 input, 50 output timesteps)
- Sets up weather integration
- Configures imputation features
- Applies normalization during data loading
- Output: `traffic_dataset.pkl`

## Configuration

### Time Parameters

```python
in_timesteps = 150      # 37.5 hours of input (150 × 15min)
out_timesteps = 50      # 12.5 hours of output (50 × 15min)
node_batch_size = 500   # Number of nodes per batch
```

### Temporal Range

```python
min_timeframe = "2024-01-01 00:00:00"
max_timeframe = "2024-12-31 23:45:00"
freq = "15min"  # 15-minute intervals
```

### Future Features

These features are known in advance and available for future timesteps:

```python
future_features = [
    "coi_puntos",      # Traffic management plans (points)
    "coi_tramo",       # Traffic management plans (tracts)
    "AGLOMERACION",    # Crowding events
    "EMERGENCIA",      # Emergency events
    "MANIFESTACION",   # Protest/demonstration events
]
```

## Directory Structure

After running the pipeline, you'll have:

```
data/datos_procesados/
├── buildings/
│   └── roads_buildings.csv
├── bus_stops/
│   └── bus_stops_per_road.parquet
├── conectivity_graph/
│   ├── road_graph.pickle
│   ├── road_graph.graphml
│   └── road_graph.json
├── events/
│   └── events_dataset.csv
├── incidents/
│   └── incidents_dataset.csv
├── irregularidades_procesadas/
│   └── irregs_proc_*.parquet (multiple files)
├── jams_procesados/
│   └── jams_*.parquet (multiple files)
├── pmts_processed/
│   ├── coi_puntos/
│   │   └── *.parquet (multiple files)
│   └── coi_tramos/
│       └── *.parquet (multiple files)
├── road_state/
│   └── road_state_dataset.parquet
├── weather/
│   ├── stations.parquet
│   ├── direccion_viento.parquet
│   ├── humedad.parquet
│   ├── velocidad_viento.parquet
│   └── precipitacion.parquet
├── sparse_datasets/
│   ├── static_dataset.h5
│   ├── *_dataset.h5 (multiple HDF5 files)
│   ├── *_metadata.pkl (metadata files)
│   ├── normalization_constants.pkl
│   └── normalization_constants_summary.txt
├── neighbor_features.parquet
└── traffic_dataset.pkl  # Final dataset
```

## Usage After Preparation

Once the pipeline completes, you can load the dataset for training:

```python
import pickle
from torch_geometric.loader import DataLoader
from src.training_utils import pad_collate

# Load the prepared dataset
with open('data/datos_procesados/traffic_dataset.pkl', 'rb') as f:
    traffic_dataset = pickle.load(f)

# Create a DataLoader
dataloader = DataLoader(
    traffic_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    collate_fn=pad_collate,
)

# Start training
for batch in dataloader:
    # Your training loop here
    # All features are automatically normalized to [-1, 1]
    x_static = batch.x_static          # Static node features (normalized)
    x_dynamic_past = batch.x_dynamic_past    # Historical dynamic features (normalized)
    x_dynamic_future = batch.x_dynamic_future  # Future known features (normalized)
    x_weather = batch.x_weather        # Weather features (normalized)
    y = batch.y                        # Target (jam levels, normalized)
    edge_index = batch.edge_index      # Graph connectivity
    # ... model forward pass ...
```

## Inference Pipeline

The inference pipeline processes real-time data and generates predictions.

**Key Components:**
- **InferenceDataLoader** (`src/inference_dataloader.py`) - Main class for loading and processing real-time data
- **Production Script** (`src/inference_production.py`) - Production-ready inference with logging and monitoring

### Quick Start

#### Run Basic Inference Example
```bash
uv run python -m src.inference_dataloader
```

#### Production Inference
```bash
uv run python -m src.inference_production \
  --mode single \
  --graph data/datos_procesados/conectivity_graph/road_graph.pickle \
  --static data/datos_procesados/sparse_datasets/static_dataset.h5 \
  --norm data/datos_procesados/sparse_datasets/normalization_constants.pkl \
  --jams data/inference_tests/df_jams_test.parquet \
  --irregularities data/inference_tests/df_irregularidades_test.parquet \
  --weather data/inference_tests/df_clima_test.parquet \
  --pmt-puntos data/inference_tests/df_pmt_puntos_test.parquet \
  --pmt-tramos data/inference_tests/df_pmt_tramos_test.parquet
```

### Basic Usage

```python
from src.inference_dataloader import InferenceDataLoader
import torch

# Initialize the inference DataLoader
loader = InferenceDataLoader(
    graph_path="data/datos_procesados/conectivity_graph/road_graph.pickle",
    static_data_path="data/datos_procesados/sparse_datasets/static_dataset.h5",
    in_timesteps=150,  # 37.5 hours of input
    out_timesteps=50,   # 12.5 hours of prediction
    node_batch_size=500,
    normalization_constants_path="data/datos_procesados/sparse_datasets/normalization_constants.pkl",
    use_ray=False,  # Set to True for parallel processing
)

# Load real-time data and prepare batches
batches = loader.load_and_prepare_batch(
    jams_path="data/inference_tests/df_jams_test.parquet",
    irregularities_path="data/inference_tests/df_irregularidades_test.parquet",
    weather_path="data/inference_tests/df_clima_test.parquet",
    pmt_puntos_path="data/inference_tests/df_pmt_puntos_test.parquet",
    pmt_tramos_path="data/inference_tests/df_pmt_tramos_test.parquet",
)

# Make predictions with your model
model = YourModel.load("path/to/model.pth")
model.eval()

with torch.no_grad():
    batch = batches[0]
    predictions = model(
        x_static=batch.x_static.unsqueeze(0),
        x_dynamic_past=batch.x_dynamic_past.unsqueeze(0),
        x_dynamic_future=batch.x_dynamic_future.unsqueeze(0),
        x_weather=batch.x_weather.unsqueeze(0),
        edge_index=batch.edge_index,
    )

loader.cleanup()
```

### Key Features

- **Real-time Processing**: Loads data from parquet files (jams, irregularities, weather, PMTs)
- **Automatic Alignment**: Aligns real-time data with graph nodes and time windows
- **Missing Data Handling**: Gracefully handles missing features (events, incidents set to 0)
- **Normalization**: Applies the same normalization as training
- **Ray Support**: Optional parallel processing for faster inference
- **Memory Efficient**: Node batching to control memory usage

### Batch Structure

Each batch contains:
```
- x_static: (num_nodes, 117) - Static road features
- x_dynamic_past: (num_nodes, 150, 64) - Historical features
  * Jams: 10 features (velocity, level, delay stats)
  * Irregularities: 42 features (Waze data)
  * PMTs: 4 features (traffic management plans)
  * Events: 3 features (placeholder, set to 0)
  * Incidents: 5 features (placeholder, set to 0)
- x_dynamic_future: (num_nodes, 50, 4) - Known future features
  * PMTs only: coi_parques, coi_puntos, coi_tramo, coos_punto
- x_weather: (num_nodes, 150, 4) - Weather features
  * wind_direction, humidity, wind_speed, precipitation
- edge_index: (2, num_edges) - Graph connectivity
```

### Data Format Requirements

Real-time data must be in parquet format:

| File | Required Columns | Description |
|------|-----------------|-------------|
| `df_jams_test.parquet` | `id_mvi`, `timeframe`, vel/level/delay stats | Traffic jam data |
| `df_irregularidades_test.parquet` | `id_mvi`, `timeframe`, Waze features | Traffic irregularities |
| `df_clima_test.parquet` | `tipo_sensor`, `IDSENSOR`, `VALORLECTURA_mean`, `timestamp_local` | Weather data |
| `df_pmt_puntos_test.parquet` | `road_id`, `timeframe`, `coi_parques`, `coi_puntos` | Traffic management (points) |
| `df_pmt_tramos_test.parquet` | `road_id`, `timeframe`, `coi_tramo`, `coos_punto` | Traffic management (tracts) |

**Important Notes:**
- `id_mvi` and `road_id` must match graph node IDs
- `timeframe` should be in ISO format or parseable by pandas
- `timestamp_local` uses format `YYYYMMDD_HHMM`

### Performance Considerations

#### Sequential vs. Ray Processing

**Sequential (Default)**
- No extra dependencies
- Simpler error handling
- Better for small datasets (<1GB)
- Lower overhead

**Ray Parallel** (Enable with `use_ray=True`)
- Faster for large datasets
- Parallelizes file loading and processing
- Requires: `uv add ray`
- Higher memory usage
- Best for production environments

#### Memory Management

For large-scale inference:
```python
# Process specific roads instead of all nodes
important_roads = ["0_FT.5", "0_FT.10", ...]
batches = loader.load_and_prepare_batch(
    ...,
    node_seeds=important_roads,
)

# Or reduce batch size
loader = InferenceDataLoader(..., node_batch_size=100)
```

### Integration Patterns

#### Real-Time Streaming
```python
import time

loader = InferenceDataLoader(...)

while True:
    time.sleep(900)  # Wait 15 minutes
    latest_files = get_latest_data_files()
    batches = loader.load_and_prepare_batch(**latest_files)
    predictions = model(batches[0])
    save_predictions(predictions)
```

#### Batch Processing
```python
# Process historical data
for date in date_range:
    file_paths = get_files_for_date(date)
    batches = loader.load_and_prepare_batch(**file_paths)
    predictions = model(batches[0])
    save_predictions(date, predictions)
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| FileNotFoundError with Ray | Use absolute paths or set `use_ray=False` |
| Missing road IDs | Normal - roads not in graph are skipped |
| Memory errors | Reduce `node_batch_size` or process fewer nodes |

### Differences from Training Pipeline

| Aspect | Training | Inference |
|--------|----------|-----------|
| Data Source | HDF5 sparse matrices | Parquet files |
| Target Variable | Required (jam levels) | Not needed |
| Time Window | Sliding windows over historical data | Latest available data |
| Events/Incidents | Full historical data | Set to 0 (not yet available) |
| Batching | All nodes via seeds | Configurable node subsets |
| Performance | Optimized for throughput | Optimized for latency |

## Troubleshooting

### Missing Data Files

If you get file not found errors, ensure you have:
- Raw data in `data/` subdirectories
- Correct file names matching the script

### Memory Errors

If you run out of memory:
- Close other applications
- Reduce batch size in jams/irregularities processors
- Process files in smaller chunks

### Import Errors

Ensure all dependencies are installed. To synchronize the local environment with the requirements in `uv.lock`, run the command
```bash
uv sync
```

If `uv` is not installed in your machine, just install it with
```bash
pip install uv
```

## Troubleshooting
- If `uv` is not installed in your machine, just install it with `pip install uv`.
- All the necessary Python packages must be installed. This is done by executing `uv sync`.
- Then, the correct environment variables must be set. See `.env.example` for an example.

## Downloading the data
To download the data in the `data/` directory, run:
```
uv run -- dotenv run -- aws s3 sync s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/ ./data
```

## Using `uv`
- To add dependencies to the project, run `uv add <package_name>`.
- To remove dependencies from the project, run `uv remove <package_name>`.
- To add dependencies to the project that will only be used during development (such as `jupyer`, `pytest` or `ruff`), run `uv add --dev <package_name>`.
- To synchronize the local environment with the requirements in `uv.lock`, run the command `uv sync`.
- A full list of `uv` commands can be found [here](https://docs.astral.sh/uv/reference/cli/).

## Rendering Jupyter notebooks
To avoid large Jupyter files in the repository, we use `jupytext` to transform the `.ipynb` files to plain text documents.
This allows easy editing and clear version control, as well as a largely reduced repository size.

To pair all `.ipynb` in the directory to `.py` files, run:
```
uv run jupytext --set-formats ipynb,py:percent notebooks/*.ipynb
```

To render a paired `.py` script back to its `.ipynb` format and synchronize its changes, run:
```
uv run jupytext --sync notebooks/<notebook_name>.py
```

For convenience, these commands are in the `Makefile` described below.

## Using the `Makefile`
- Run `make test` to run all the tests inside the `tests/` directory.
- Run `make test_cov` to get the test coverage report.
- Run `make format` to format all the code using `ruff`.
- Run `make to_jupyter` to convert all `.py` files in the `notebooks/` directory to their paired `.ipynb` notebooks.
- Run `make from_jupyter` to convert all `.ipynb` notebooks to their `.py` format ready to be pushed to the repository. -->