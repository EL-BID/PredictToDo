# Preprocessing Pipeline

## Purpose
- Convert raw geospatial, traffic, event, PMT, weather, and road-state inputs into a sparse graph-temporal training dataset.

## Orchestration Entry Point
- `src/prepare_dataset.py`

## Stage 1: Feature Generation

### Static / structural features
- buildings
- bus stops
- graph construction
- road state

### Dynamic / temporal features
- events
- incidents
- irregularities
- jams
- PMT points
- PMT tracts
- weather

## Stage 2: Neighbor-Based Imputation

### Entry
- `src/preprocessing/imputation.py`

### Core idea
- Some features such as trend/speed-like signals are sparse or missing.
- The pipeline computes neighbor aggregates over the graph, constrained by street type when possible.
- Output is later used by the dataset for feature fallback/imputation.

### Test coverage confirms
- disconnected node handling
- street-type means
- neighbor aggregation
- fallback imputation behavior

## Stage 3: Sparse Dataset Merging

### Entry
- `src/preprocessing/merging.py`

### Output shape
- sparse HDF5 files with only non-zero entries

### Why sparse HDF5 matters
- raw traffic-like signals are extremely sparse
- dense storage would be too large and too slow

## Stage 4: Normalization

### Entry
- `src/preprocessing/normalize.py`

### Strategy
- percentile-based absolute-value scaling
- feature-wise constants
- protection against outliers
- protection against very small scales via minimum floor

### Test coverage confirms
- dynamic and static handling
- outlier robustness
- negative-value handling
- missing feature handling

## Stage 5: Dataset Instantiation

### Result
- `TrafficDataset` can be created over the produced sparse HDF5 structure
- graph and static features stay in memory
- dynamic features and targets are fetched lazily by time window

## Durable Conceptual Model

### Data model
- graph nodes are road entities
- each temporal source becomes sparse feature observations indexed by:
  - node
  - timeframe
  - feature

### Target model
- target file must contain `mean_jam_level`
- may also contain `mean_jam_velocity`

## Important Caveat
- There are duplicate or experimental preprocessing files such as `merging2.py`.
- The canonical path for the main documented pipeline remains:
  - `prepare_dataset.py`
  - `preprocessing/merging.py`
  - `preprocessing/normalize.py`
