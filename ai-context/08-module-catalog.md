# Module Catalog

This file organizes the repository by module responsibility rather than by chronology.

## Core Dataset Layer

### `src/sparse_dataloader.py`
- Main dataset implementation.
- Central class:
  - `TrafficDataset`
- Responsibilities:
  - load graph and static features into memory
  - lazy-load dynamic sparse HDF5 features on demand
  - load target sparse HDF5 data
  - support weather integration
  - support imputation features from neighbor aggregates
  - normalize static, dynamic, target, and weather features
  - construct seed-local node subgraphs
  - map flat integer dataset indices to:
    - seed slot
    - `time_idx`
- Important internal ideas:
  - `batch_mapping` maps each seed to a reachable node subset
  - target features can be:
    - level only
    - level + velocity
  - missing sparse entries are effectively treated as zero in reconstruction

### `src/temporal_split.py`
- Temporal wrappers around `TrafficDataset`.
- Key pieces:
  - `TemporalSubset`
  - `temporal_train_test_split`
  - `temporal_train_val_test_split`
- Purpose:
  - split time windows chronologically while sharing the same underlying dataset

## Training Runtime Layer

### `src/training_entrypoint.py`
- Main SageMaker entrypoint.
- Responsibilities:
  - parse CLI arguments
  - verify SageMaker channels
  - optionally cache/copy input files
  - build `TrafficDataset`
  - split into train/val by time
  - instantiate model and optimizer/scaler
  - choose the sampling path:
    - old seeded sampler
    - new seed-time planner path
  - run training and validation loops
  - log metrics in SageMaker-friendly format

### `src/training_utils.py`
- Sampling and collation utilities.
- Key functions/classes:
  - `pad_collate`
  - `SeededBatchSampler`
  - `SeedTimeEpochPlan`
  - `SeedTimeEpochPlanner`
  - `PlannedSeedTimeBatchSampler`
  - `Node50Picker`
  - `EpochUniquePicker`
- Important distinction:
  - `SeededBatchSampler` is the older runtime path
  - `SeedTimeEpochPlanner` + `PlannedSeedTimeBatchSampler` are the new artifact-driven path

### `src/training/loop.py`
- Contains `train_one_epoch`.
- Handles:
  - forward pass
  - loss computation
  - backward pass
  - optimizer stepping
  - timing metrics

### `src/training/losses.py`
- Contains `masked_mse`.
- The codebase still uses explicit level/velocity splits and additional operational metrics in the entrypoint.

## Model Layer

### `src/models/gnn_temporal.py`
- Contains:
  - `make_batched_edge_index`
  - `StaticEncoder`
  - `GNN_Temporal_Model`
- Purpose:
  - central runtime wrapper over supported GNN-temporal architectures
  - handles graph batching and static feature encoding

## Preprocessing / Dataset Build Layer

### `src/prepare_dataset.py`
- High-level orchestration script for building the processed dataset.
- Major stages:
  - feature generation
  - imputation generation
  - sparse dataset merging
  - dataset instantiation

### `src/preprocessing/features/*.py`
- Feature-specific processors.
- Main processor classes:
  - `BuildingsProcessor`
  - `BusStopsProcessor`
  - `RoadGraphConstructor`
  - `EventsProcessor`
  - `IncidentsProcessor`
  - `IrregularitiesProcessor`
  - `JamsProcessor`
  - `PMTSPointsProcessor`
  - `PMTSTractsProcessor`
  - `RoadStateProcessor`
  - `WeatherProcessor`

### `src/preprocessing/imputation.py`
- Builds neighbor-based imputation features.
- Uses graph structure + aggregated irregularity features.

### `src/preprocessing/merging.py`
- Converts processed source datasets into sparse HDF5 datasets.

### `src/preprocessing/normalize.py`
- Computes normalization constants from sparse HDF5 files.
- Uses percentile-based scaling to avoid outlier domination.

## Inference Layer

### `src/inference_dataloader.py`
- Runtime loader for real-time inference.
- Different from training:
  - no target `y`
  - reads current parquet inputs rather than training HDF5 target files
  - can use Ray

### `src/inference_production.py`
- Production inference pipeline.
- Responsible for:
  - loading data and model
  - running prediction
  - exporting reconstructed outputs

### `src/prediction_reconstruction.py`
- Reconstructs batch predictions back to:
  - `NodeID`
  - `Timestamp`
- Useful for post-processing and downstream consumption

## Artifact / Sampling Builder Layer

### `src/rebalance_sampling.py`
- Builds shardable `(seed_id, time_idx)` artifacts with scores and buckets.
- Supports:
  - S3 input download
  - S3 checkpoint upload
  - partial checkpoints
  - quantile bucketization

### `src/seed_strategy.py`
- Builds strategic seed sets:
  - coverage
  - top congestion

### `src/seed_sharding.py`
- Splits seed lists into multiple shard CSVs.

### `src/time_index_strategy.py`
- Builds the selected `time_idx` universe for artifact generation.

### `src/merge_seed_time_artifacts.py`
- Merges shard outputs into final parquets and metadata.

## Utility Layer

### `src/utils/logging.py`
- Shared logging helpers used by training runtime.

### `src/utils/pickers.py`
- Older selection utilities tied to earlier sampling logic.

### `src/utils/tensor_stats.py`
- Tensor diagnostics helpers.

## Duplicate / Legacy-Like Files To Treat Carefully
- `src/sparse_dataloader(2).py`
- `src/sparse_dataloader_jose.py`
- `src/training_entrypoint(2).py`
- `src/training_utils(2).py`

These appear to be alternate or older copies and should not be treated as the canonical runtime unless explicitly chosen.
