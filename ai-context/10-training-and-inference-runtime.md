# Training And Inference Runtime

## Training Runtime

### Canonical entrypoint
- `src/training_entrypoint.py`

### High-level flow
1. parse CLI args
2. resolve SageMaker train/validation channels
3. optionally copy/cache required files locally
4. instantiate `TrafficDataset`
5. split by time into train/validation subsets
6. build model
7. choose sampler path
8. train for configured epochs
9. evaluate on validation
10. emit metrics and artifacts

## Sampler Paths

### Old path
- `SeededBatchSampler`
- uses:
  - `train_seed_list`
  - `train_time_indices`

### New path
- enabled by:
  - `--seed-time-artifact-path`
- uses:
  - `SeedTimeEpochPlanner`
  - `PlannedSeedTimeBatchSampler`

## New Epoch Planning Semantics

### Planner output
- ordered seed list for the epoch
- `time_idx` list per batch

### Batch constraints
- one seed only
- many time windows in that batch

### Current intended epoch size
- with combined artifact:
  - `1937` seeds
  - `batch_size = 16`
  - `seed_time_epoch_cycles = 1`
- so:
  - `1937` batches
  - `30,992` sample slots per epoch

## Runtime Metrics

### Training loop logs
- loss metrics
- operational metrics based on rounded congestion levels
- timing metrics:
  - `time_data_wait`
  - `time_transfer`
  - `time_forward`
  - `time_loss`
  - `time_backward`
  - `time_step`
  - `time_total`

### SageMaker metric limits
- Training jobs currently use a reduced metric set to stay within SageMaker metric definition limits.
- This means not every theoretically available metric is surfaced as a parsed SageMaker metric.

## Current Loss Stack

### Canonical implementation
- `src/training/losses.py`

### Base congestion loss
- the training objective still uses the existing composite congestion loss:
  - continuous regression term
  - differentiable class CE term
  - ordinal class-distance penalty
  - large-error penalty
  - high-level MAE penalty
  - severe-level MAE penalty
  - underprediction penalty

### New shape component
- a new auxiliary loss term was added:
  - node-wise Pearson shape loss over the full forecast horizon
- active parameters:
  - `--loss-shape-lambda`
  - `--loss-shape-min-std`
  - `--loss-shape-eps`
  - `--loss-shape-weight-std`
  - `--loss-shape-weight-range`
  - `--loss-shape-weight-max`

### Shape-loss semantics
- compute the level channel in real level space (`0..4`)
- treat the last tensor dimension as forecast time
- flatten all leading dimensions into independent node-series
- compute Pearson correlation between predicted and target future trajectory for each node-series
- convert to loss with:
  - `l_i = 1 - corr_i`
- aggregate with weighted mean:
  - larger weight for node-series with higher:
    - target standard deviation
    - target range
    - target maximum level

### Why this was added
- the existing loss already constrains level accuracy and severe errors
- it did not directly reward preserving the temporal shape of the future trajectory
- the new term is meant to help the model follow:
  - when congestion rises
  - how the full 50-step trajectory evolves
  - where peaks and declines occur for each node

### Important caveats
- this is not a calibration term
- it is a shape-following term
- if `loss-shape-lambda` is too large, the model may preserve trajectory shape while giving up too much absolute level accuracy
- first experiments should keep it small, e.g. `0.05`

## Inference Runtime

### Canonical components
- `src/inference_dataloader.py`
- `src/inference_production.py`
- `src/prediction_reconstruction.py`

### Inference differs from training because
- it reads current parquet source data instead of training HDF5 targets
- it has no target `y`
- it reconstructs predictions back into node/timestamp outputs

## Important Runtime Invariants

### Preserve performance
- The project intentionally keeps the core dataset materialization logic stable.
- The new sampling logic is externalized into planner/sampler code.

### Preserve graph consistency
- One batch must share one `edge_index`
- Therefore one batch must share one seed-local subgraph
