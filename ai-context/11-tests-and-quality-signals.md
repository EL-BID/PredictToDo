# Tests And Quality Signals

## Existing Test Coverage

### Dataset / dataloader
- `tests/test_sparse_dataloader.py`
- Covers:
  - synthetic graph/data generation
  - static and dynamic sparse HDF5 loading
  - target loading
  - weather support
  - index semantics

### Temporal splitting
- `tests/test_temporal_split.py`
- Covers:
  - chronological split behavior
  - subset indexing
  - sample count semantics

### Prediction reconstruction
- `tests/test_prediction_reconstruction.py`
- Covers:
  - mapping predictions back to node/timestamp outputs
  - temporal alignment
  - node ordering
  - custom column naming

### Preprocessing imputation
- `tests/preprocessing/test_imputation.py`
- Covers:
  - graph loading
  - irregularities aggregation
  - disconnected node handling
  - street type extraction
  - neighbor feature generation
  - imputation fallbacks

### Preprocessing normalization
- `tests/preprocessing/test_normalize.py`
- Covers:
  - static and temporal normalization constants
  - percentile/outlier behavior
  - negative values
  - minimum scale logic
  - multi-dataset aggregation

## What Is Not Well Covered Yet
- The new seed-time runtime integration in:
  - `src/training_utils.py`
  - `src/training_entrypoint.py`
- SageMaker-specific artifact download/cache flow
- full online inference pipeline integration
- end-to-end training loop correctness under the new sampler

## Practical Quality Signals Beyond Unit Tests

### Artifact generation line
- The shard-based seed-time builder completed successfully for:
  - initial `937` seeds
  - extra `1000` seeds
- merged artifacts exist and look internally consistent

### Resource analysis
- Previous training jobs were inspected in CloudWatch for:
  - host memory
  - GPU memory
  - GPU utilization
  - CPU utilization

### Live runtime validation now in progress
- Four live SageMaker training jobs are running with the new sampler.
- This is the most important current validation layer.

## Recommendation For Future Quality Work
- Add targeted unit tests for:
  - `SeedTimeEpochPlanner`
  - `PlannedSeedTimeBatchSampler`
  - artifact path resolution from S3/local/EFS
- Add a lightweight integration test for:
  - one epoch plan over a toy artifact
