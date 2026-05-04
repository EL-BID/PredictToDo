# Architecture And Dataflow

## End-To-End Runtime Shape

1. raw sparse graph-temporal data is read from EFS
2. `TrafficDataset` materializes local graph samples
3. a sampler chooses `(seed, time_idx)` combinations
4. batches are built with:
   - one seed per batch
   - multiple time indices in the same batch
5. model forward/loss/metrics run inside `src/training/loop.py`
6. SageMaker captures selected metrics from logs

## Sparse Dataset Contract

### Core rule
- One batch must keep a single seed.
- Reason:
  - all examples in the batch share the same `edge_index`
  - mixing seeds inside a batch would break that contract or force expensive re-materialization

### What can vary within the batch
- `time_idx`

### Why this matters
- The artifact is row-based on `(seed_id, time_idx)`, but the runtime unit is actually:
  - one seed
  - plus a list of `time_idx`

## Old Training Sampling Path

### Old logic
- `train_seed_list`
- `train_time_indices`
- `SeededBatchSampler`

### Behavior
- one seed per batch
- multiple time windows per batch
- but no sample-aligned severity artifact

## New Training Sampling Path

### Enabling condition
- pass `--seed-time-artifact-path`

### New logic
1. load the consolidated parquet
2. group rows by:
   - `seed_id`
   - `bucket`
3. at epoch start:
   - build an epoch plan
   - decide seed order
   - decide `time_idx` list per batch
4. set dataset seed list to the plan order
5. sampler maps each batch to standard dataset global indices

### Important design choice
- `TrafficDataset.__getitem__()` was not redesigned
- this keeps the fast path and avoids slowing sample materialization

## Artifact Download / Cache Flow

If `--seed-time-artifact-path` points to `s3://...`:
1. `training_entrypoint.py` downloads it locally
2. if `/data/datos_procesados` is writable/mounted, it also tries to persist it under:
   - `/data/datos_procesados/seed_time_artifacts/`
3. training uses the local cached copy

This means the runtime does not depend on the artifact already being present in EFS.

## Epoch Semantics Under The New Sampler

### Current intended behavior
- `seed_time_epoch_cycles = 1`
- one epoch = one pass over every seed in the artifact

### For the combined artifact
- `1937` seeds
- `batch_size = 16`
- therefore:
  - `1937` batches per epoch
  - `30,992` sample slots per epoch

## Bucket Mix Semantics

The target ratio discussed and implemented in planner configuration is:
- `45% high`
- `30% medium`
- `25% low`

With `batch_size = 16`, the practical per-batch approximation is:
- `7 high`
- `5 medium`
- `4 low`

That gives:
- `43.75% high`
- `31.25% medium`
- `25% low`

The difference is small and acceptable for the first runtime path.
