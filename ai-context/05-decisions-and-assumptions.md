# Decisions And Assumptions

## Durable Technical Decisions

### 1. One batch must keep one seed
- Status: active
- Reason:
  - `edge_index` is shared across all examples in the batch
  - changing seed inside the batch would break the current dataset contract or make batch construction much more expensive

### 2. Time indices may vary within the batch
- Status: active
- Reason:
  - this preserves the shared graph structure while still allowing temporal diversity

### 3. The dataset fast path should remain unchanged
- Status: active
- Current owner:
  - `src/sparse_dataloader.py`
- Reason:
  - performance matters
  - the sampler/planner should change batch composition, not the core tensor materialization logic

### 4. Sparse missing target values count as zero
- Status: active
- Reason:
  - the sparse representation does not explicitly materialize all `0` targets
  - scoring that ignores missing-as-zero severely overstates congestion severity

### 5. Seed-time artifacts are the durable control layer for the new sampler
- Status: active
- Reason:
  - expensive scoring should happen offline
  - training should consume a precomputed artifact instead of recomputing severity live

### 6. The consolidated artifact can live in S3, not necessarily EFS
- Status: active
- Reason:
  - artifact size is small enough for quick download
  - `training_entrypoint.py` now supports S3 download + local cache

### 7. One epoch should reflect one pass over all seeds when using the new sampler
- Status: active
- Current parameter:
  - `seed_time_epoch_cycles = 1`
- Reason:
  - this makes epoch semantics clearer and aligns with the new seed universe

### 8. The current target bucket mix is high-skewed on purpose
- Status: active experiment design
- Target:
  - `45% high`
  - `30% medium`
  - `25% low`
- Reason:
  - the previous training mix appeared too dominated by easy low-severity cases

### 9. Temporal trajectory shape now matters explicitly in training
- Status: active experiment
- Mechanism:
  - auxiliary node-wise Pearson loss over the forecast horizon
- Reason:
  - for operations, it is not enough to predict the next step correctly
  - the model should better preserve the full future congestion trajectory per node
  - this is especially important if downstream calibration can later correct level scale

## Working Assumptions
- Main AWS region is `us-east-1`
- SageMaker training image is stable enough for the current experiment line
- `batch_size = 16` remains the training batch size
- current active model comparison line is:
  - `GConvLSTM`
  - `GConvGRU`
  - `TGCN`
- `A3TGCN` remains a historical comparison point, but not the latest active 3-model line
- if a task depends on exact current job state, re-check SageMaker rather than trusting this file blindly
