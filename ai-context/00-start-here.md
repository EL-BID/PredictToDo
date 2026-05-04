# Start Here

## What This Repository Is
- A traffic congestion forecasting repository for Bogota.
- It builds sparse graph-temporal datasets, trains GNN-temporal models, and runs SageMaker jobs for training and sampling experiments.
- The current practical center of gravity is:
  - sparse temporal dataset construction
  - SageMaker training for 3 main model families now actively compared (`GConvLSTM`, `GConvGRU`, `TGCN`)
  - seed-time artifact generation for better training sampling
  - offline and real-time evaluation against dense `prediction + jams fillna(0)` universes
  - calibration and inference-traceability cleanup

## What Not To Assume
- Do not assume the current training loop still uses the original seeded sampler only.
- Do not assume sampling is global by `time_idx` only.
- Do not assume the artifact must already exist in EFS; the entrypoint now supports `s3://...` seed-time artifacts and caches them locally.
- Do not assume epochs are tied only to `len(train_time_indices)` anymore when the new sampler is enabled.
- Do not assume the real-time inference repository shares the exact same code as this repo; it is a sibling repo with overlapping concepts but separate implementation constraints.
- Do not assume prediction parquet backups in the inference repo carry model version metadata. They currently do not.
- Do not assume Redis and S3 prediction outputs have identical semantics. Redis main keys are filtered, S3 backups are intended to be full-surface.

## Current High-Level Direction
- Training was previously dominated by low-severity samples and class `0`.
- The repo now has a precomputed `(seed_id, time_idx)` artifact and a new planner-based sampler.
- Each batch should use:
  - exactly one seed
  - multiple `time_idx`
  - a bucket mix close to `45% high / 30% medium / 25% low`

## Default Reading Order For Any Agent
1. `01-repository-map.md`
2. `02-architecture-and-dataflow.md`
3. `03-seed-time-sampling-status.md`
4. `04-artifacts-and-locations.md`
5. `05-decisions-and-assumptions.md`
6. `06-current-workspace-state.md`
7. `07-open-work-and-next-steps.md`

## Fast Task Triage

### Understand training runtime
- Start with:
  - `02-architecture-and-dataflow.md`
  - `03-seed-time-sampling-status.md`

### Modify sampling logic
- Start with:
  - `src/training_utils.py`
  - `src/training_entrypoint.py`
  - `03-seed-time-sampling-status.md`

### Rebuild or analyze seed-time artifacts
- Start with:
  - `src/rebalance_sampling.py`
  - `src/seed_strategy.py`
  - `src/seed_sharding.py`
  - `src/time_index_strategy.py`
  - `src/merge_seed_time_artifacts.py`
  - `04-artifacts-and-locations.md`

### Launch SageMaker training
- Start with:
  - `notebooks/sagemaker_training_job.ipynb`
  - `06-current-workspace-state.md`

### Understand the current model line
- Start with:
  - `06-current-workspace-state.md`
  - `07-open-work-and-next-steps.md`
  - `10-training-and-inference-runtime.md`

### Analyze or modify real-time scoring
- Start with:
  - `14-real-time-inference-repo.md`
- External sibling repository path:
  - `<inference-repo-root>`

## Operating Assumptions
- AWS region: `us-east-1`
- Main training image:
  - `195419001736.dkr.ecr.us-east-1.amazonaws.com/congestion-bogota-sagemaker:temporal-features-20260331-194332`
- Training data comes from EFS:
  - `fs-099e29197c0a3e353`
  - `/data/datos_procesados`
- Canonical AI handoff layer:
  - `ai-context/`
- Related real-time inference repository:
  - `<inference-repo-root>`
