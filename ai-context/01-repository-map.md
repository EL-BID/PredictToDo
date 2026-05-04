# Repository Map

## Workspace Shape
- This repo mixes:
  - core runtime code under `src/`
  - notebooks used for experiments and SageMaker job launching
  - operational scripts under `scripts/`
  - tests under `tests/`
  - human-facing docs under `docs/`
  - AI handoff notes under `ai-context/`
  - paper source under `white_paper/`

## Core Runtime Paths

### `src/`
- Main runtime code.
- Important files:
  - `src/sparse_dataloader.py`
  - `src/training_entrypoint.py`
  - `src/training_utils.py`
  - `src/models/gnn_temporal.py`
  - `src/training/loop.py`
  - `src/training/losses.py`

### `src/sparse_dataloader.py`
- Owns dataset materialization.
- Key property:
  - global sample index is interpreted as `seed_slot * num_time_windows + time_idx`
- This contract is intentionally preserved by the new sampler to keep performance stable.

### `src/training_entrypoint.py`
- Main SageMaker training entrypoint.
- Now supports:
  - old seeded sampler path
  - new seed-time artifact path
  - S3 download/cache for the artifact

### `src/training_utils.py`
- Contains sampling/planning utilities.
- Now includes:
  - `SeedTimeEpochPlanner`
  - `PlannedSeedTimeBatchSampler`

### `src/rebalance_sampling.py`
- Offline builder for `(seed_id, time_idx)` scoring artifacts.
- Supports shard-based execution and partial checkpoint uploads.

### `src/seed_strategy.py`
- Builds seed sets with coverage + congestion logic.

### `src/time_index_strategy.py`
- Builds the selected `time_idx` list used in the seed-time artifact runs.

### `src/merge_seed_time_artifacts.py`
- Merges shard outputs into a consolidated final parquet.

## Notebook Layer

### `notebooks/sagemaker_training_job.ipynb`
- Primary training launcher.
- Was modified to:
  - include `model_type`
  - separate output prefixes by model
  - support the new seed-time sampler flags

### `notebooks/sagemaker_rebalance_probe.ipynb`
- Probe job launcher for time-only rebalancing experiments.

### `notebooks/sagemaker_seed_time_artifact_builder.ipynb`
- Earlier artifact-building notebook for non-sharded or smaller seed-time experiments.

### `notebooks/sagemaker_seed_time_sharded_launcher.ipynb`
- Launches shard-based seed-time artifact generation.

## Artifact Layer

Large artifacts are not tracked in Git. Local artifact paths may exist under ignored folders such as `data/artifacts/`, while canonical shared artifacts are usually in S3 or EFS.

### `data/artifacts/seed_time_merged/`
- Consolidated artifact for the initial `937`-seed universe.

### `data/artifacts/seed_time_merged_extra1000/`
- Consolidated artifact for the later `1000`-seed expansion.

### `data/artifacts/seed_time_merged_all/`
- Final combined artifact used by the new training sampler.

### `data/artifacts/seed_strategy/`
- Coverage + congestion seed set and shard manifests for the first universe.

### `data/artifacts/seed_strategy_extra1000/`
- Additional seed set and shard manifests for the expansion.

## Design Notes

### `docs/rebalance_sampler_design.md`
- Durable rationale for why the repo moved from time-only sampling to sample-aligned `(seed, time_idx)` planning.
