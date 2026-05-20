# Process Runbooks

This document gives procedural steps for common workflows. It is intentionally operational: use it with `docs/code-reference.md` when editing code.

## Local Setup

1. Create or activate a Python 3.12 environment.
2. Install dependencies with `uv sync --all-groups` when available.
3. Install matching PyTorch Geometric wheels if local `uv sync` tries to compile them from source.
4. Run static validation before editing:

```bash
python -m py_compile src/training_entrypoint.py src/training/losses.py
```

## Build Or Refresh The Dataset

1. Confirm raw data paths and AWS credentials.
2. Run the preprocessing entrypoint:

```bash
uv run python -m src.prepare_dataset
```

3. Check generated graph, static, temporal, and normalization artifacts.
4. Confirm feature columns against the expected feature-order contract.
5. Run targeted preprocessing tests if dependencies are available:

```bash
python -m pytest tests/preprocessing tests/test_sparse_dataloader.py
```

## Compute Global Seed Coverage

Use:

```bash
python scripts/compute_seed_cover.py --help
```

The expected output is a seed list that covers the graph under the configured hop distance. This is an inference coverage artifact, not a seed-time training sampler.

## Build Seed-Time Training Artifacts

The seed-time pipeline creates `(seed_id, time_idx)` rows for balanced training exposure.

Primary modules:

- `src/seed_strategy.py`
- `src/time_index_strategy.py`
- `src/rebalance_sampling.py`
- `src/merge_seed_time_artifacts.py`
- `src/seed_sharding.py`

Procedure:

1. Build or select seed and time-index strategies.
2. Generate sharded seed-time rows if the full job is too large.
3. Merge shards into a final parquet artifact.
4. Confirm bucket counts and target ratios.
5. Pass the artifact to training with `--seed-time-artifact-path`.

## Build Regional Artifacts

1. Build `node_region_map.csv`:

```bash
python scripts/build_corridor_region_map.py --help
```

2. Build regional seed strategies:

```bash
python scripts/build_region_seed_strategies.py --help
```

3. Prepare or retry regional seed-time shard jobs:

```bash
python scripts/prepare_region_seed_time_jobs.py --help
python scripts/retry_failed_seed_time_jobs.py --help
```

4. Merge final regional seed-time artifacts:

```bash
python scripts/finalize_region_seed_time_artifacts.py --help
```

5. Validate that central and noncentral artifacts are separately bucketed and reference the same `node_region_map.csv`.

## Launch SageMaker Training

Use `notebooks/sagemaker_training_job.ipynb` for the standard launch path. The entrypoint is:

```bash
python -m src.training_entrypoint
```

For global runs, use a global seed-time artifact. For regional runs, provide:

```text
--training-region central|noncentral
--node-region-map-path s3://.../node_region_map.csv
--seed-time-artifact-path s3://.../seed_time_bucket_rows_<region>.parquet
```

Before calling a model final, check:

- training completed without non-finite losses;
- validation operational metrics are present;
- region masks selected nonzero nodes;
- model artifact includes feature-order and config files;
- logs identify model family, input lengths, output horizon, seed artifact, and region.

## Export Calibration Artifacts

Use:

```bash
python scripts/export_calibration_artifacts.py --help
```

Calibration artifacts should be versioned with the model or evaluation window used to build them. Do not deploy calibration from a different model line without documenting the mismatch.

## Run Dense Real-Time Evaluation

Use the scripts:

```bash
python scripts/realtime_eval_backfill.py --help
python scripts/realtime_eval_last7d_dense.py --help
python scripts/realtime_eval_metrics.py --help
python scripts/realtime_eval_sample.py --help
```

Dense evaluation should specify:

- prediction source;
- jam source;
- date range;
- whether missing jams are filled as level `0`;
- whether predictions are rounded/clipped for operational metrics.

## Prepare Serving Artifacts

For the real-time service, publish or hand off:

- model tarball;
- extracted `model.pth`;
- `model_config.pkl`;
- `feature_order.json`;
- normalization constants;
- graph pickle;
- static dataset;
- seed list or regional seed lists;
- `node_region_map.csv` for regional execution;
- calibration artifacts if enabled.

Update the inference repository documentation when any of these contracts changes.

