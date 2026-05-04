# Training

## Entry Point

The SageMaker-compatible training entrypoint is:

```bash
python -m src.training_entrypoint
```

In practice, use `notebooks/sagemaker_training_job.ipynb` to launch jobs with the correct image, EFS mount, VPC settings, model family, and artifact paths.

## Active Model Families

The current comparison line focuses on:

- `GConvLSTM`
- `GConvGRU`
- `TGCN`

`A3TGCN` remains a supported or historical architecture point, but it is not the center of the latest three-model comparison line documented in `ai-context/06-current-workspace-state.md`.

## Sampling Paths

The training runtime supports two paths:

- legacy seed-based sampling
- artifact-driven seed-time sampling enabled by `--seed-time-artifact-path`

The artifact-driven path plans batches over `(seed_id, time_idx)` rows while preserving the invariant that each batch uses one seed-local graph.

## Current Artifact Direction

The current seed-time artifact direction uses a combined seed universe and a bucket mix intended to increase informative congestion exposure. See [rebalance_sampler_design.md](rebalance_sampler_design.md).

## Loss Stack

The training objective combines congestion-level regression/classification penalties with a newer auxiliary shape-following term over the forecast horizon.

Shape-loss controls include:

- `--loss-shape-lambda`
- `--loss-shape-min-std`
- `--loss-shape-eps`
- `--loss-shape-weight-std`
- `--loss-shape-weight-range`
- `--loss-shape-weight-max`

## Known Training Gaps

- Shape-loss metrics are not yet surfaced explicitly enough in SageMaker logs.
- The new seed-time planner and sampler need targeted unit tests.
- Checkpoints need durable model identity metadata for downstream inference traceability.
