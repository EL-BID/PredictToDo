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

`A3TGCN` is also supported by the model implementation.

## Sampling Paths

The training runtime supports two paths:

- legacy seed-based sampling
- artifact-driven seed-time sampling enabled by `--seed-time-artifact-path`

The artifact-driven path plans batches over `(seed_id, time_idx)` rows while preserving the invariant that each batch uses one seed-local graph.

## Current Artifact Direction

The current seed-time artifact direction uses a combined seed universe and a bucket mix intended to increase informative congestion exposure. See [rebalance_sampler_design.md](rebalance_sampler_design.md).

Regional training uses separate central and noncentral artifacts plus `--training-region` and `--node-region-map-path`. See [artifacts-and-contracts.md](artifacts-and-contracts.md) for the regional data contract.

## SageMaker Launch Configuration

The current notebook launch path should preserve the successful global training configuration from the latest completed shape-loss line:

- region `us-east-1`
- role `arn:aws:iam::195419001736:role/SagemakerExecutionRole`
- image `195419001736.dkr.ecr.us-east-1.amazonaws.com/congestion-bogota-sagemaker:temporal-features-20260331-194332`
- EFS `fs-099e29197c0a3e353` mounted from `/data/datos_procesados`
- seed-time artifact `s3://sagemaker-congestion-bogota/congestion-bogota/seed-time-artifacts/final_all/seed_time_bucket_rows_merged_all.parquet`
- global model instances: `GConvGRU` on `ml.g5.4xlarge`, `GConvLSTM` on `ml.g4dn.8xlarge`, and `TGCN` on `ml.g5.8xlarge`
- previous global jobs used `epochs = 2`, `batch-size = 16`, `node-batch-size = 500`, `input-seq-len = 150`, `output-seq-len = 50`, and `seed-time-bucket-target-ratios = 0.45,0.30,0.25`

## Loss Stack

The training objective combines congestion-level regression/classification penalties with a newer auxiliary shape-following term over the forecast horizon.

Shape-loss controls include:

- `--loss-shape-lambda`
- `--loss-shape-min-std`
- `--loss-shape-eps`
- `--loss-shape-weight-std`
- `--loss-shape-weight-range`
- `--loss-shape-weight-max`

The loss stack also supports optional operational calibration terms. They are disabled or neutral by default, so existing runs are backward compatible:

- `--loss-horizon-short-steps`, `--loss-horizon-mid-steps`
- `--loss-horizon-short-weight`, `--loss-horizon-mid-weight`, `--loss-horizon-long-weight`
- `--loss-zero-overprediction-lambda`, `--loss-zero-overprediction-threshold`
- `--loss-high-underprediction-lambda`, `--loss-high-underprediction-threshold`

These terms were added after dense real-time evaluation showed two deployment-specific issues: overprediction on true level-0 road-time pairs in early horizons, and underprediction of levels 3 and 4. The recommended first experiment keeps the existing loss terms and adds short-horizon emphasis, a target-zero false-positive penalty, and a high-level underprediction penalty.

For regional runs, the training entrypoint builds a node mask from `node_region_map.csv` and applies it to the loss and operational metrics. The model still sees the full seed-local graph context; only the training objective and metrics are region-scoped.

## Known Training Gaps

- Shape-loss metrics are not yet surfaced explicitly enough in SageMaker logs.
- The new seed-time planner and sampler need targeted unit tests.
- Checkpoints need durable model identity metadata for downstream inference traceability.
