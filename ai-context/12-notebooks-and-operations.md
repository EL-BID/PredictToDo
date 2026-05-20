# Notebooks And Operations

## Main Notebooks

### `notebooks/sagemaker_training_job.ipynb`
- Primary training launcher.
- Important current properties:
  - model name included in job naming
  - output path separated by model type
  - supports current training image / VPC / EFS flow
  - passes seed-time artifact parameters
  - passes the current shape-loss parameters

### `notebooks/sagemaker_rebalance_probe.ipynb`
- Time-only rebalance probe launcher.
- Useful for understanding earlier global `time_idx` logic.

### `notebooks/sagemaker_seed_time_artifact_builder.ipynb`
- Smaller-scale artifact builder launcher.

### `notebooks/sagemaker_seed_time_sharded_launcher.ipynb`
- Large-scale shard launcher for seed-time artifact generation.

## Main Operational Pattern

### Training
- upload source bundle to S3
- create SageMaker training job through notebook or script
- mount EFS channels
- read or download seed-time artifact
- train and log metrics

### Artifact generation
- shard seeds
- run 24 shard jobs
- checkpoint partial outputs
- merge final outputs into one parquet

## Current SageMaker Environment Assumptions
- region:
  - `us-east-1`
- image:
  - `195419001736.dkr.ecr.us-east-1.amazonaws.com/congestion-bogota-sagemaker:temporal-features-20260331-194332`
- role:
  - `arn:aws:iam::195419001736:role/SagemakerExecutionRole`
- EFS:
  - `fs-099e29197c0a3e353`
  - `/data/datos_procesados`

## Current Model Comparison Line
- `GConvLSTM`
- `GConvGRU`
- `TGCN`

## Latest Completed Training Jobs
- `congestion-bogota-gconvlstm-20260420-142912`
- `congestion-bogota-gconvgru-20260420-142911`
- `congestion-bogota-tgcn-20260420-143023`

## SageMaker Launch Configuration To Preserve
- Region: `us-east-1`
- Role: `arn:aws:iam::195419001736:role/SagemakerExecutionRole`
- Bucket: `sagemaker-congestion-bogota`
- Image: `195419001736.dkr.ecr.us-east-1.amazonaws.com/congestion-bogota-sagemaker:temporal-features-20260331-194332`
- EFS id: `fs-099e29197c0a3e353`
- EFS directory: `/data/datos_procesados`
- Subnet: `subnet-017c79447d5151efe`
- Security groups: `sg-0962698c0c8352ddd`, `sg-03590ecb70a82d570`
- Seed-time artifact: `s3://sagemaker-congestion-bogota/congestion-bogota/seed-time-artifacts/final_all/seed_time_bucket_rows_merged_all.parquet`
- Previous global shape-loss line used `epochs = 2`, `batch-size = 16`, `node-batch-size = 500`, `input-seq-len = 150`, `output-seq-len = 50`, and `seed-time-bucket-target-ratios = 0.45,0.30,0.25`.
- Previous successful global instances were `GConvGRU` on `ml.g5.4xlarge`, `GConvLSTM` on `ml.g4dn.8xlarge`, and `TGCN` on `ml.g5.8xlarge`.
- Do not create a separate launch path unless explicitly requested. Use the existing SageMaker notebook/configuration path and update hyperparameters there.

## Current Shape-Loss Hyperparameters In The Main Notebook
- `loss-shape-lambda = 0.05`
- `loss-shape-min-std = 1e-3`
- `loss-shape-eps = 1e-6`
- `loss-shape-weight-std = 0.75`
- `loss-shape-weight-range = 0.25`
- `loss-shape-weight-max = 1.0`

## Current Operational Loss Hyperparameters In The Main Notebook
- `loss-horizon-short-steps = 16`
- `loss-horizon-mid-steps = 32`
- `loss-horizon-short-weight = 1.5`
- `loss-horizon-mid-weight = 1.0`
- `loss-horizon-long-weight = 0.6`
- `loss-zero-overprediction-lambda = 0.4`
- `loss-zero-overprediction-threshold = 0.5`
- `loss-high-underprediction-lambda = 0.6`
- `loss-high-underprediction-threshold = 3`

These knobs were added after the fresh dense real-time analysis showed early-horizon overprediction on true zeros and long-horizon underprediction of levels `3` and `4`. They are additive to the existing composite loss and shape loss; they do not remove any previous loss components.

## Important Operational Lesson Already Learned
- SageMaker metric definitions have a hard limit, so metric lists must be curated.
- The repo now uses a reduced set for job launch while still logging richer details internally.
- The shape-loss experiment line currently lacks explicit metric surfacing for the shape term itself.
