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

## Current Shape-Loss Hyperparameters In The Main Notebook
- `loss-shape-lambda = 0.05`
- `loss-shape-min-std = 1e-3`
- `loss-shape-eps = 1e-6`
- `loss-shape-weight-std = 0.75`
- `loss-shape-weight-range = 0.25`
- `loss-shape-weight-max = 1.0`

## Important Operational Lesson Already Learned
- SageMaker metric definitions have a hard limit, so metric lists must be curated.
- The repo now uses a reduced set for job launch while still logging richer details internally.
- The shape-loss experiment line currently lacks explicit metric surfacing for the shape term itself.
