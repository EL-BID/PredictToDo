# Current Workspace State

Last refreshed: 2026-05-04

## Bottom-Line Summary
- The repo has a working seed-time artifact pipeline and planner-based sampler.
- A second training line added an auxiliary node-wise Pearson shape loss over the full forecast horizon.
- The latest completed SageMaker comparison line is now the `2026-04-20` shape-loss run for:
  - `GConvLSTM`
  - `GConvGRU`
  - `TGCN`
- Those three latest artifacts were downloaded locally and uploaded to the shared artifacts bucket.
- Real-time evaluation work is now active against dense `prediction + jams fillna(0)` surfaces from the sibling inference system.

## Integrated Code Changes That Matter

### `src/training_utils.py`
- Added:
  - `SeedTimeEpochPlan`
  - `SeedTimeEpochPlanner`
  - `PlannedSeedTimeBatchSampler`
- Planner groups artifact rows by seed and bucket and builds epoch plans.

### `src/training_entrypoint.py`
- Added support for:
  - `--seed-time-artifact-path`
  - `--seed-time-epoch-cycles`
  - `--seed-time-bucket-target-ratios`
  - `--seed-time-shuffle-seeds`
- Added S3 download/cache logic for the artifact.
- Uses planner path when a seed-time artifact is provided.
- Added support for shape-following loss controls:
  - `--loss-shape-lambda`
  - `--loss-shape-min-std`
  - `--loss-shape-eps`
  - `--loss-shape-weight-std`
  - `--loss-shape-weight-range`
  - `--loss-shape-weight-max`

### `notebooks/sagemaker_training_job.ipynb`
- Updated to:
  - include model name in `job_name`
  - separate `S3OutputPath` by model family
  - pass seed-time artifact parameters
  - pass shape-loss parameters
  - use the current stable SageMaker image / VPC / EFS recipe

## Current Artifact Status
- base merged artifact:
  - `937` seeds
  - `1,874,000` rows
- extra merged artifact:
  - `1000` seeds
  - `2,000,000` rows
- final combined artifact:
  - `1937` seeds
  - `3,874,000` rows

## Latest Completed Shape-Loss Training Jobs

### `GConvLSTM`
- job:
  - `congestion-bogota-gconvlstm-20260420-142912`
- instance:
  - `ml.g4dn.8xlarge`
- output:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/gconvlstm`

### `GConvGRU`
- job:
  - `congestion-bogota-gconvgru-20260420-142911`
- instance:
  - `ml.g5.4xlarge`
- output:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/gconvgru`

### `TGCN`
- job:
  - `congestion-bogota-tgcn-20260420-143023`
- instance:
  - `ml.g5.8xlarge`
- output:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/tgcn`

## Latest Comparative Validation Reading

### Best global model
- `GConvLSTM`
- latest validation summary:
  - `val_accuracy_exact = 0.6538`
  - `val_accuracy_pm1 = 0.8357`
  - `val_mae_level = 0.6301`

### Best high-congestion model
- `GConvGRU`
- remains strongest on levels `3` and `4` among the latest 3-model line

### Most improved vs previous own version
- `TGCN`
- improved global validation metrics materially versus its `2026-04-10` line
- but that improvement is driven more by lower levels than by severe congestion

## Real-Time Evaluation Status
- Dense evaluation now uses the full prediction surface from S3 backups plus jams with missing values interpreted as `0`.
- The current partial stable summary from `10` prediction files is:
  - `mae = 1.7331`
  - `rmse = 1.9473`
  - `bias = 1.6535`
  - `acc_exact = 0.1897`
  - `acc_pm1 = 0.3351`
- The important pattern is:
  - low levels are still overpredicted
  - levels `3` and `4` are still underpredicted
  - but the model shows real signal in levels `2` and `3` under honest dense evaluation
