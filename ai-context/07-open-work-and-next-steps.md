# Open Work And Next Steps

## Highest-Priority Runtime Questions

### 1. Do we keep the shape loss experiment line?
- Compare the `2026-04-20` jobs against the `2026-04-10` baseline per model.
- Current reading:
  - `GConvLSTM` stayed strongest globally but regressed slightly
  - `GConvGRU` improved some global softness metrics but softened a bit in high-severity exactness
  - `TGCN` improved most clearly overall, mostly on lower levels

### 2. Do we expose explicit shape metrics during training?
- The Pearson-based shape term is integrated into loss, but current SageMaker runs do not surface:
  - `train_shape_loss`
  - `val_shape_loss`
  - `train_shape_corr_mean`
  - `val_shape_corr_mean`
- This should be fixed before the next serious experiment line.

### 3. Does the new sampler improve non-trivial class behavior?
- Compare against earlier baseline jobs on:
  - `val_accuracy_exact`
  - `val_class_distance_mean`
  - `val_level_3_*`
  - `val_level_4_*`
  - class counts if available
- The real goal is not just lower loss.
- The real goal is better behavior on informative congestion cases without breaking level `0`.

### 4. What is the correct deployment and calibration path?
- Real-time evaluation now shows:
  - overprediction at low levels
  - underprediction at high levels
- The next practical decision is whether to:
  - recalibrate the current best model line
  - retrain again with further loss/sampling changes
  - or both

## Likely Follow-Up Engineering Work

### Better runtime observability
- Add explicit logs for:
  - number of seeds in the epoch plan
  - number of batches
  - expected samples per epoch
  - realized per-epoch bucket counts
  - explicit shape-loss metrics

### Stronger per-class monitoring
- Metric definition limit forced a reduced metric set.
- If deeper per-class visibility is needed:
  - write a compact JSON artifact at epoch end
  - or a metrics summary file in `/opt/ml/model/`

### Inference traceability fixes
- Prediction parquet backups in the sibling repo currently do not persist model identity.
- The service does build `pred_meta`, but only Redis gets it.
- S3 backup should also persist:
  - `model_version`
  - `model_md5`
  - `training_job_name`
  - `model_artifact_s3`
  - `inference_code_version`

### Checkpoint identity fixes
- Training checkpoints currently do not save:
  - `version`
  - `run_id`
  - `training_job_name`
  - `artifact source`
- Because of that, the sibling inference repo often resolves `model_version = unknown`.

### Real-time evaluation completion
- Finish the last-7-day dense evaluation over the larger downloaded set, not only the 10-file partial snapshot.
- Then produce:
  - global metrics
  - metrics by real level
  - metrics by forecast step
  - false-positive analysis for `actual=0`

## Practical Commands To Re-Run Or Inspect

### Inspect current training jobs
```bash
aws sagemaker describe-training-job --region us-east-1 --training-job-name <job_name>
```

### List latest jobs by model
```bash
aws sagemaker list-training-jobs --region us-east-1 --name-contains congestion-bogota-gconvgru --sort-by CreationTime --sort-order Descending
```

### Canonical artifact used by training
```text
s3://sagemaker-congestion-bogota/congestion-bogota/seed-time-artifacts/final_all/seed_time_bucket_rows_merged_all.parquet
```

## Recommended Next AI-Agent Tasks
1. add explicit shape-loss metrics to logs and SageMaker metric definitions
2. finish the larger dense real-time evaluation run
3. patch the sibling inference repo to persist model identity with parquet outputs
4. patch training checkpoints so inference can read a real model version instead of `unknown`
