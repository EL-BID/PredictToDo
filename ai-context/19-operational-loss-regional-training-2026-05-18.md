# Operational Loss Regional Training 2026-05-18

This note records the SageMaker regional training jobs launched after the fresh dense real-time evaluation showed early-horizon overprediction on true zeros and high-level underprediction.

## Jobs Launched

Six regional jobs were launched, preserving the previous regional split between central/principal corridors and noncentral/non-principal roads. The first two launched jobs were `GConvGRU` because the previous regional validation line identified it as the useful current two-model candidate for both groups; the remaining `GConvLSTM` and `TGCN` jobs were then launched to complete the full regional/model matrix.

Update on 2026-05-18 21:26 Europe/Madrid: the first launch set (`oploss-...-180835`) failed before training because the synced paper branch had dropped the regional entrypoint arguments `--training-region` and `--node-region-map-path`. The regional masking logic was restored from the previous successful regional source package and combined with the new operational-loss terms. The pending old `noncentral / GConvGRU` job was stopped before it started, and the six jobs were relaunched as `oploss2-...-192556` with source package:

`s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/code/20260518-192556/sourcedir.tar.gz`

| Region | Job name | Base job | Instance | Status at launch |
|---|---|---|---|---|
| central / principales | `congestion-bogota-oploss-central-gconvgru-20260518-180835` | `congestion-bogota-cen-gconvgru-20260514230035` | `ml.g5.8xlarge` | `InProgress / Pending` |
| central / principales | `congestion-bogota-oploss-central-gconvlstm-20260518-180835` | `congestion-bogota-cen-gconvlstm-20260514230035` | `ml.g4dn.8xlarge` | `InProgress / Starting` |
| central / principales | `congestion-bogota-oploss-central-tgcn-20260518-180835` | `congestion-bogota-cen-tgcn-20260514230035` | `ml.g5.4xlarge` | `InProgress / Pending` |
| noncentral / no principales | `congestion-bogota-oploss-noncentral-gconvgru-20260518-180835` | `congestion-bogota-non-gconvgru-g512xl-20260514230153` | `ml.g5.12xlarge` | `InProgress / Pending` |
| noncentral / no principales | `congestion-bogota-oploss-noncentral-gconvlstm-20260518-180835` | `congestion-bogota-non-gconvlstm-g4dn12xl-20260514230149` | `ml.g4dn.12xlarge` | `InProgress / Pending` |
| noncentral / no principales | `congestion-bogota-oploss-noncentral-tgcn-20260518-180835` | `congestion-bogota-non-tgcn-g4dn16xl-20260514230154` | `ml.g4dn.16xlarge` | `InProgress / Pending` |

Corrected relaunch names:

| Region | Corrected job name | Instance | Status at relaunch check |
|---|---|---|---|
| central / principales | `congestion-bogota-oploss2-central-gconvgru-20260518-192556` | `ml.g5.8xlarge` | `InProgress / Pending` |
| central / principales | `congestion-bogota-oploss2-central-gconvlstm-20260518-192556` | `ml.g4dn.8xlarge` | `InProgress / Downloading` |
| central / principales | `congestion-bogota-oploss2-central-tgcn-20260518-192556` | `ml.g5.4xlarge` | `InProgress / Pending` |
| noncentral / no principales | `congestion-bogota-oploss2-noncentral-gconvgru-20260518-192556` | `ml.g5.12xlarge` | `InProgress / Pending` |
| noncentral / no principales | `congestion-bogota-oploss2-noncentral-gconvlstm-20260518-192556` | `ml.g4dn.12xlarge` | `InProgress / Pending` |
| noncentral / no principales | `congestion-bogota-oploss2-noncentral-tgcn-20260518-192556` | `ml.g4dn.16xlarge` | `InProgress / Pending` |

Note: the base job for `central / TGCN`, `congestion-bogota-cen-tgcn-20260514230035`, had failed in the previous regional line. The new `central / TGCN` operational-loss job was still launched to complete the requested six-job matrix, but it should be watched closely.

Shared uploaded source package:

`s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/code/20260518-180835/sourcedir.tar.gz`

Corrected source package after regional-entrypoint restoration:

`s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/code/20260518-192556/sourcedir.tar.gz`

## Preserved Configuration

The launch reused the successful regional configuration from `2026-05-15`:

- region: `us-east-1`
- role: `arn:aws:iam::195419001736:role/SagemakerExecutionRole`
- image: `195419001736.dkr.ecr.us-east-1.amazonaws.com/congestion-bogota-sagemaker:temporal-features-20260331-194332`
- EFS id: `fs-099e29197c0a3e353`
- EFS directory: `/data/datos_procesados`
- subnet: `subnet-017c79447d5151efe`
- security groups: `sg-0962698c0c8352ddd`, `sg-03590ecb70a82d570`
- epochs: `2`
- batch size: `16`
- input horizon: `150`
- output horizon: `50`
- node batch size: `500`
- seed-time bucket ratios: `0.30,0.30,0.40`

Regional artifacts:

- central seed-time artifact: `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/artifacts/20260514-230035/seed_time_bucket_rows_central.parquet`
- noncentral seed-time artifact: `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/artifacts/20260514-230035/seed_time_bucket_rows_noncentral.parquet`
- node-region map: `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/artifacts/20260514-230035/node_region_map.csv`

## New Operational Loss Terms

The jobs keep the existing composite loss and shape loss, and add the new operational terms:

| Hyperparameter | Value |
|---|---:|
| `loss-horizon-short-steps` | `16` |
| `loss-horizon-mid-steps` | `32` |
| `loss-horizon-short-weight` | `1.5` |
| `loss-horizon-mid-weight` | `1.0` |
| `loss-horizon-long-weight` | `0.6` |
| `loss-zero-overprediction-lambda` | `0.4` |
| `loss-zero-overprediction-threshold` | `0.5` |
| `loss-high-underprediction-lambda` | `0.6` |
| `loss-high-underprediction-threshold` | `3` |

These terms target the two main dense-evaluation problems:

- true level `0` is strongly overpredicted in the first 12--16 forecast steps;
- true levels `3` and `4` are underpredicted, especially as the horizon grows.

## How To Check Status

```bash
aws sagemaker describe-training-job \
  --region us-east-1 \
  --training-job-name congestion-bogota-oploss-central-gconvgru-20260518-180835

aws sagemaker describe-training-job \
  --region us-east-1 \
  --training-job-name congestion-bogota-oploss-noncentral-gconvgru-20260518-180835

aws sagemaker list-training-jobs \
  --region us-east-1 \
  --name-contains congestion-bogota-oploss2 \
  --sort-by CreationTime \
  --sort-order Descending
```

When they finish, compare the final validation metrics against:

- central base: `val_accuracy_exact = 0.589541`, `val_accuracy_pm1 = 0.759458`, `val_mae_level = 0.803451`;
- noncentral base: `val_accuracy_exact = 0.694478`, `val_accuracy_pm1 = 0.848492`, `val_mae_level = 0.579191`.

The most important downstream check is not only validation accuracy. The new models should be pushed through dense real-time evaluation with zeros included, summarized globally, by forecast step, and by true level. The expected improvement is lower early-horizon false positives for true level `0` without losing too much performance on levels `2` and `3`.
