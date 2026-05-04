# Artifacts And Locations

## Main Training Artifact

### Local combined parquet
- `data/artifacts/seed_time_merged_all/seed_time_bucket_rows_merged_all.parquet`

### Local combined metadata
- `data/artifacts/seed_time_merged_all/seed_time_bucket_metadata_merged_all.json`

### S3 parquet used by training
- `s3://sagemaker-congestion-bogota/congestion-bogota/seed-time-artifacts/final_all/seed_time_bucket_rows_merged_all.parquet`

### S3 metadata
- `s3://sagemaker-congestion-bogota/congestion-bogota/seed-time-artifacts/final_all/seed_time_bucket_metadata_merged_all.json`

## Earlier Merged Artifacts

### Base 937-seed artifact
- local parquet:
  - `data/artifacts/seed_time_merged/seed_time_bucket_rows_merged.parquet`
- metadata:
  - `data/artifacts/seed_time_merged/seed_time_bucket_metadata_merged.json`

### Extra-1000 artifact
- local parquet:
  - `data/artifacts/seed_time_merged_extra1000/seed_time_bucket_rows_merged.parquet`
- metadata:
  - `data/artifacts/seed_time_merged_extra1000/seed_time_bucket_metadata_merged.json`

## Seed Strategy Artifacts

### First seed universe
- `data/artifacts/seed_strategy/seed_strategy.csv`
- `data/artifacts/seed_strategy/seed_strategy_metadata.json`
- `data/artifacts/seed_strategy/shards_24/seed_shard_manifest.json`

### Expanded seed universe
- `data/artifacts/seed_strategy_extra1000/seed_strategy_extra1000.csv`
- `data/artifacts/seed_strategy_extra1000/seed_strategy_extra1000_metadata.json`
- `data/artifacts/seed_strategy_extra1000/shards_24/seed_shard_manifest.json`

## Time Index Artifact
- `data/artifacts/time_index_strategy/time_idx_strategy_2000.csv`
- `data/artifacts/time_index_strategy/time_idx_strategy_2000_metadata.json`

## Shard Output Prefixes

### First checkpointed shard run
- `s3://sagemaker-congestion-bogota/congestion-bogota/seed-time-artifact-checkpointed/outputs/`

### Extra-1000 checkpointed shard run
- `s3://sagemaker-congestion-bogota/congestion-bogota/seed-time-artifact-extra1000-checkpointed/outputs/`

## Current Training Output Prefixes

The training notebook and manual launches now separate outputs by model:
- `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/a3tgcn`
- `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/gconvlstm`
- `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/gconvgru`
- `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/tgcn`

## Latest Downloaded Model Artifacts

### Local extracted model folders
- `data/artifacts/model_gconvgru_2026_04_20`
- `data/artifacts/model_gconvlstm_2026_04_20`
- `data/artifacts/model_tgcn_2026_04_20`

### Local tarballs
- `data/artifacts/model_gconvgru_2026_04_20.tar.gz`
- `data/artifacts/model_gconvlstm_2026_04_20.tar.gz`
- `data/artifacts/model_tgcn_2026_04_20.tar.gz`

### Shared S3 tarballs
- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/model_gconvgru_2026_04_20.tar.gz`
- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/model_gconvlstm_2026_04_20.tar.gz`
- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/model_tgcn_2026_04_20.tar.gz`

### Upload manifests
- `data/artifacts/uploaded_model_artifacts_shared_2026_04_23.json`
- each extracted model directory also contains:
  - `artifact_source.json`

## Real-Time Evaluation Artifacts

### Last-7-day manifest and local downloads
- `data/artifacts/realtime_last7d_eval/manifest.json`
- predictions:
  - `data/artifacts/realtime_last7d_eval/downloads_pred/`
- jams:
  - `data/artifacts/realtime_last7d_eval/downloads_jam/`

### Dense evaluation outputs
- `data/artifacts/realtime_last7d_eval/dense_eval/summary.json`
- `data/artifacts/realtime_last7d_eval/dense_eval/metrics_by_prediction_file.csv`
- `data/artifacts/realtime_last7d_eval/dense_eval/metrics_by_level_per_file.csv`
- `data/artifacts/realtime_last7d_eval/dense_eval/metrics_by_step_per_file.csv`

### Snapshot evaluation workdir
- `data/artifacts/realtime_last7d_eval/dense_eval_snapshot/`
- `data/artifacts/realtime_last7d_eval/current_pred_snapshot.txt`

## EFS Paths
- SageMaker training data mounts from:
  - filesystem id: `fs-099e29197c0a3e353`
  - directory: `/data/datos_procesados`

## Important Runtime Note
- The seed-time artifact does not need to pre-exist in EFS.
- `training_entrypoint.py` can download it from S3, cache locally, and optionally persist a copy under:
  - `/data/datos_procesados/seed_time_artifacts/`
