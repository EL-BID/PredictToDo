# Regional Corridor Status

Generated: 2026-05-18

This file captures the current state after reviewing `ai-context`, local artifacts, SageMaker job status, and local testability. It is the current handoff for agents continuing the central/noncentral regional model line.

## Bottom Line

The project has moved beyond a single global model experiment line. The regional split is implemented and has produced usable SageMaker model artifacts for two operational regions:

- `central`: prioritized corridor nodes, equivalent to the main corridor or principal street model line.
- `noncentral`: all remaining graph nodes, equivalent to the secondary/non-prioritized street model line.

As of 2026-05-18, no `congestion-bogota` SageMaker training jobs are in progress. The latest useful regional training jobs completed on 2026-05-15.

The best available two-model candidate is:

| Region | Recommended model | Job | Validation MAE | Exact accuracy | +/-1 accuracy |
|---|---|---|---:|---:|---:|
| central | `GConvGRU` | `congestion-bogota-cen-gconvgru-20260514230035` | `0.803451` | `0.589541` | `0.759458` |
| noncentral | `GConvGRU` | `congestion-bogota-non-gconvgru-g512xl-20260514230153` | `0.579191` | `0.694478` | `0.848492` |

Use those numbers only as regional validation metrics from the SageMaker training loop. They are not yet a final deployed real-time two-model system metric.

## Regional Split Artifacts

The region map was built from:

- `artifacts/21 CORREDORES PRIORIZADOS.lpkx`
- extracted shapefile: `data/artifacts/corridors_prioritized_lpkx/commondata/corredores_principales/PLMB_V2.shp`
- graph: `data/datos_procesados/conectivity_graph/road_graph.pickle`

Region map outputs:

- `data/artifacts/corridor_regions/node_region_map.csv`
- `data/artifacts/corridor_regions/central_nodes.csv`
- `data/artifacts/corridor_regions/noncentral_nodes.csv`
- `data/artifacts/corridor_regions/node_region_map_metadata.json`

Verified counts:

| Item | Count |
|---|---:|
| graph nodes | `139107` |
| central nodes | `17308` |
| noncentral nodes | `121799` |
| prioritized corridors | `21` |

Largest corridor node counts:

| Corridor | Nodes |
|---|---:|
| `Av Boyaca` | `2496` |
| `Av. Caracas` | `1833` |
| `Av Ciudad de Cali` | `1450` |
| `Av Carrera 7` | `1002` |
| `Av Carrera 68` | `943` |
| `NQS` | `926` |
| `Av Calle 72 y Calle 68` | `913` |
| `Autopista Norte` | `881` |
| `Av Calle 80` | `836` |
| `Av Centenario` | `822` |

## Regional Seed Strategy

Seed strategy outputs:

- `data/artifacts/region_seed_strategy/seed_strategy_central.csv`
- `data/artifacts/region_seed_strategy/seed_strategy_noncentral.csv`
- `data/artifacts/region_seed_strategy/region_seed_strategy_manifest.json`

Strategy counts:

| Region | Region nodes | Coverage seeds | Top-congestion extra seeds | Total desired seeds |
|---|---:|---:|---:|---:|
| central | `17308` | `133` | `3271` | `3404` |
| noncentral | `121799` | `604` | `2800` | `3404` |

Important: both regions target `3404` seeds, but the current training artifacts used for the latest successful jobs were partial regional artifacts, not the full 3404-seed artifacts.

## Current Regional Seed-Time Artifacts

Current useful local regional artifact directories:

- `data/artifacts/region_seed_time_final_partial/`
- `data/artifacts/region_seed_time_fast_partial/`

The two directories currently contain equivalent-looking partial outputs. The relevant files are:

- `data/artifacts/region_seed_time_final_partial/seed_time_bucket_rows_central.parquet`
- `data/artifacts/region_seed_time_final_partial/seed_time_bucket_metadata_central.json`
- `data/artifacts/region_seed_time_final_partial/seed_time_bucket_rows_noncentral.parquet`
- `data/artifacts/region_seed_time_final_partial/seed_time_bucket_metadata_noncentral.json`
- `data/artifacts/region_seed_time_final_partial/final_region_seed_time_manifest.json`

Partial artifact counts:

| Region | Rows | Seeds | Time indices | Score mean | Score max | Bucket thresholds |
|---|---:|---:|---:|---:|---:|---|
| central | `6260000` | `3130` | `2000` | `0.020457` | `0.30000` | `[0.01592, 0.02756]` |
| noncentral | `5110000` | `2555` | `2000` | `0.016177` | `0.15352` | `[0.01132, 0.02176]` |

Bucket counts:

| Region | High | Medium | Low |
|---|---:|---:|---:|
| central | `1566655` | `1559613` | `3133732` |
| noncentral | `1277580` | `1274324` | `2558096` |

Coverage limitation:

- central is missing `274` desired seeds versus the `3404` target.
- noncentral is missing `849` desired seeds versus the `3404` target.
- These artifacts were good enough to launch and complete the latest regional training jobs, but should not be described as complete final regional seed-time artifacts.

## SageMaker Job Status

Checked from AWS SageMaker in `us-east-1` on 2026-05-18.

There are no `congestion-bogota` SageMaker jobs currently `InProgress`.

Latest completed regional jobs:

| Region | Model | Job | Status | Instance | Validation loss | Validation MAE | Exact accuracy | +/-1 accuracy | Distance ge2 rate |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| central | `GConvGRU` | `congestion-bogota-cen-gconvgru-20260514230035` | `Completed` | `ml.g5.8xlarge` | `5.617597` | `0.803451` | `0.589541` | `0.759458` | `0.240542` |
| central | `GConvLSTM` | `congestion-bogota-cen-gconvlstm-20260514230035` | `Completed` | `ml.g4dn.8xlarge` | `5.755886` | `0.872182` | `0.556664` | `0.728042` | `0.271958` |
| central | `TGCN` | `congestion-bogota-cen-tgcn-20260514230035` | `Failed` | `ml.g5.4xlarge` | `5.573224` | `1.035137` | `0.432726` | `0.678476` | `0.321524` |
| noncentral | `GConvGRU` | `congestion-bogota-non-gconvgru-g512xl-20260514230153` | `Completed` | `ml.g5.12xlarge` | `5.440405` | `0.579191` | `0.694478` | `0.848492` | `0.151508` |
| noncentral | `GConvLSTM` | `congestion-bogota-non-gconvlstm-g4dn12xl-20260514230149` | `Completed` | `ml.g4dn.12xlarge` | `5.497463` | `0.587470` | `0.690261` | `0.833881` | `0.166119` |
| noncentral | `TGCN` | `congestion-bogota-non-tgcn-g4dn16xl-20260514230154` | `Completed` | `ml.g4dn.16xlarge` | `5.512827` | `0.613939` | `0.651643` | `0.845691` | `0.154309` |

The latest completed jobs used:

- `epochs = 2`
- `node-batch-size = 500`
- `seed-time-bucket-target-ratios = 0.30,0.30,0.40`
- `loss-shape-lambda = 0.05`
- regional masking via `--training-region` and `--node-region-map-path`

Model artifact S3 URIs:

- central `GConvGRU`: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/central/gconvgru/congestion-bogota-cen-gconvgru-20260514230035/output/model.tar.gz`
- central `GConvLSTM`: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/central/gconvlstm/congestion-bogota-cen-gconvlstm-20260514230035/output/model.tar.gz`
- noncentral `GConvGRU`: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/noncentral/gconvgru/congestion-bogota-non-gconvgru-g512xl-20260514230153/output/model.tar.gz`
- noncentral `GConvLSTM`: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/noncentral/gconvlstm/congestion-bogota-non-gconvlstm-g4dn12xl-20260514230149/output/model.tar.gz`
- noncentral `TGCN`: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/noncentral/tgcn/congestion-bogota-non-tgcn-g4dn16xl-20260514230154/output/model.tar.gz`

## Published Final Candidate Artifacts

Updated on 2026-05-18 after downloading the two winning regional `GConvGRU` SageMaker artifacts locally and uploading them to the shared artifacts bucket.

Local artifact outputs:

| Region | Local tarball | Local extracted directory | SHA-256 |
|---|---|---|---|
| central | `data/artifacts/model_gconvgru_regional_central_2026_05_15.tar.gz` | `data/artifacts/model_gconvgru_regional_central_2026_05_15/` | `1c2fd23618923d58f5b28b2883a2d660c9da213343592306633b0c63b7393367` |
| noncentral | `data/artifacts/model_gconvgru_regional_noncentral_2026_05_15.tar.gz` | `data/artifacts/model_gconvgru_regional_noncentral_2026_05_15/` | `ff9f93ef7e759a6e87c6a4d0550a771490cccf8f990e6741fcf05c1bc3bfbf19` |

Shared S3 artifact outputs:

| Region | Shared S3 tarball |
|---|---|
| central | `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/model_gconvgru_regional_central_2026_05_15.tar.gz` |
| noncentral | `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/model_gconvgru_regional_noncentral_2026_05_15.tar.gz` |

Shared manifest:

- local: `data/artifacts/regional_gconvgru_final_models_2026_05_18.json`
- S3: `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/regional_gconvgru_final_models_2026_05_18.json`

The extracted local directories each contain:

- `model.pth`
- `model_config.pkl`
- `metrics.pkl`
- `training_args.pkl`
- `feature_order.json`
- `data_meta.pkl`
- `artifact_source.json`

S3 object tags were applied:

| Object | Tags |
|---|---|
| central tarball | `project=congestion-bogota`, `model_type=GConvGRU`, `region=central`, `role=regional-final-candidate`, `training_date=2026-05-15` |
| noncentral tarball | `project=congestion-bogota`, `model_type=GConvGRU`, `region=noncentral`, `role=regional-final-candidate`, `training_date=2026-05-15` |
| manifest | `project=congestion-bogota`, `model_type=GConvGRU`, `region=central-noncentral`, `role=regional-final-manifest`, `created_date=2026-05-18` |

## Failed Or Stopped Attempts

Many earlier regional job attempts were stopped manually or failed. Do not treat their S3 artifact paths as usable model artifacts unless verified.

Important failure pattern:

- The 2026-05-14 22:18/22:21 attempts failed with:
  - `RuntimeError: The size of tensor a (400000) must match the size of tensor b (500) at non-singleton dimension 0`
- This points to a regional mask tensor-shape bug.
- The later 2026-05-14 23:00/23:01 jobs completed for `GConvLSTM` and `GConvGRU`, so the effective launched code for that batch had moved past that specific failure.

Central `TGCN` in the latest useful batch failed with:

- `ClientError: Please use an instance type with more memory, or reduce the size of job data processed on an instance.`

## Real-Time Evaluation Status

The existing dense real-time evaluation artifact is not yet a two-regional-model evaluation. It evaluates the currently available prediction parquet surface from the sibling inference system.

Existing dense evaluation:

- directory: `data/artifacts/realtime_last7d_eval/dense_eval/`
- summary: `data/artifacts/realtime_last7d_eval/dense_eval/summary.json`
- prediction files evaluated: `10`
- jams files loaded: `487`
- rows evaluated: `69553500`

Current dense raw summary:

| Metric | Value |
|---|---:|
| MAE | `1.733092` |
| RMSE | `1.947313` |
| Bias | `1.653519` |
| Exact accuracy | `0.189698` |
| +/-1 accuracy | `0.335142` |

Pattern:

- low observed levels are overpredicted.
- observed levels `3` and `4` are underpredicted.
- the model shows some useful signal for levels `2` and `3`, but calibration is still required.

Calibration artifacts:

- `data/artifacts/calibration_realtime/`
- `data/artifacts/calibration_realtime_repr/`

Best existing calibration model is `qmap_global`.

Large representative calibration summary:

| Model | Rows | MAE | Exact accuracy | +/-1 accuracy |
|---|---:|---:|---:|---:|
| raw | `9086287` | `1.794080` | `0.108446` | `0.512466` |
| qmap_global | `9086287` | `0.805915` | `0.358689` | `0.841514` |
| hybrid_0_6 | `9086287` | `1.452679` | `0.206977` | `0.647175` |
| hybrid_0_8 | `9086287` | `1.640970` | `0.151246` | `0.574981` |

Do not present the `qmap_global` numbers as regional model deployment metrics. They are a calibration result over existing real-time outputs.

## Local Test Status

Attempted local test commands:

```bash
pytest -q
```

Result:

- failed because `pytest` is not installed in the default shell.

Then attempted:

```bash
/home/jose_marquez/miniconda3/envs/traffic-gnn-py310/bin/python -m pytest -q
```

Result:

- test collection failed when importing `torch`:
  - `ImportError: ... libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`

Also attempted:

```bash
LD_PRELOAD=/home/jose_marquez/miniconda3/envs/traffic-gnn-py310/lib/libiomp5.so /home/jose_marquez/miniconda3/envs/traffic-gnn-py310/bin/python -m pytest -q
```

Result:

- same `torch` import failure.

Preprocessing tests that do not import `torch` pass:

```bash
/home/jose_marquez/miniconda3/envs/traffic-gnn-py310/bin/python -m pytest -q tests/preprocessing
```

Result:

- `22 passed in 0.80s`

Implication:

- The local environment cannot currently validate Torch-dependent tests.
- This is an environment/package issue, not evidence that the regional model code is failing.
- Before relying on local test results, fix the PyTorch/Intel runtime mismatch in `traffic-gnn-py310` or use the SageMaker training image/container.

## What To Do Next

Recommended next work:

1. Download and extract the two recommended regional `GConvGRU` artifacts from S3 into clear local directories under `data/artifacts/`.
2. Decide whether to complete missing seed-time shards and rebuild full regional artifacts before any final training claim.
3. Run a longer or final training line for the chosen two-model setup, ideally with complete regional seed-time artifacts.
4. Adapt or verify the sibling inference repository can route nodes through:
   - central model for `is_central = true`
   - noncentral model for `is_central = false`
5. Run dense real-time evaluation on the actual two-model output surface.
6. Apply calibration only after the two-model inference path has produced comparable prediction parquet backups.
7. Fix local Torch test environment or run tests in the SageMaker-compatible image.

If a quick interim number is needed, use the regional validation numbers from the latest completed jobs:

- central `GConvGRU`: exact `58.95%`, +/-1 `75.95%`, MAE `0.8035`
- noncentral `GConvGRU`: exact `69.45%`, +/-1 `84.85%`, MAE `0.5792`

If a real-time calibrated number is needed, clearly label it as existing real-time calibration, not regional deployment:

- `qmap_global`: exact `35.87%`, +/-1 `84.15%`, MAE `0.8059`, rows `9,086,287`
