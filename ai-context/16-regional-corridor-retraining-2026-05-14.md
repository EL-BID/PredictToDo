# Regional Corridor Retraining Handoff

Generated: 2026-05-14 18:20 CEST

Source session:
- `019e256b-e427-7b40-b54a-b09b76a64175`
- Thread: `Plan reentrenamiento por corredores`
- Raw transcript: `/home/jose_marquez/.codex/sessions/2026/05/14/rollout-2026-05-14T09-38-08-019e256b-e427-7b40-b54a-b09b76a64175.jsonl`

## Problem Framing

The working hypothesis is that one global congestion model is under-serving two different regions:
- central prioritized corridors
- noncentral or less-principal corridors

The chosen direction is to build two regional training lines:
- one model focused on central corridor nodes
- one model focused on noncentral nodes

This requires region-specific:
- node maps
- seed strategies
- `(seed_id, time_idx)` severity artifacts
- loss masking or weighting during model training

## Corridor Region Map

Input package:
- `artifacts/21 CORREDORES PRIORIZADOS.lpkx`

Extraction details:
- `.lpkx` is a 7-zip archive
- expected corridor shapefile after extraction:
  - `data/artifacts/corridors_prioritized_lpkx/commondata/corredores_principales/PLMB_V2.shp`
- required corridor column:
  - `CORREDOR`

Implementation:
- `scripts/build_corridor_region_map.py`

Method:
- read `PLMB_V2.shp`
- project to `EPSG:3116`
- buffer corridors by `20m`
- build repo road ids from `MVI.geojson`
- intersect MVI road geometries against corridor buffers
- restrict to graph nodes from `data/datos_procesados/conectivity_graph/road_graph.pickle`

Outputs:
- `data/artifacts/corridor_regions/node_region_map.csv`
- `data/artifacts/corridor_regions/central_nodes.csv`
- `data/artifacts/corridor_regions/noncentral_nodes.csv`
- `data/artifacts/corridor_regions/node_region_map_metadata.json`

Counts:
- graph nodes: `139107`
- central nodes: `17308`
- noncentral nodes: `121799`
- corridor count: `21`

Largest corridor node counts:
- `Av Boyaca`: `2496`
- `Av. Caracas`: `1833`
- `Av Ciudad de Cali`: `1450`
- `Av Carrera 7`: `1002`
- `Av Carrera 68`: `943`
- `NQS`: `926`
- `Av Calle 72 y Calle 68`: `913`
- `Autopista Norte`: `881`
- `Av Calle 80`: `836`
- `Av Centenario`: `822`

## Regional Seed Strategies

Implementation:
- `scripts/build_region_seed_strategies.py`

Inputs:
- region map: `data/artifacts/corridor_regions/node_region_map.csv`
- coverage seeds: `data/datos_procesados/optimal_seeds.pkl`
- congestion ranking: `data/datos_procesados/nodes_traffic_info.csv`

Outputs:
- `data/artifacts/region_seed_strategy/seed_strategy_central.csv`
- `data/artifacts/region_seed_strategy/seed_strategy_noncentral.csv`
- `data/artifacts/region_seed_strategy/region_seed_strategy_manifest.json`

Strategy counts:

| Region | Region Nodes | Coverage Seeds | Top-Congestion Extra | Total Seeds |
|---|---:|---:|---:|---:|
| central | 17308 | 133 | 3271 | 3404 |
| noncentral | 121799 | 604 | 2800 | 3404 |

Design note:
- both regional strategies target `3404` seeds
- coverage seeds are kept when they fall inside the region
- extra seeds come from the highest `level_mean` nodes in that same region

## Existing Artifact Reuse

Implementation:
- `scripts/reuse_existing_seed_time_by_region.py`

Existing artifact reused where possible:
- `data/artifacts/seed_time_merged_all/seed_time_bucket_rows_merged_all.parquet`

Output manifest:
- `data/artifacts/region_seed_time_reuse/reuse_manifest.json`

Reuse/missing counts:

| Region | Desired Seeds | Reused Seeds | Missing Seeds | Reused Rows | Expected Missing Rows |
|---|---:|---:|---:|---:|---:|
| central | 3404 | 662 | 2742 | 1324000 | 5484000 |
| noncentral | 3404 | 1275 | 2129 | 2550000 | 4258000 |

Important:
- existing rows keep `score` and the old `bucket` only as diagnostics
- final regional artifacts must be rebucketed per region after merging reused and newly generated rows

## SageMaker Seed-Time Generation

Implementation:
- `scripts/prepare_region_seed_time_jobs.py`
- SageMaker program: `src/rebalance_sampling.py`

Purpose:
- score missing regional seeds against the selected `time_idx` strategy
- emit shard-level `(seed_id, time_idx, score, bucket)` rows

S3 prefix:
- `s3://sagemaker-congestion-bogota/congestion-bogota/region-seed-time-artifacts/`

Local job manifest:
- `data/artifacts/region_seed_time_jobs_missing_only/region_seed_time_job_manifest.json`

Runtime decisions:
- use SageMaker Training jobs, not local processing
- use EFS only as read-only source input
- cache data locally under `/opt/ml/input/data/cache/{region}/shard_{id}`
- set `cache-local=True`
- fail early if the intended cached dataset paths are not used

Shard shape:
- `20` central shards
- `20` noncentral shards
- total desired missing-seed jobs: `40`

Initial launch status from morning:
- `30` jobs launched and active
- account-level training concurrency limit observed: `30`
- `10` jobs remained pending because of quota/concurrency
- `ml.r6i.4xlarge` was rejected as invalid for SageMaker Training in this account/region
- `ml.r5d.4xlarge` has quota `0`

Morning active instance mix:
- `ml.m6i.4xlarge`: `10`
- `ml.r7i.4xlarge`: `8`
- `ml.m7i.4xlarge`: `8`
- `ml.m5.4xlarge`: `2`
- `ml.c7i.4xlarge`: `1`
- `ml.r5.4xlarge`: `1`

## Current Verified Status

Checked at `2026-05-14 18:20 CEST`.

SageMaker:
- `30` launched jobs are now `Completed`
- `0` in progress
- `0` failed
- `0` stopped/stopping

Completed shards:
- central: `18/20`
  - completed: `00,01,02,03,04,06,07,08,09,10,11,12,13,14,15,16,18,19`
  - pending: `05,17`
- noncentral: `12/20`
  - completed: `00,02,03,04,05,06,07,08,10,11,15,19`
  - pending: `01,09,12,13,14,16,17,18`

S3 output scan:
- total parquet outputs under `outputs/`: `1538`
- checkpoint parquets: `1478`
- partial parquets: `30`
- final parquets: `30`
- central parquet objects: `1010`
- noncentral parquet objects: `528`

Latest pending retry set from manifest:
- `congestion-bogota-seedtime-cen-05-20260514-081138`
- `congestion-bogota-seedtime-cen-17-20260514-081138`
- `congestion-bogota-seedtime-non-01-20260514-081138`
- `congestion-bogota-seedtime-non-09-20260514-081138`
- `congestion-bogota-seedtime-non-12-20260514-081138`
- `congestion-bogota-seedtime-non-13-20260514-081138`
- `congestion-bogota-seedtime-non-14-20260514-081138`
- `congestion-bogota-seedtime-non-16-20260514-081138`
- `congestion-bogota-seedtime-non-17-20260514-081138`
- `congestion-bogota-seedtime-non-18-20260514-081138`

Retry helper:
- `scripts/retry_failed_seed_time_jobs.py`

Recommended retry command:

```bash
/home/jose_marquez/miniconda3/envs/traffic-gnn-py310/bin/python scripts/retry_failed_seed_time_jobs.py --wait-for-slots
```

## Finalization Step

Implementation:
- `scripts/finalize_region_seed_time_artifacts.py`

Purpose:
- download/collect regional shard outputs
- merge newly generated rows with reused existing rows
- pick final shard outputs when available, otherwise latest checkpoint or partial output
- rebucket per region using quantiles
- optionally upload final regional artifacts back to S3

Expected command after all missing shards complete:

```bash
/home/jose_marquez/miniconda3/envs/traffic-gnn-py310/bin/python scripts/finalize_region_seed_time_artifacts.py --download --upload-final
```

Expected local outputs:
- `data/artifacts/region_seed_time_final/seed_time_bucket_rows_central.parquet`
- `data/artifacts/region_seed_time_final/seed_time_bucket_metadata_central.json`
- `data/artifacts/region_seed_time_final/seed_time_bucket_rows_noncentral.parquet`
- `data/artifacts/region_seed_time_final/seed_time_bucket_metadata_noncentral.json`
- `data/artifacts/region_seed_time_final/final_region_seed_time_manifest.json`

Expected S3 final location:
- `s3://sagemaker-congestion-bogota/congestion-bogota/region-seed-time-artifacts/final/central/`
- `s3://sagemaker-congestion-bogota/congestion-bogota/region-seed-time-artifacts/final/noncentral/`

## Training Modification Started

Checked/updated at `2026-05-14 18:30 CEST`.

Direction:
- do not block the next phase on relaunching the 10 missing seed-time shards
- start adapting training so central/noncentral runs can optimize only their target region

Operational note:
- an attempted retry briefly launched five `r2` SageMaker jobs:
  - `cb-seedtime-cen-05-r2-162626`
  - `cb-seedtime-cen-17-r2-162626`
  - `cb-seedtime-non-01-r2-162626`
  - `cb-seedtime-non-09-r2-162626`
  - `cb-seedtime-non-12-r2-162626`
- all five were stopped after the user redirected away from relaunching
- no final retry manifest was written for that interrupted local helper run

Code work started:
- `TrafficDataset.__getitem__` now emits `node_indices` so training can map a batch-local node position back to graph node ids
- `pad_collate` validates shared `node_indices` across the batch and passes them through
- `training_entrypoint.py` now has:
  - `--training-region {all,central,noncentral}`
  - `--node-region-map-path`
  - S3/local/train-channel resolution for the node region map
  - per-batch regional node masks for loss and operational metrics
- `masked_mse` now handles fully masked targets and avoids deriving class ids from NaN targets

Expected regional training args:

```bash
--training-region central \
--node-region-map-path s3://<bucket>/<path>/node_region_map.csv \
--seed-time-artifact-path s3://<bucket>/<path>/seed_time_bucket_rows_central.parquet
```

and similarly for `noncentral`.

Validation caveat:
- `python -m py_compile` passed for the touched Python files
- targeted pytest could not run in the local `traffic-gnn-py310` environment because importing Torch fails with:
  - `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`

## Regional Training Launch Started

Updated at `2026-05-14 18:50 CEST`.

Seed-time status:
- the 10 missing seed-time shards were relaunched and left running as `r2-163536`
- retry manifest:
  - `data/artifacts/region_seed_time_jobs_missing_only/retry_failed_jobs_20260514-163536.json`
- all 10 relaunches were accepted by SageMaker

Fast partial regional artifacts were created from completed final shard parquets only:
- central:
  - shards included: `18/20`
  - rows: `6,260,000`
  - seeds: `3,130`
  - local: `data/artifacts/region_seed_time_fast_partial/seed_time_bucket_rows_central.parquet`
- noncentral:
  - shards included: `12/20`
  - rows: `5,110,000`
  - seeds: `2,555`
  - local: `data/artifacts/region_seed_time_fast_partial/seed_time_bucket_rows_noncentral.parquet`

Uploaded fast partial artifacts:
- prefix: `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/artifacts/20260514-164619/`
- node map:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/artifacts/20260514-164619/node_region_map.csv`
- central seed-time:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/artifacts/20260514-164619/seed_time_bucket_rows_central.parquet`
- noncentral seed-time:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/artifacts/20260514-164619/seed_time_bucket_rows_noncentral.parquet`

Launcher added:
- `scripts/launch_region_training_jobs.py`

Initial regional smoke jobs launched:
- central:
  - job: `congestion-bogota-cen-gconvlstm-20260514164619`
  - model: `GConvLSTM`
  - instance: `ml.g4dn.8xlarge`
  - epochs: `2`
  - output: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/central/gconvlstm`
- noncentral:
  - job: `congestion-bogota-non-gconvlstm-g5-20260514164619`
  - model: `GConvLSTM`
  - instance: `ml.g5.4xlarge`
  - epochs: `2`
  - output: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/noncentral/gconvlstm`

Launch manifest:
- `data/artifacts/regional_training_jobs/regional_training_manual_launch_20260514-164619.json`

Note:
- `finalize_region_seed_time_artifacts.py --download --output-dir data/artifacts/region_seed_time_final_partial` also completed and produced the same partial seed counts from selected final shard outputs.

Additional model launches at `2026-05-14 18:56 CEST`:
- GConvGRU central:
  - job: `congestion-bogota-cen-gconvgru-g5-20260514185200`
  - instance: `ml.g5.8xlarge`
  - output: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/central/gconvgru`
  - request: `data/artifacts/regional_training_jobs/congestion-bogota-cen-gconvgru-g5-20260514185200.json`
- GConvGRU noncentral:
  - job: `congestion-bogota-non-gconvgru-g4dn-20260514185200`
  - instance: `ml.g4dn.4xlarge`
  - output: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/noncentral/gconvgru`
  - request: `data/artifacts/regional_training_jobs/congestion-bogota-non-gconvgru-g4dn-20260514185200.json`
- TGCN central:
  - job: `cb-cen-tgcn-g5-2xlarge-20260514185600`
  - instance: `ml.g5.2xlarge`
  - output: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/central/tgcn`
  - request: `data/artifacts/regional_training_jobs/cb-cen-tgcn-g5-2xlarge-20260514185600.json`
- TGCN noncentral:
  - job: `cb-non-tgcn-g5-xlarge-20260514185600`
  - instance: `ml.g5.xlarge`
  - output: `s3://sagemaker-congestion-bogota/congestion-bogota/training-runs/regional/noncentral/tgcn`
  - request: `data/artifacts/regional_training_jobs/cb-non-tgcn-g5-xlarge-20260514185600.json`

Launch manifests:
- `data/artifacts/regional_training_jobs/regional_training_gconvgru_tgcn_launch_20260514185200.json`
- `data/artifacts/regional_training_jobs/regional_training_tgcn_launch_20260514185600.json`

## Regional Training Relaunch With Lower Low-Bucket Bias

Updated at `2026-05-14 19:12 CEST`.

Reason:
- user requested stopping the regional training jobs to change the training bucket mix
- previous active model jobs were stopped
- seed-time artifact jobs were intentionally left running

New training sampler mix:
- hyperparameter: `seed-time-bucket-target-ratios=0.30,0.30,0.40`
- order is `high,medium,low`
- intended mix: `30% high / 30% medium / 40% low`

Launcher update:
- `scripts/launch_region_training_jobs.py` now exposes `--seed-time-bucket-target-ratios`
- default changed to `0.30,0.30,0.40`

New regional model jobs:
- `cb-cen-gconvlstm-g4dn-8xlarge-r303040-20260514191200`
- `cb-non-gconvlstm-g5-4xlarge-r303040-20260514191200`
- `cb-cen-gconvgru-g5-8xlarge-r303040-20260514191200`
- `cb-non-gconvgru-g4dn-4xlarge-r303040-20260514191200`
- `cb-cen-tgcn-g5-2xlarge-r303040-20260514191200`
- `cb-non-tgcn-g5-xlarge-r303040-20260514191200`

New launch manifest:
- `data/artifacts/regional_training_jobs/regional_training_r303040_launch_20260514191200.json`

## Remaining Engineering Work

1. Retry the `10` pending shard jobs now that the first `30` jobs completed.
2. Verify the retry jobs complete and produce final parquets.
3. Run `scripts/finalize_region_seed_time_artifacts.py --download --upload-final`.
4. Add training support for regional artifacts:
   - central model uses central seed-time artifact
   - noncentral model uses noncentral seed-time artifact
5. Add region-aware loss masking or weighting:
   - central model should penalize central nodes
   - noncentral model should penalize noncentral nodes
   - avoid accidentally optimizing only sampled seed nodes if the intended target is all nodes in the region-local graph
6. Launch separate regional training jobs and compare against the current global model.

## Key Caution

Do not train the regional models from the original combined artifact. The whole point of this pass is to rebucket and sample from region-specific severity distributions.

## NaN Loss Investigation And Masking Fix

Updated at `2026-05-15 CEST`.

Observed in CloudWatch for `cb-cen-gconvgru-g5-8xlarge-r303040-20260514191200/algo-1-1778778789`:
- regional training mask was active:
  - global selected nodes: `17308 / 139107`
  - first train batch selected nodes: `243 / 500`
- epoch planner honored the requested split closely:
  - high: about `30.8%`
  - medium: about `31.4%`
  - low: about `37.8%`
- first logged train step had:
  - `train_loss=nan`
  - `loss_level=0.460756`
  - `loss_velocity=0.748487`
- subsequent steps had NaN in the simple split losses too.

Interpretation:
- the first forward pass was probably finite because the simple level/velocity diagnostics were finite
- the composite `masked_mse` became NaN before the first backward pass
- after backpropagating NaN once, model weights were contaminated and later diagnostics also became NaN

Code fix applied:
- keep model outputs unchanged: models still predict all nodes in the sampled subgraph/batch
- do not apply regional training by changing model output shape
- pass an explicit boolean `valid_mask` into `src/training/losses.py::masked_mse`
- all composite loss terms now use only finite positions inside that explicit mask
- training/validation now pass the regional node mask directly into the loss instead of relying on NaN-masked targets
- operational metrics also accept the node mask so central/noncentral metrics are computed over the intended region
- added fail-fast diagnostics before `backward()` if a non-finite loss appears again

Verification:
- `python -m py_compile src/training/losses.py src/training_entrypoint.py tests/test_regional_training_masks.py` passed
- local pytest cannot run in the base Python because `torch` is not installed in that environment

Recommended next step:
- run one short SageMaker debug training job with `--debug-batches` before relaunching the full GConvGRU/TGCN/GConvLSTM matrix
- if the debug job still fails, the new log should say whether NaN originated in model outputs before loss or inside the masked composite loss path

## Full Regional Job Relaunch After Explicit Mask Fix

Updated at `2026-05-15 00:22 CEST`.

User requested launching all jobs immediately.

Code/artifact upload:
- source code URI:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/code/20260514-221827/sourcedir.tar.gz`
- artifact prefix:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/artifacts/20260514-221827/`
- artifacts:
  - `node_region_map.csv`
  - `seed_time_bucket_rows_central.parquet`
  - `seed_time_bucket_rows_noncentral.parquet`

Launched jobs:
- central GConvLSTM:
  - `congestion-bogota-cen-gconvlstm-20260514221827`
  - instance: `ml.g4dn.8xlarge`
- central GConvGRU:
  - `congestion-bogota-cen-gconvgru-20260514221827`
  - instance: `ml.g5.8xlarge`
- central TGCN:
  - `congestion-bogota-cen-tgcn-g54-20260514222025`
  - instance: `ml.g5.4xlarge`
- noncentral GConvLSTM:
  - `congestion-bogota-non-gconvlstm-g4dn12xl-20260514222126`
  - instance: `ml.g4dn.12xlarge`
- noncentral GConvGRU:
  - `congestion-bogota-non-gconvgru-g512xl-20260514222128`
  - instance: `ml.g5.12xlarge`
- noncentral TGCN:
  - `congestion-bogota-non-tgcn-g4dn16xl-20260514222130`
  - instance: `ml.g4dn.16xlarge`

Quota notes:
- `ml.g5.8xlarge` quota is `1`; central GConvGRU occupies it
- `ml.g4dn.8xlarge` quota is `1`; central GConvLSTM occupies it
- `ml.g5.4xlarge` quota is `1`; central TGCN occupies it
- `ml.g4dn.12xlarge` quota is `1`; noncentral GConvLSTM occupies it
- noncentral GConvGRU and TGCN were launched on larger available alternatives

Status check immediately after launch:
- all six new jobs were `InProgress`

Launch/request files:
- central request JSONs:
  - `data/artifacts/regional_training_jobs/congestion-bogota-cen-gconvlstm-20260514221827.json`
  - `data/artifacts/regional_training_jobs/congestion-bogota-cen-gconvgru-20260514221827.json`
  - `data/artifacts/regional_training_jobs/congestion-bogota-cen-tgcn-g54-20260514222025.json`
- noncentral request JSONs:
  - `data/artifacts/regional_training_jobs/congestion-bogota-non-gconvlstm-g4dn12xl-20260514222126.json`
  - `data/artifacts/regional_training_jobs/congestion-bogota-non-gconvgru-g512xl-20260514222128.json`
  - `data/artifacts/regional_training_jobs/congestion-bogota-non-tgcn-g4dn16xl-20260514222130.json`
- noncentral adaptive manifest:
  - `data/artifacts/regional_training_jobs/regional_training_noncentral_adaptive_20260514-222132.json`

Next monitoring priority:
- watch first CloudWatch `[METRIC] train` lines
- confirm `train_loss`, `loss_level`, and `loss_velocity` remain finite after first backward
- if a fail-fast `Non-finite train loss before backward` appears, inspect the emitted `pred_stats`, `target_stats`, and selected node counts

## Status Update: Metric Mask Shape Failure And Relaunch

Updated at `2026-05-15 01:03 CEST`.

First full relaunch after the explicit mask fix failed quickly for all six jobs with:
- `RuntimeError: The size of tensor a (400000) must match the size of tensor b (500) at non-singleton dimension 0`

Cause:
- the regional node mask was correctly shaped for loss broadcasting
- but operational metrics flattened `B*N*H` predictions/targets and compared them to the unexpanded `(1,N,1)` mask
- with `B=16`, `N=500`, `H=50`, metrics expected `400000` booleans but had only `500`

Fix:
- `_node_valid_mask_for_target` now broadcasts the regional node mask to `target.shape`
- added a test asserting a `(16,500,50)` target receives a full-shape valid mask
- `python -m py_compile src/training_entrypoint.py tests/test_regional_training_masks.py` passed

Relaunched with new code/artifact upload:
- source code URI:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/code/20260514-230035/sourcedir.tar.gz`
- artifact prefix:
  - `s3://sagemaker-congestion-bogota/congestion-bogota/regional-training/artifacts/20260514-230035/`

Current relaunched jobs:
- `congestion-bogota-cen-gconvlstm-20260514230035` - `ml.g4dn.8xlarge`
- `congestion-bogota-cen-gconvgru-20260514230035` - `ml.g5.8xlarge`
- `congestion-bogota-cen-tgcn-20260514230035` - `ml.g5.4xlarge`
- `congestion-bogota-non-gconvlstm-g4dn12xl-20260514230149` - `ml.g4dn.12xlarge`
- `congestion-bogota-non-gconvgru-g512xl-20260514230153` - `ml.g5.12xlarge`
- `congestion-bogota-non-tgcn-g4dn16xl-20260514230154` - `ml.g4dn.16xlarge`

Immediate status after relaunch:
- all six are `InProgress`
- central GConvLSTM had reached `Downloading`
- the other five were still `Pending`
- no CloudWatch training metric lines were available yet

## Status Update: Training Metrics Are Finite

Updated at `2026-05-15 01:22 CEST`.

All six relaunched jobs are `InProgress` with secondary status `Training`:
- `congestion-bogota-cen-gconvlstm-20260514230035`
- `congestion-bogota-cen-gconvgru-20260514230035`
- `congestion-bogota-cen-tgcn-20260514230035`
- `congestion-bogota-non-gconvlstm-g4dn12xl-20260514230149`
- `congestion-bogota-non-gconvgru-g512xl-20260514230153`
- `congestion-bogota-non-tgcn-g4dn16xl-20260514230154`

CloudWatch metrics show finite train losses for all six:
- central GConvLSTM:
  - `train_loss` latest `5.174501`, earliest `9.373883`, points `10`
- central GConvGRU:
  - `train_loss` latest `5.440147`, earliest `7.105253`, points `9`
- central TGCN:
  - `train_loss` latest `5.545184`, earliest `8.059904`, points `7`
- noncentral GConvLSTM:
  - `train_loss` latest `5.487144`, earliest `9.544284`, points `9`
- noncentral GConvGRU:
  - `train_loss` latest `5.397654`, earliest `5.884426`, points `9`
- noncentral TGCN:
  - `train_loss` latest `5.332961`, earliest `8.197734`, points `8`

Other checks:
- no `Non-finite` fail-fast log messages found
- `val_loss` has no points yet, so jobs appear to still be in training epoch work

## Status Update: Training Metric Snapshot

Updated at `2026-05-15 CEST`.

All six relaunched jobs remain `InProgress`.

Latest CloudWatch training metrics, no validation metrics yet:
- central GConvLSTM:
  - `train_loss=5.391822`
  - `loss_level=0.141598`
  - `loss_velocity=0.061816`
  - `train_probe_mae_level=1.175869`
  - `train_probe_accuracy_exact=0.316866`
  - `train_probe_accuracy_pm1=0.627997`
  - `train_probe_class_distance_mean=1.118141`
  - `train_probe_class_distance_ge2_rate=0.372003`
- central GConvGRU:
  - `train_loss=5.385429`
  - `loss_level=0.133435`
  - `loss_velocity=0.048750`
  - `train_probe_mae_level=1.168528`
  - `train_probe_accuracy_exact=0.338978`
  - `train_probe_accuracy_pm1=0.625372`
  - `train_probe_class_distance_mean=1.118927`
  - `train_probe_class_distance_ge2_rate=0.374628`
- central TGCN:
  - `train_loss=5.529241`
  - `loss_level=0.168777`
  - `loss_velocity=0.057374`
  - `train_probe_mae_level=1.299347`
  - `train_probe_accuracy_exact=0.287156`
  - `train_probe_accuracy_pm1=0.533763`
  - `train_probe_class_distance_mean=1.256569`
  - `train_probe_class_distance_ge2_rate=0.466237`
- noncentral GConvLSTM:
  - `train_loss=5.398844`
  - `loss_level=0.079539`
  - `loss_velocity=0.082653`
  - `train_probe_mae_level=0.751126`
  - `train_probe_accuracy_exact=0.527914`
  - `train_probe_accuracy_pm1=0.818669`
  - `train_probe_class_distance_mean=0.669025`
  - `train_probe_class_distance_ge2_rate=0.181331`
- noncentral GConvGRU:
  - `train_loss=5.356152`
  - `loss_level=0.066916`
  - `loss_velocity=0.081872`
  - `train_probe_mae_level=0.668003`
  - `train_probe_accuracy_exact=0.583081`
  - `train_probe_accuracy_pm1=0.860857`
  - `train_probe_class_distance_mean=0.577927`
  - `train_probe_class_distance_ge2_rate=0.139143`
- noncentral TGCN:
  - `train_loss=5.877429`
  - `loss_level=0.068099`
  - `loss_velocity=0.076076`
  - `train_probe_mae_level=0.817644`
  - `train_probe_accuracy_exact=0.490077`
  - `train_probe_accuracy_pm1=0.801483`
  - `train_probe_class_distance_mean=0.743010`
  - `train_probe_class_distance_ge2_rate=0.198517`

Early read:
- central GConvGRU is slightly ahead of central GConvLSTM on current probe metrics
- central TGCN is currently behind on probe metrics
- noncentral GConvGRU is clearly strongest in the current probe snapshot
- wait for validation before making model-selection decisions
