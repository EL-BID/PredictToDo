# Regional Inference Execution Update

Generated: 2026-05-20

This note records the serving-side update made in the sibling inference repository:

`/mnt/c/Users/magno/Documents/GitHub/sdm_pred_congestion_pipeline_inference`

## What Changed

The inference repo now uses the regional seed artifacts documented in `19-regional-inference-seeds-2026-05-18.md`.

Regional execution is central-first:

1. The service always runs the central/principal-road model first when regional inference is enabled.
2. The central pass uses `PREDICTION_CENTRAL_SEEDS_PATH`.
3. The noncentral/non-principal-road pass runs only when `PREDICTION_RUN_NONCENTRAL=true`.
4. The noncentral pass uses `PREDICTION_NONCENTRAL_SEEDS_PATH`.
5. Feature tensors are still built once for the full graph and reused by both passes.

## Region Isolation

Both models may use all roads available in their seed-local subgraphs as input context. The isolation happens at output reconstruction:

- central model outputs are kept only for `is_central=true` roads.
- noncentral model outputs are kept only for `is_central=false` roads.
- predictions for the opposite region are discarded.

The inference dataframe also adds `model_region` for traceability.

## Operational Assets

The node-region map needed by inference was uploaded to the shared artifacts bucket:

```text
s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/corridor_regions/node_region_map.csv
s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/corridor_regions/node_region_map_metadata.json
```

The regional model and seed-list artifacts are:

```text
s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/model_gconvgru_regional_central_2026_05_15.tar.gz
s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/model_gconvgru_regional_noncentral_2026_05_15.tar.gz
s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/regional_seed_lists/seeds_central_k6_minimal.json
s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/regional_seed_lists/seeds_noncentral_k6_minimal.json
s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/regional_seed_lists/regional_seed_cover_manifest_k6.json
```

## Runtime Interpretation

Loading the second model should not be the bottleneck; the extracted model files are small. The meaningful cost is feature assembly plus model forward passes.

The old regional design risked running both models across the same `4398` global seeds. That is roughly `8796` model-seed forward equivalents. The new full regional execution is `657 + 2582 = 3239` forward equivalents, about 63% fewer than the old dual-model global-seed design.

The central seed list touches a materially smaller active graph neighborhood, but the noncentral seed list touches nearly the whole graph. Therefore, the current choice to process features once globally and reuse them for both models is the cleanest near-term serving design.

## Validation Status

The serving repo could not be run end-to-end locally from this environment. Static checks passed:

```text
python -m py_compile service/services/prediction_service.py service/main.py
git diff --check -- service/services/prediction_service.py service/main.py docker-compose.yml REGIONAL_MODELS.md
```

`docker compose config` was blocked by a missing repository-local `.env`.

