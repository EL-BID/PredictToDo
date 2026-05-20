# Real-Time Inference Repo

## Purpose
- This document points AI agents to the sibling repository that implements real-time scoring and prediction serving for the same Bogota congestion project.
- The sibling repo is:
  - `<inference-repo-root>`

## Why This Matters
- `congestion-bogota` is the training and data-preparation repository.
- `sdm_pred_congestion_pipeline_inference` is the real-time ingestion and inference repository.
- They share concepts:
  - graph and subgraph inference by seed
  - GNN temporal model families
  - static and dynamic feature contracts
  - reconstruction of city-wide predictions from overlapping seed-local batches
- They do not share the exact same source code.
- That divergence is acceptable because the inference repo has runtime-specific requirements:
  - Ray actor orchestration
  - Redis data hub
  - GPU mega-batching of seeds
  - storage and filtering of predictions for downstream consumers

## What An Agent Should Not Assume
- Do not assume code changes in this repo automatically exist in the inference repo.
- Do not assume model code is identical across repos.
- Do not assume dataloader feature ordering is automatically aligned across repos.
- Do not assume the main Redis prediction key contains the full city-wide prediction surface.

## Main Entry Points In The Inference Repo
- Orchestration:
  - `<inference-repo-root>/service/main.py`
- Prediction runtime:
  - `<inference-repo-root>/service/services/prediction_service.py`
- Real-time feature assembly:
  - `<inference-repo-root>/service/prediction/inference_dataloader.py`
- Redis loading adapter:
  - `<inference-repo-root>/service/prediction/redis_data_adapter.py`
- Inference model implementation:
  - `<inference-repo-root>/service/prediction/models/gnn_temporal.py`
- Deployment topology:
  - `<inference-repo-root>/docker-compose.yml`

## Current Architectural Understanding
- Ray actors ingest multiple external sources and persist them in Redis.
- `PredictionService` periodically:
  - reads Redis
  - determines the current observational anchor timestamp
  - builds graph-aligned dynamic tensors for all nodes
  - slices seed-local subgraphs
  - groups many seeds into a GPU mega-batch
  - runs one forward pass per seed group
  - reconstructs city-wide predictions by averaging overlaps
- The inference repo supports variable-edge mega-batching per seed group, which is a runtime optimization specific to serving.

## Current Regional Execution Finding
- As of 2026-05-20, regional serving in the sibling repo is central-first.
- Regional mode uses `PREDICTION_CENTRAL_SEEDS_PATH` for the required central pass.
- It runs the noncentral pass with `PREDICTION_NONCENTRAL_SEEDS_PATH` only when `PREDICTION_RUN_NONCENTRAL=true`.
- Each model can use all subgraph context as input, but reconstruction filters outputs by `node_region_map.csv`.
- See `20-regional-inference-execution-update-2026-05-20.md` for details.

## Important Behavioral Detail
- In `PredictionService._store()` the primary Redis key stores filtered predictions with:
  - `predicted_level >= 2`
- The per-timeslot Redis keys store unfiltered data for each future slot.
- S3 backup, when enabled, stores the complete unfiltered prediction surface.
- Any downstream consumer must be clear about which storage path it is reading.

## Current Traceability Finding
- The inference repo constructs model identity metadata in memory:
  - `model_file`
  - `model_md5`
  - `model_version`
  - `model_kwargs`
- That metadata is merged into `pred_meta`.
- `pred_meta` is persisted into Redis payloads.
- The S3 parquet backup does **not** persist that metadata.
- Today, prediction parquet files therefore do **not** tell you which deployed model produced them.

## Current Checkpoint-Identity Finding
- `PredictionService` tries to infer model version from checkpoint fields such as:
  - `version`
  - `model_version`
  - `run_id`
- The current training repo checkpoints do not save those fields by default.
- Therefore, even Redis metadata often ends up with:
  - `model_version = unknown`

## Current Real-Time Evaluation Finding
- The S3 backup parquet is the correct source for dense retrospective evaluation because it is intended to hold the full city-wide prediction surface.
- That matters because the main Redis key is filtered and would bias evaluation badly.
- Dense evaluation should therefore use:
  - S3 prediction parquet backups
  - jam parquet files
  - missing jam values interpreted as `0`

## Key Risk Areas To Review Before Modifying Real-Time Scoring

### 1. Model parity
- Training model code in this repo:
  - `src/models/gnn_temporal.py`
- Inference model code in sibling repo:
  - `<inference-repo-root>/service/prediction/models/gnn_temporal.py`
- They are related but not identical.
- Any change to model input contracts, output semantics, or batching assumptions should be checked in both places.

### 2. Feature parity
- Training sparse dataset logic lives here:
  - `src/sparse_dataloader.py`
- Real-time feature assembly lives there:
  - `<inference-repo-root>/service/prediction/inference_dataloader.py`
- This is a high-risk divergence point.
- If features, normalization, ordering, or semantics change in training, the inference repo may need explicit updates.

### 3. Seed strategy divergence
- Training now has a new seed-time artifact and planner-based sampler.
- The real-time inference repo still uses a fixed seed list for city coverage and does not use the training sampler logic.
- This is expected.
- Real-time serving is anchored to "now", not to sampled historical `time_idx`.

## When To Touch The Inference Repo
- When model input dimensions change.
- When feature order or normalization logic changes.
- When seed coverage files or graph assets change.
- When output storage semantics need to change for downstream consumers.
- When real-time scoring quality is suspected to differ from offline validation.

## Recommended Workflow For Future Agents
1. Read this file.
2. Open the sibling repo at:
   - `<inference-repo-root>`
3. Compare training and inference contracts before changing code:
   - model inputs
   - feature groups
   - normalization
   - seed/subgraph batching assumptions
4. Prefer preserving runtime-specific optimizations in the inference repo.
5. Do not mechanically copy training code into serving code without validating:
   - Redis data availability
   - memory profile
   - GPU mega-batch behavior
   - downstream Redis/S3 consumers

## Current Conclusion
- The inference repo is a sibling operational system, not a duplicate code mirror.
- It should evolve with this training repo, but through explicit parity checks and runtime-aware adaptations rather than blind code sync.
