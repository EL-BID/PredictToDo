# System Overview

This repository is the training, artifact-building, and evaluation side of the Bogota congestion system. It is not the real-time service repository.

## Repository Boundaries

`congestion-bogota-git` owns:

- preprocessing logic for historical and static datasets;
- graph-temporal dataset construction;
- model definitions used for training;
- SageMaker-compatible training entrypoints;
- loss functions and operational metrics;
- seed, seed-time, calibration, and regional artifact scripts;
- offline and dense real-time evaluation utilities;
- human-facing documentation and AI handoff notes.

The real-time Bogota service lives separately in:

```text
/mnt/c/Users/magno/Documents/GitHub/sdm_pred_congestion_pipeline_inference
```

That service owns Ray actors, Redis ingestion, GPU inference execution, Docker deployment, and production scheduling. Code should not be copied blindly between repositories because the runtime assumptions are different.

## Main System Flow

The training-side flow is:

1. Raw and shared data are stored in S3, EFS, or local `data/` folders.
2. Preprocessing modules build static, dynamic, graph, and normalization artifacts.
3. `TrafficDataset` reads graph-aligned artifacts and yields seed-local temporal graph batches.
4. Training jobs run through `src/training_entrypoint.py`, usually launched by SageMaker notebooks.
5. Optional seed-time artifacts rebalance training exposure across congestion buckets.
6. Optional regional masks train central and noncentral objectives over the intended nodes while preserving seed-local graph context.
7. Evaluation and calibration scripts produce dense metrics, calibration artifacts, and model-selection evidence.
8. Final model, feature-order, seed-list, and region-map artifacts are published for the real-time service.

## Core Invariants

The most important technical invariants are:

- A training batch uses one seed-local graph context at a time.
- `edge_index` must correspond to the selected seed-local node set.
- Static, past-dynamic, future-dynamic, and target tensors must keep the feature order expected by the model.
- Regional training and regional inference do not remove graph context; they scope the loss, metrics, or published outputs by node region.
- The real-time inference repository may reuse trained artifacts, but it has a separate data loader and serving implementation.

## Where To Start

For a new maintainer:

1. Read `docs/README.md`.
2. Read this file.
3. Read `docs/artifacts-and-contracts.md`.
4. Read `docs/process-runbooks.md`.
5. Use `docs/code-reference.md` to locate functions/classes.
6. Review the relevant documentation and tests before making changes.

