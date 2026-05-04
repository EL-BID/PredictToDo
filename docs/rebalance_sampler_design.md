# Seed-Time Sampler Design

## Problem

Early training was dominated by low-congestion and zero-congestion samples. Sampling by time alone did not guarantee that a selected time window was informative for a given seed-local subgraph.

## Design

The newer approach precomputes artifact rows keyed by:

- `seed_id`
- `time_idx`
- severity bucket

At epoch construction time, the planner groups artifact rows by seed and bucket, then builds one batch per seed with a controlled mix of time indices.

## Batch Invariant

Each batch must use exactly one seed. This keeps all examples in the batch on one seed-local subgraph and therefore one compatible `edge_index`.

The sampler may vary `time_idx` inside the batch, but it must not mix unrelated seeds inside the same batch.

## Current Target Mix

The intended bucket mix is approximately:

- high: `45%`
- medium: `30%`
- low: `25%`

With `batch_size = 16`, the practical approximation is `7` high, `5` medium, and `4` low samples.

## Implementation

Primary files:

- `src/rebalance_sampling.py`
- `src/seed_strategy.py`
- `src/time_index_strategy.py`
- `src/merge_seed_time_artifacts.py`
- `src/training_utils.py`
- `src/training_entrypoint.py`

The runtime intentionally preserves `TrafficDataset.__getitem__()` semantics and externalizes planning into sampler utilities.
