# Seed-Time Sampling Status

## Problem That Drove The Change
- Earlier training runs looked heavily dominated by low-severity cases and class `0`.
- Time-only probes showed:
  - global rebalance was technically possible
  - but sample quality still needed to align with the actual seed-local graph seen by the model

## Main Design Decision
- The durable sampling unit should be:
  - `(seed_id, time_idx)`
- not only:
  - `time_idx`

Reason:
- the model sees a sample defined by a seed-local subgraph plus a time window
- a global time-only severity score is less faithful than a seed-local one

## Important Correction
- Sparse missing values must count as `0`
- This applied both to:
  - probe scoring
  - artifact scoring

Without this correction, high-severity fractions were inflated.

## Artifact Generation Strategy

### First universe
- `937` seeds
- `2000` `time_idx`
- `1,874,000` rows after merge

### Expansion
- additional `1000` seeds
- same `2000` `time_idx`
- `2,000,000` rows after merge

### Final combined universe
- `1937` seeds
- `2000` `time_idx`
- `3,874,000` rows

## Final Combined Thresholds
From `data/artifacts/seed_time_merged_all/seed_time_bucket_metadata_merged_all.json`:
- `score_mean = 0.016727063891599978`
- `score_max = 0.3`
- bucket thresholds:
  - low/high cut 1: `0.01168`
  - low/high cut 2: `0.0228`

## Final Combined Bucket Counts
- `low = 1,939,990`
- `medium = 963,980`
- `high = 970,030`

This is very close to:
- `50% low`
- `25% medium`
- `25% high`

## Runtime Training Behavior
- The artifact is not sampled row-by-row globally.
- Instead, training uses it to build:
  - one seed per batch
  - 16 `time_idx` per batch
  - approximately `45/30/25` bucket mix inside the batch

## Current Implementation Status
- `src/training_utils.py`
  - planner + planned sampler implemented
- `src/training_entrypoint.py`
  - integrated
- `src/sparse_dataloader.py`
  - intentionally left on the original indexing/materialization contract

## Why Performance Should Hold
- The new logic sits outside the tensor materialization path.
- The dataset still receives standard integer indices.
- The heavy change is in planning, not in `__getitem__()`.
