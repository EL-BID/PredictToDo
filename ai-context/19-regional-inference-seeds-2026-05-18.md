# Regional Inference Seed Lists

Generated: 2026-05-18

This file documents the recalculated base seed lists for regional real-time inference. These are inference coverage seeds only; they do not include the top-congestion extra seeds used for training-time sampling.

## Objective

The previous operational seed list was global:

- `data/artifacts/seed_lists/seeds_k6.json`
- `4398` seeds
- `k = 6`
- graph: `data/datos_procesados/conectivity_graph/road_graph.pickle`

For the regional two-model deployment, the executor needs one seed list for principal/central roads and another seed list for secondary/noncentral roads. The goal was to keep the same `k=6` neighborhood convention while minimizing seed count per region.

## Method

The new script is:

- `scripts/compute_regional_seed_cover.py`

It uses a greedy set-cover approximation per region:

1. Load `road_graph.pickle`.
2. Load `data/artifacts/corridor_regions/node_region_map.csv`.
3. Split nodes by `is_central`.
4. For each region, use only candidate seeds inside that same region.
5. For each candidate seed, compute reachable nodes within `k=6` graph hops, counting only target nodes from that region.
6. Repeatedly select the seed that covers the largest number of still-uncovered regional nodes.

The produced seeds are therefore optimized for regional inference coverage, not training sample diversity.

## Outputs

Local outputs:

| Region | Path | Region nodes | Seeds | Coverage |
|---|---|---:|---:|---:|
| central | `data/artifacts/regional_seed_lists/seeds_central_k6_minimal.json` | `17308` | `657` | `100.0%` |
| noncentral | `data/artifacts/regional_seed_lists/seeds_noncentral_k6_minimal.json` | `121799` | `2582` | `100.0%` |
| manifest | `data/artifacts/regional_seed_lists/regional_seed_cover_manifest_k6.json` | `139107` | `3239` | `100.0%` |

Compared with the old global `4398` seed list, the regional split uses `3239` total seeds. That is `1159` fewer seeds, a `26.35%` reduction.

## Shared S3 Artifacts

The same files were uploaded to the shared artifacts bucket:

- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/regional_seed_lists/seeds_central_k6_minimal.json`
- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/regional_seed_lists/seeds_noncentral_k6_minimal.json`
- `s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/artifacts/regional_seed_lists/regional_seed_cover_manifest_k6.json`

S3 tags applied:

- central seeds: `project=congestion-bogota`, `role=regional-inference-seeds`, `region=central`, `k_hop=6`, `created_date=2026-05-18`
- noncentral seeds: `project=congestion-bogota`, `role=regional-inference-seeds`, `region=noncentral`, `k_hop=6`, `created_date=2026-05-18`
- manifest: `project=congestion-bogota`, `role=regional-inference-seed-manifest`, `region=central-noncentral`, `k_hop=6`, `created_date=2026-05-18`

## Execution Implication

The existing single-list inference path can still use `seeds_k6.json`, but that does not exploit this reduction. To use these new minimal lists, the real-time executor should run central inference over `seeds_central_k6_minimal.json` and noncentral inference over `seeds_noncentral_k6_minimal.json`, then merge/reconstruct outputs by road and timestamp.

If the executor still runs both models on the same seed list, these artifacts will not reduce runtime.
