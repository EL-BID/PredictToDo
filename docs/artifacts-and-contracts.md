# Artifacts And Contracts

This document records the files and contracts that connect preprocessing, training, evaluation, and the real-time service.

## Data Locations

The runnable sample is included under `examples/bogota/`. Larger inputs are
provided separately, through local directories or storage configured by the user.
SageMaker notebooks obtain deployment-specific locations from environment
variables; see [setup.md](setup.md#optional-aws-configuration).

## Graph Contract

The road graph is a NetworkX graph whose node IDs must match `road_id` values used in feature, seed, target, and region-map artifacts.

Typical local path:

```text
data/datos_procesados/conectivity_graph/road_graph.pickle
```

The graph is used for:

- seed-neighborhood construction;
- edge-index extraction;
- regional seed coverage;
- training and inference subgraph context.

## Dataset Contract

`TrafficDataset` expects graph-aligned static and temporal artifacts. The model-facing tensors are:

| Tensor | Meaning |
|---|---|
| `x_static` | per-node static features |
| `x_dynamic_past` | historical dynamic features over the input window |
| `x_dynamic_future` | known future covariates over the forecast horizon |
| `edge_index` | subgraph edges for the selected seed-local node set |
| `y` | target congestion level and optionally velocity |

Feature order is a contract. Any change to feature generation must be reflected in the saved feature map and verified against both training and inference.

## Model Contract

The active model families are:

- `GConvLSTM`
- `GConvGRU`
- `TGCN`
- historical/support path: `A3TGCN`

Trained artifacts should include:

- model weights;
- model configuration;
- feature-order metadata;
- normalization constants or pointers to them;
- enough identity metadata to trace model lineage downstream.

## Seed Contracts

There are two separate seed concepts:

- Training seeds and seed-time rows are used for sampling historical training examples.
- Inference seeds are used to cover the graph at serving time.

Do not mix these artifacts without checking the intended use.

Global inference seed list:

```text
data/artifacts/seed_lists/seeds_k6.json
```

Regional inference seed lists:

```text
data/artifacts/regional_seed_lists/seeds_central_k6_minimal.json
data/artifacts/regional_seed_lists/seeds_noncentral_k6_minimal.json
data/artifacts/regional_seed_lists/regional_seed_cover_manifest_k6.json
```

## Regional Contract

The regional split is defined by `node_region_map.csv` with at least:

| Column | Meaning |
|---|---|
| `road_id` | graph node / road identifier |
| `is_central` | true for principal/central corridor roads |

Training uses this map to mask loss and operational metrics. Inference uses the same map to filter published outputs by model region.

Provide `node_region_map.csv` and `node_region_map_metadata.json` from the same
regional preprocessing version, using local paths or your configured S3 location.

## Calibration And Evaluation Contract

Dense real-time evaluation uses prediction parquet backups plus jam parquet files, interpreting missing jam values as level `0`. Redis filtered keys and S3 full-surface backups can have different semantics, so evaluation code must declare which source it is reading.

Calibration artifacts are exported for the real-time service and should be treated as serving assets, not training-only outputs.

