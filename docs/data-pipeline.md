# Data Pipeline

## Purpose

The preprocessing pipeline converts raw geospatial, traffic, event, incident, PMT, weather, and road-state inputs into sparse graph-temporal datasets suitable for GNN-temporal training.

## Entry Point

```bash
uv run python -m src.prepare_dataset
```

## Main Stages

1. Feature generation creates processed datasets from raw source files.
2. Neighbor imputation computes graph-aware fallbacks for sparse or missing signals.
3. Sparse merging aligns features into HDF5 COO-style matrices.
4. Normalization computes feature-wise constants.
5. Dataset instantiation creates objects compatible with `TrafficDataset`.

## Feature Families

Static and structural features include buildings, bus stops, road graph attributes, and road state.

Dynamic features include Waze jams, Waze irregularities, public events, incidents, PMT points, PMT tracts, and weather.

## Storage Contract

Large data files are intentionally excluded from Git. Expected local folders include:

- `data/`
- `datos/`
- `datos_procesados/`

Training jobs commonly read preprocessed data from EFS and can download some artifacts from S3.

## Important Runtime Contract

`TrafficDataset` keeps batch graph consistency by using one seed-local subgraph per sample batch. The seed determines the node neighborhood; time windows vary inside the training plan.
