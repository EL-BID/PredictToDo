# Inference And Evaluation

## Inference Components

Primary files:

- `src/inference_dataloader.py`
- `src/inference_production.py`
- `src/prediction_reconstruction.py`

Inference differs from training because it reads current parquet inputs rather than historical sparse HDF5 targets.

## Basic Inference Command

```bash
uv run python -m src.inference_production \
  --mode single \
  --graph data/datos_procesados/conectivity_graph/road_graph.pickle \
  --static data/datos_procesados/sparse_datasets/static_dataset.h5 \
  --norm data/datos_procesados/sparse_datasets/normalization_constants.pkl \
  --jams data/inference_tests/df_jams_test.parquet \
  --irregularities data/inference_tests/df_irregularidades_test.parquet \
  --weather data/inference_tests/df_clima_test.parquet \
  --pmt-puntos data/inference_tests/df_pmt_puntos_test.parquet \
  --pmt-tramos data/inference_tests/df_pmt_tramos_test.parquet
```

## Evaluation Scripts

Real-time evaluation utilities live in `scripts/`:

- `realtime_eval_sample.py`
- `realtime_eval_metrics.py`
- `realtime_eval_backfill.py`
- `realtime_eval_last7d_dense.py`

## Current Evaluation Caveats

The latest dense real-time evaluation work shows useful signal but is still incomplete for full last-seven-day conclusions. The repository needs a stable summary artifact and clearer model traceability in prediction backups.
