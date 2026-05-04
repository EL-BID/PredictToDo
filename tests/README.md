# Tests

The test suite covers core behavior using synthetic data.

Main coverage areas:

- `test_sparse_dataloader.py`: graph-local batching, sparse HDF5 loading, targets, weather, and imputation behavior
- `test_temporal_split.py`: chronological split semantics and subset access
- `test_prediction_reconstruction.py`: prediction-to-road/timestamp reconstruction
- `preprocessing/test_imputation.py`: graph-aware imputation helpers
- `preprocessing/test_normalize.py`: normalization constants and edge cases

Run:

```bash
conda activate congestion-bogota-py312
python -m pytest --disable-warnings tests
```

Known current failures are documented in `docs/cleanup-plan.md`.
