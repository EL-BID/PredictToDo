# Repository Structure

## Runtime Code

`src/` contains the Python package used by preprocessing, training, inference, and evaluation.

- `src/preprocessing/` contains feature engineering, sparse merging, imputation, and normalization.
- `src/models/` contains the graph-temporal model definitions.
- `src/training/` contains loss and loop code used by SageMaker training.
- `src/utils/` contains small shared helpers.
- `src/sparse_dataloader.py` owns `TrafficDataset`, the central dataset abstraction.
- `src/training_entrypoint.py` is the SageMaker training entrypoint.
- `src/inference_dataloader.py`, `src/inference_production.py`, and `src/prediction_reconstruction.py` support inference and output reconstruction.

## Tests

`tests/` contains focused unit tests for:

- sparse data loading and graph-local batching
- temporal train/test splitting
- prediction reconstruction
- preprocessing imputation
- preprocessing normalization

## Notebooks

`notebooks/` contains SageMaker launch notebooks and exploratory notebooks. The canonical operational notebooks are documented in [../notebooks/README.md](../notebooks/README.md).

## Scripts

`scripts/` contains command-line utilities for seed coverage, calibration artifact export, output transforms, and real-time evaluation. See [../scripts/README.md](../scripts/README.md).

## Documentation And Paper

`docs/` contains project documentation. `ai-context/` contains AI handoff state. `white_paper/` contains the LaTeX paper source.

## Generated Files

Generated data, sparse artifacts, model artifacts, notebook checkpoints, Python caches, and LaTeX build outputs should not be committed. Large artifacts should be stored in S3, EFS, or ignored local data folders.
