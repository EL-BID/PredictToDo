# Source Package

`src/` contains the runtime code for the project.

Important modules:

- `prepare_dataset.py`: orchestration for preprocessing and dataset preparation
- `sparse_dataloader.py`: `TrafficDataset`, the central graph-temporal dataset abstraction
- `training_entrypoint.py`: SageMaker-compatible training entrypoint
- `training_utils.py`: samplers, collate functions, and batch-planning utilities
- `models/gnn_temporal.py`: graph-temporal model definitions
- `training/`: training loop and loss code
- `preprocessing/`: feature generation, merging, imputation, and normalization
- `inference_dataloader.py`: real-time/batch inference input preparation
- `inference_production.py`: production-style inference command
- `prediction_reconstruction.py`: maps predictions back to road/timestamp outputs

The most important invariant is that one batch must use one seed-local graph. This keeps the `edge_index` contract stable.

For a generated inventory of Python classes, functions, and methods, see `../docs/code-reference.md`.
