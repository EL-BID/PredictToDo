# Bogota Traffic Congestion Forecasting

This repository contains the research and operational code for graph-temporal traffic congestion forecasting in Bogota. It builds sparse spatio-temporal datasets, trains GNN-temporal models on SageMaker, reconstructs forecasts for inference, and supports offline and real-time evaluation.

The current project center is not only dataset preparation. The active system includes:

- sparse graph-temporal preprocessing over road, Waze, PMT, event, incident, weather, bus stop, building, and road-state sources
- `TrafficDataset`, a seed-local graph batching dataset used by training and evaluation
- GNN-temporal models including `GConvLSTM`, `GConvGRU`, and `TGCN`
- SageMaker training entrypoints and launch notebooks
- artifact-driven seed-time sampling for balanced congestion training
- calibration and real-time dense evaluation utilities
- a LaTeX white paper under `white_paper/`

## Repository Layout

```text
.
├── src/                 # Runtime package: preprocessing, dataset, models, training, inference
├── tests/               # Unit tests for dataset, temporal split, reconstruction, preprocessing
├── notebooks/           # SageMaker launchers and exploratory notebooks
├── scripts/             # Operational and evaluation scripts
├── docs/                # Human-facing project documentation
├── ai-context/          # AI handoff notes and current experiment state
├── white_paper/         # LaTeX source for the project paper
├── pyproject.toml       # Python dependencies and tool configuration
├── uv.lock              # Locked Python 3.12 dependency resolution
├── Makefile             # Common local commands
└── Dockerfile.sagemaker # SageMaker image recipe
```

Read these first:

- [docs/README.md](docs/README.md) for the documentation index
- [docs/repository-structure.md](docs/repository-structure.md) for the code map
- [docs/setup.md](docs/setup.md) for local environment setup
- [ai-context/00-start-here.md](ai-context/00-start-here.md) for the latest AI handoff state

## Environment

The repository is standardized on Python `3.12`.

An already-created local conda environment for this workspace is:

```bash
conda activate congestion-bogota-py312
```

To create a fresh environment elsewhere:

```bash
conda create -n congestion-bogota-py312 python=3.12 -y
conda activate congestion-bogota-py312
python -m pip install uv
uv sync --all-groups
```

PyTorch Geometric extension packages such as `torch-scatter` and `torch-sparse` may need prebuilt wheels that match the local PyTorch build. If `uv sync` tries to compile them from source, use the official PyG wheel index for your PyTorch version.

For the current CPU validation environment:

```bash
python -m pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
```

## Common Commands

Run tests:

```bash
make test
```

Run lint:

```bash
make lint
```

Format Python files:

```bash
make format
```

Download source data, if you have AWS access:

```bash
uv run -- dotenv run -- aws s3 sync \
  s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/ \
  ./data
```

## Data Pipeline

The canonical preprocessing entrypoint is:

```bash
uv run python -m src.prepare_dataset
```

The pipeline creates processed static and dynamic features, merges them into sparse HDF5 datasets, computes normalization constants, and creates a `TrafficDataset`-compatible artifact set. See [docs/data-pipeline.md](docs/data-pipeline.md).

Large data and generated artifacts are intentionally ignored by Git. They should live under `data/`, `datos/`, `datos_procesados/`, EFS, or S3 depending on the workflow.

## Training

The canonical SageMaker entrypoint is:

```bash
python -m src.training_entrypoint
```

In practice, training is launched through the SageMaker notebooks in `notebooks/`, especially `notebooks/sagemaker_training_job.ipynb`.

The active training path supports:

- old seed-based sampling
- artifact-driven `(seed_id, time_idx)` sampling
- model selection among `GConvLSTM`, `GConvGRU`, and `TGCN`
- shape-following auxiliary loss controls

See [docs/training.md](docs/training.md) and [docs/rebalance_sampler_design.md](docs/rebalance_sampler_design.md).

## Inference And Evaluation

Inference utilities are in:

- `src/inference_dataloader.py`
- `src/inference_production.py`
- `src/prediction_reconstruction.py`

Evaluation scripts are in `scripts/`, including recent real-time dense evaluation utilities.

See [docs/inference-evaluation.md](docs/inference-evaluation.md).

## Current Quality State

As of the latest local validation in Python `3.12.12`:

- `118` tests pass
- `ruff check .` passes

Remaining cleanup tasks are tracked in [docs/cleanup-plan.md](docs/cleanup-plan.md).

## White Paper

The paper source is in `white_paper/traffic_forecasting.tex`, with references in `white_paper/references.bib`.

Build outputs such as `.aux`, `.bbl`, `.fls`, `.out`, and `.synctex.gz` are ignored. See [white_paper/README.md](white_paper/README.md).

## AI Context

`ai-context/` is a handoff layer for AI-assisted development. It captures current decisions, experiment status, known gaps, and next steps. It should not replace the public-facing documentation in `docs/`, but it is the best place to check the current state before changing training, inference, or evaluation behavior.
