# Setup

## Python Version

The repository targets Python `3.12`.

Local metadata is configured through:

- `.python-version`
- `pyproject.toml`
- `uv.lock`

## Existing Local Environment

This workspace has a prepared conda environment:

```bash
conda activate congestion-bogota-py312
```

It uses Python `3.12.12` and includes the PyTorch Geometric packages required by the test suite and model imports.

## Fresh Environment

```bash
conda create -n congestion-bogota-py312 python=3.12 -y
conda activate congestion-bogota-py312
python -m pip install uv
uv sync --all-groups
```

If PyG extension packages try to compile locally, install matching prebuilt wheels. For the current CPU validation environment:

```bash
python -m pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
```

## Local Checks

```bash
make test
make lint
make format
```

Current known failures are listed in [cleanup-plan.md](cleanup-plan.md).
