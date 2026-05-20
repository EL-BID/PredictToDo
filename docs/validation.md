# Validation Checklist

Use this checklist before handing code or artifacts to another person, agent, or deployment workflow.

## Static Checks

Run syntax checks on files touched by the change:

```bash
python -m py_compile <files>
```

Run whitespace/conflict checks:

```bash
git diff --check
```

## Unit Tests

When `pytest` is installed, run targeted tests first:

```bash
python -m pytest tests/test_training_losses.py tests/test_regional_training_masks.py
python -m pytest tests/test_sparse_dataloader.py tests/test_prediction_reconstruction.py
python -m pytest tests/preprocessing
```

Then run the full suite when dependency availability allows it:

```bash
python -m pytest
```

## Training Readiness

Before launching SageMaker:

- Confirm the source package contains the intended `src/training_entrypoint.py` and `src/training/losses.py`.
- Confirm `--seed-time-artifact-path` points to the intended global or regional parquet.
- Confirm regional jobs include both `--training-region` and `--node-region-map-path`.
- Confirm instance type, batch size, node batch size, input sequence length, and forecast horizon match the planned run.
- Confirm output S3 prefix is unique for the experiment.

## Artifact Readiness

Before handing assets to inference:

- Confirm model and feature-order files came from the same training job.
- Confirm normalization constants match the dataset and target scale.
- Confirm regional models use the same feature contract before serving them together.
- Confirm `node_region_map.csv` covers the graph nodes expected by serving.
- Confirm seed lists are inference coverage lists, not training sampling artifacts.

## Documentation Readiness

Every substantial change should update:

- `docs/` when it changes public workflow, architecture, commands, or contracts;
- `ai-context/` when it changes current state, decisions, artifacts, or next steps;
- `docs/code-reference.md` when new Python files, functions, classes, or methods are added.

Regenerate the code reference with the AST inventory script used in the latest documentation pass, or update the affected section manually.

