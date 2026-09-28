# Cleanup Plan

## Current Status

The repository is now configured for Python `3.12`, but several cleanup tasks remain before an external handoff.

## Priority 1: Make Git Reviewable

- Normalize line endings through `.gitattributes`.
- Avoid committing generated files from Python, Jupyter, LaTeX, data, or model outputs.
- Review the current large CRLF-only diff before creating a final commit.

## Priority 2: Restore Green Local Checks

Current Python `3.12.12` test result:

- `118` passed

The sparse dataloader tests now verify raw dynamic values in the first feature channels and temporal covariates in the appended channels.

Current Ruff result:

- `ruff check .` passes

## Priority 3: Documentation Completeness

- Keep root `README.md` as the project entrypoint.
- Keep `docs/` as human-facing documentation.
- Keep `white_paper/` focused on LaTeX source, bibliography, and final paper assets.

## Priority 4: Runtime Traceability

- Add durable identity metadata to training checkpoints.
- Persist model identity into real-time prediction backup parquet outputs.
- Emit richer JSON metrics when SageMaker metric-definition limits are too restrictive.
