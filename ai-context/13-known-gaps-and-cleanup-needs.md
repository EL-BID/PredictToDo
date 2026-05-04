# Known Gaps And Cleanup Needs

## 1. Git Review Noise Needs Final Normalization
- The repo previously had large CRLF-only diffs.
- `.gitattributes` now pins text files to LF.
- Before final delivery, review the remaining diff and avoid mixing line-ending normalization with unrelated logic changes.

## 2. Documentation Layer Was Added But Needs Review
- Root `README.md` now describes the full project, not only dataset preparation.
- `docs/` now contains first-pass human-facing documentation.
- These docs should be reviewed for stakeholder-specific wording before external handoff.

## 3. Sampling Runtime Still Needs More Explicit Observability
- The current code works, but logs should eventually expose:
  - realized epoch seed count
  - realized bucket counts per epoch
  - plan size and batch composition
  - explicit shape-loss component metrics

## 4. Test Coverage Still Lags Behind Recent Runtime Changes
- Local tests currently pass in Python `3.12.12`.
- The new seed-time planner and sampler are still not strongly unit-tested.

## 5. Metric Surface Is Compressed By SageMaker Limits
- The runtime can compute more metrics than SageMaker can conveniently parse.
- If richer inspection is needed, structured JSON metric artifacts will be better than relying only on `MetricDefinitions`.

## 6. Some Notebook / Artifact Areas Are Still Operational Rather Than Curated
- The repo has many notebooks and generated artifacts.
- A future cleanup should separate:
  - canonical launch notebooks
  - exploratory notebooks
  - generated artifact directories

## 7. Current Live Experiment Results Still Need Synthesis

## 8. Inference Backup Parquets Lack Model Traceability
- The sibling inference repo writes parquet backups to S3 without model identity metadata.
- It does compute `pred_meta` in memory, but only Redis gets it.
- This blocks reliable retrospective attribution of a prediction file to a specific deployed model.

## 9. Training Checkpoints Do Not Carry A Durable Version Identity
- `training_entrypoint.py` currently saves:
  - `state_dict`
  - `model_kwargs`
  - `torch_version`
- It does not save:
  - `version`
  - `run_id`
  - `training_job_name`
  - `artifact source`
- As a result, the sibling inference repo usually resolves `model_version = unknown`.

## 10. Real-Time Dense Evaluation Is Still Partial
- A stable `10`-prediction-file dense evaluation exists.
- The larger last-7-day run is still incomplete.
- That means current conclusions about real-time error are directionally useful but not yet final.
