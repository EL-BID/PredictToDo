# Known Gaps And Cleanup Needs

## 1. Duplicate Files Need Clarification
- `src/sparse_dataloader(2).py`
- `src/sparse_dataloader_jose.py`
- `src/training_entrypoint(2).py`
- `src/training_utils(2).py`

These should either be:
- clearly documented as archived/experimental
- or removed once no longer needed

## 2. README Is More Dataset-Focused Than Repo-Focused
- The root `README.md` explains dataset preparation well.
- It does not yet explain the newer SageMaker training and artifact-driven sampling architecture in equivalent detail.

## 3. Sampling Runtime Still Needs More Explicit Observability
- The current code works, but logs should eventually expose:
  - realized epoch seed count
  - realized bucket counts per epoch
  - plan size and batch composition
  - explicit shape-loss component metrics

## 4. Test Coverage Lags Behind Recent Runtime Changes
- The new seed-time planner and sampler are not yet strongly unit-tested.

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
## 7. Inference Backup Parquets Lack Model Traceability
- The sibling inference repo writes parquet backups to S3 without model identity metadata.
- It does compute `pred_meta` in memory, but only Redis gets it.
- This blocks reliable retrospective attribution of a prediction file to a specific deployed model.

## 8. Training Checkpoints Do Not Carry A Durable Version Identity
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

## 9. Real-Time Dense Evaluation Is Still Partial
- A stable `10`-prediction-file dense evaluation exists.
- The larger last-7-day run is still incomplete.
- That means current conclusions about real-time error are directionally useful but not yet final.
