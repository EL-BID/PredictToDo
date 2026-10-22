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

See [validation guidance](validation.md) for the checks and their scope.

## Optional AWS Configuration

The local example does not require AWS. To use the SageMaker development
notebooks, export the following settings in the environment used to start
Jupyter. A `.env` file is not loaded automatically. The first cell lists the
settings required by that notebook and stops if any are missing.

| Environment variable | Value to supply |
|---|---|
| `SAGEMAKER_BUCKET` | Your training/source-code S3 bucket name |
| `SAGEMAKER_ROLE_ARN` | Execution role ARN in your AWS account |
| `SAGEMAKER_IMAGE_URI` | Compatible training container URI in ECR |
| `SAGEMAKER_EFS_ID` | EFS filesystem containing prepared inputs |
| `SAGEMAKER_SECURITY_GROUP_IDS` | Comma-separated security group IDs |
| `SAGEMAKER_SUBNET_IDS` | Comma-separated subnet IDs |
| `SAGEMAKER_OUTPUT_S3_URI` | Full S3 output prefix for artifact/probe jobs |
| `SEED_TIME_ARTIFACT_S3_URI` | Full URI of the prepared seed-time artifact |
| `MODEL_GCONVGRU_S3_URI` | GConvGRU model bundle URI, for calibration |
| `MODEL_GCONVLSTM_S3_URI` | GConvLSTM model bundle URI, for calibration |
| `MODEL_TGCN_S3_URI` | TGCN model bundle URI, for calibration |

Configure AWS authentication using your own profile/session and check the region,
EFS mount directory and instance types in the notebook before running jobs.
SageMaker launch cells create billable resources when executed.

The CLI tools take explicit deployment arguments:

- `launch_region_training_jobs.py` and `prepare_region_seed_time_jobs.py` require
  `--bucket`, `--role-arn`, `--image`, `--efs-id`, `--subnets` and
  `--security-group-ids` (comma-separated lists for the last two).
- `finalize_region_seed_time_artifacts.py` requires `--bucket`.
- `realtime_eval_sample.py` and `realtime_eval_backfill.py` require `--bucket`
  and `--s3-prefix`. The prefix must contain `inference/` and `jams/` folders;
  listing and downloading both use the supplied bucket. `--profile` defaults
  to the AWS CLI profile named `default`.

Use `--help` for other parameters. The production-inference and epoch-training
notebooks resolve paths from the repository root (or its `notebooks/` folder);
provide the required local artifacts at the configured relative paths.
