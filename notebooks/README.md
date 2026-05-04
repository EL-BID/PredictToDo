# Notebooks

This folder contains both operational launch notebooks and exploratory notebooks.

Canonical operational notebooks:

- `sagemaker_training_job.ipynb`: primary SageMaker training launcher
- `sagemaker_calibration_job.ipynb`: calibration launcher
- `sagemaker_seed_time_sharded_launcher.ipynb`: shard-based seed-time artifact generation
- `sagemaker_seed_time_artifact_builder.ipynb`: smaller or earlier seed-time artifact builder
- `inference_production_run.ipynb`: inference run helper

Exploratory or historical notebooks/scripts:

- `training_example.ipynb`
- `training_example_weighted.ipynb`
- `training_epoch.ipynb`
- `example_sparsity.ipynb`
- `evaluation_example.ipynb`
- `loss_tests.py`
- `connected_components_analysis.py`
- `Extract_buildings.py`

Before external delivery, this folder should be split or labeled more strictly so launch notebooks are not mixed with experiments.
