## Setting up the environment
- If `uv` is not installed in your machine, just install it with `pip install uv`.
- All the necessary Python packages must be installed. This is done by executing `uv sync`.
- Then, the correct environment variables must be set. See `.env.example` for an example.

## Downloading the data
To download the data in the `data/` directory, run:
```
uv run -- dotenv run -- aws s3 sync s3://ine-projects-incoming-data/congestion-prediction/co/movilidadbogota/ ./data
```

## Using `uv`
- To add dependencies to the project, run `uv add <package_name>`.
- To remove dependencies from the project, run `uv remove <package_name>`.
- To add dependencies to the project that will only be used during development (such as `jupyer`, `pytest` or `ruff`), run `uv add --dev <package_name>`.
- To synchronize the local environment with the requirements in `uv.lock`, run the command `uv sync`.
- A full list of `uv` commands can be found [here](https://docs.astral.sh/uv/reference/cli/).

## Using the `Makefile`
- Run `make test` to run all the tests inside the `tests/` directory.
- Run `make test_cov` to get the test coverage report.
- Run `make format` to format all the code using `ruff`.

## Reminders
- The first notebook that defines the skeleton of the graph is `graphs_construction.ipynb` (uses `2_Estructura_via`).
- The data in `3_clima` has to be sent again by the Bogota people.
- The second notebook is `irregularities_processing.ipynb` builds the irregularities in `datos_procesados/irregularidades_procesadas` from `4_velocidades_waze/irregularidades`.
- The folders `4_velocidades_waze/jams` and `4_velocidades_waze/velocidades` already has the processed data (for the former, the code is in `tratamiento_jamsBID.ipynb`).
- `5_volumenes_sema` has not been dealt with yet (it only has information about the spots where there are traffic lights--we can't use it, at least easily).
- `incidentsPreprocessing.ipynb` processes the data in `7_incidentes` into `datos_procesados/incidents/`.
- 