# The single source of truth for formatting and linting rules is `pyproject.toml`
# Pre-commit hooks enforce these rules automatically on commit

.PHONY: all help format check lint test test-cov test-unit test-integration test-watch test-profile test-extended spell-check spell-fix

# Let's silence the command echoing for a cleaner output
.SILENT:

TEST_DIR=./tests
SOURCE_DIR=./src
NOTEBOOKS_DIR=./notebooks

# Define tool shortcuts for consistency and easy updates
UV=uv run
PYTEST=$(UV) pytest
RUFF=$(UV) ruff
CODESPELL=$(UV) codespell
JUPYTEXT=$(UV) jupytext

# ------------------------------------------------------------------------------
# CODE QUALITY (Formatting & Linting)
# ------------------------------------------------------------------------------

format:
	@echo "=> Formatting all Python files with Ruff..."
	$(RUFF) format .
	@echo "=> Sorting imports with Ruff..."
	$(RUFF) check --select I --fix .

check: lint

lint:
	@echo "=> Checking formatting (re-run 'make format' if this fails)..."
	$(RUFF) format --diff .
	@echo "\n=> Checking for linting errors with Ruff..."
	$(RUFF) check .

spell-check:
	@echo "=> Checking for spelling errors..."
	$(CODESPELL) --toml pyproject.toml

spell-fix:
	@echo "=> Fixing spelling errors..."
	$(CODESPELL) --toml pyproject.toml -w

# ------------------------------------------------------------------------------
# TESTING
# ------------------------------------------------------------------------------

test:
	@echo "=> Running the complete test suite..."
	$(PYTEST) --disable-warnings $(TEST_DIR)

test-cov:
	@echo "=> Running tests and generating coverage report with pytest-cov..."
	$(PYTEST) --cov=$(SOURCE_DIR) --cov-report=term-missing --cov-report=html $(TEST_DIR)
	@echo "\nCoverage report generated. Open 'htmlcov/index.html' to view."

# ------------------------------------------------------------------------------
# JUPYTER NOTEBOOKS
# ------------------------------------------------------------------------------

from_jupyter:
	@echo "=> Pairing all Jupyter notebooks with their Python scripts..."
	$(JUPYTEXT) --set-formats ipynb,py:percent $(NOTEBOOKS_DIR)/*.ipynb

to_jupyter:
	@echo "=> Converting all Python scripts back to Jupyter notebooks..."
	@for pyfile in $(NOTEBOOKS_DIR)/*.py; do \
		$(JUPYTEXT) --sync "$$pyfile"; \
	done