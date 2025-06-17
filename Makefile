# Default target executed when no arguments are given to make
all: help

TEST_FILES = tests/
PYTHON_FILES=src/

test:
	uv run pytest $(TEST_FILES)

test_cov:
	uv run coverage run -m pytest $(TEST_FILES)
	uv run coverage report -m
	uv run coverage html

format:
	uv run ruff format $(PYTHON_FILES)
	uv run ruff format $(TEST_FILES)