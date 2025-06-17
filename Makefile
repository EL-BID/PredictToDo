# Default target executed when no arguments are given to make
all: help

TEST_FILES = tests/
PYTHON_FILES=./src
MYPY_CACHE=.mypy_cache

test_integration:
	python -m pytest tests/integration_tests/

test_unit:
	python -m pytest tests/unit_tests/ 

test:
	python -m pytest --disable-warnings $(TEST_FILES)

test_profile:
	python -m pytest -vv tests/unit_tests/ --profile-svg


######################
# LINTING AND FORMATTING
######################

lint format: PYTHON_FILES=./src
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d main | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=./src
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	python -m ruff check ./src
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff format $(PYTHON_FILES) --diff
	[ "$(PYTHON_FILES)" = "" ] || python -m ruff check --select I $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || python -m mypy --strict $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || mkdir -p $(MYPY_CACHE) && python -m mypy --strict $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

# format format_diff:
# 	ruff format $(PYTHON_FILES)
# 	ruff check --select I --fix $(PYTHON_FILES)

test_cov:
	coverage run -m pytest $(TEST_FILES)
	coverage report -m
	coverage html

format:
	ruff format $(PYTHON_FILES)
	ruff format $(TEST_FILES)
	ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	codespell --toml pyproject.toml

spell_fix:
	codespell --toml pyproject.toml -w

######################
# HELP
######################

help:
	@echo '----'
	@echo 'format                       - run code formatters'
	@echo 'lint                         - run linters'
	@echo 'test                         - run unit tests'
	@echo 'tests                        - run unit tests'
	@echo 'test TEST_FILES=<test_fileS>   - run all tests in file'
	@echo 'test_watch                   - run unit tests in watch mode'