PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin
CONFIG ?= configs/default.yaml
FAST   := configs/fast.yaml
SEEDS  ?= 5

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: $(BIN)/python ## Install runtime and development dependencies
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

.PHONY: data
data: ## Download the UCI Air Quality dataset into data/raw
	$(BIN)/python scripts/download_data.py

.PHONY: fmt
fmt: ## Apply formatting and import ordering
	$(BIN)/black src tests scripts examples
	$(BIN)/ruff check --fix src tests scripts examples

.PHONY: lint
lint: ## Run linters and formatting checks
	$(BIN)/ruff check src tests scripts examples
	$(BIN)/black --check src tests scripts examples

.PHONY: typecheck
typecheck: ## Run static type analysis
	$(BIN)/mypy

.PHONY: test
test: ## Run the test suite, skipping model training
	$(BIN)/pytest -m "not slow"

.PHONY: test-all
test-all: ## Run every test, including the ones that train models
	$(BIN)/pytest

.PHONY: coverage
coverage: ## Run tests with a coverage report
	$(BIN)/pytest -m "not slow" --cov=aqf --cov-report=term-missing

.PHONY: check
check: lint typecheck test ## Run every quality gate

.PHONY: experiment
experiment: ## Compare baselines and networks end to end
	$(BIN)/python scripts/run_experiment.py --config $(CONFIG)

.PHONY: smoke
smoke: ## Run the pipeline quickly to prove it works; the numbers are not results
	$(BIN)/python scripts/run_experiment.py --config $(FAST)

.PHONY: sweep
sweep: ## Repeat the comparison across seeds and report the spread
	$(BIN)/python scripts/run_seed_sweep.py --config $(CONFIG) --seeds $(SEEDS)

.PHONY: secrets-scan
secrets-scan: ## Look for credential-shaped strings in tracked files
	$(BIN)/python scripts/secrets_scan.py

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
