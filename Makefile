# Farm Pal - developer command runner.
# All targets run from the repository root. Run `make help` for the list.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Use the venv's tools if present, else fall back to system.
PY := python3
PIP := $(PY) -m pip

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- setup -------------------------------------------------------------

.PHONY: install
install: ## Install runtime + dev dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

.PHONY: setup
setup: ## Full local bootstrap (venv, deps, dirs, llama.cpp build)
	bash scripts/setup_local.sh

.PHONY: hooks
hooks: ## Install pre-commit hooks
	$(PY) -m pre_commit install

# --- quality -----------------------------------------------------------

.PHONY: format
format: ## Auto-format with black + ruff --fix
	$(PY) -m black backend
	$(PY) -m ruff check --fix backend

.PHONY: lint
lint: ## Lint with ruff (no changes)
	$(PY) -m ruff check backend

.PHONY: typecheck
typecheck: ## Type-check the app with mypy
	$(PY) -m mypy backend/app

.PHONY: check
check: lint typecheck ## Run all static checks (lint + types)

# --- tests -------------------------------------------------------------

.PHONY: test
test: ## Run the full test suite (all 98, on fakes)
	$(PY) -m pytest

.PHONY: test-unit
test-unit: ## Run only the fast unit tests
	$(PY) -m pytest -m unit

.PHONY: test-cov
test-cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov=backend/app --cov-report=term-missing

# --- run ---------------------------------------------------------------

.PHONY: run
run: ## Run the API (dev, autoreload) - needs llama-server separately
	$(PY) -m uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000

.PHONY: run-prod
run-prod: ## Run the API (production settings)
	APP_ENV=production DEBUG=false $(PY) -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000

# --- data / model ------------------------------------------------------

.PHONY: ingest
ingest: ## Ingest raw documents (data/raw_documents) into the processed corpus
	$(PY) scripts/ingest.py

.PHONY: build-index
build-index: ## Build the vector index from the processed corpus
	$(PY) scripts/build_index.py

.PHONY: probe
probe: ## Print raw similarity scores for a query (tune SIMILARITY_THRESHOLD). Usage: make probe Q="maize brown spots"
	$(PY) scripts/probe_retrieval.py "$(Q)"

.PHONY: corpus
corpus: ingest build-index ## Full corpus build: ingest raw docs, then index them

.PHONY: build-llama
build-llama: ## Clone + build llama.cpp on this machine (into ./llama.cpp)
	bash scripts/setup_local.sh --llama-only

# --- benchmarks (run on the TARGET machine for real numbers) -----------

.PHONY: benchmark
benchmark: ## Run the benchmark suite (real components; needs llama-server)
	$(PY) -m backend.scripts.run_benchmarks

.PHONY: profile-memory
profile-memory: ## Measure peak RAM of a full query vs the 7GB ceiling
	$(PY) -m backend.scripts.profile_memory

# --- docker ------------------------------------------------------------

.PHONY: docker-build
docker-build: ## Build the backend Docker image
	docker build -t shamba-rafiki-backend:latest .

.PHONY: docker-up
docker-up: ## Start via docker-compose
	docker compose up --build

# --- housekeeping ------------------------------------------------------

.PHONY: clean
clean: ## Remove caches + build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info