# Developer Guide

## Requirements

- Python 3.12+
- git, cmake, a C++ toolchain (build-essential), for building llama.cpp
- ~2, 3 GB disk for the model + dependencies

## Quick start

From the repository root:

```bash./scripts/setup_local.sh # venv, deps, dirs, build llama.cpp, run tests
```

This creates a `.venv`, installs dependencies, creates runtime
directories, clones and builds llama.cpp on this machine, and runs the
full test suite as a smoke check.

To skip the (slow) llama.cpp build while developing:
```bash./scripts/setup_local.sh --no-llama
```

## Common commands (Makefile)

All run from the repository root:

| Command | What it does |
|---|---|
| `make install` | Install runtime + dev dependencies |
| `make setup` | Full bootstrap (calls setup_local.sh) |
| `make hooks` | Install pre-commit hooks |
| `make test` | Run the full test suite (all on fakes) |
| `make test-unit` | Run only the fast unit tests |
| `make test-cov` | Tests with a coverage report |
| `make lint` | Lint with ruff |
| `make format` | Auto-format with black + ruff --fix |
| `make typecheck` | Type-check with mypy |
| `make check` | lint + typecheck |
| `make run` | Run the API (dev, autoreload) |
| `make build-index` | Build the vector index from the corpus |
| `make build-llama` | Clone + build llama.cpp |
| `make benchmark` | Run the benchmark suite (needs llama-server) |
| `make profile-memory` | Measure peak RAM vs the 7 GB ceiling |

## Running the backend

The backend and the LLM run as **separate** processes.

1. Start llama-server with a GGUF model:
   ```bash./llama.cpp/build/bin/llama-server \
     -m models/llama.gguf --host 0.0.0.0 --port 8080 -c 4096 -t 8
   ```
2. Start the backend (in another terminal):
   ```bash
   make run
   ```
3. Ask a question:
   ```bash
   curl -s localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"query":"How do I treat maize blight?","language":"en"}'
   ```

The backend runs fine without llama-server, it degrades cleanly and
`/health` reports `llm_available: false`. Retrieval, language analysis,
and verification all work; only generation needs the model.

## Testing

Tests use a deterministic fake embedder and fake LLM (in
`backend/tests/conftest.py`), so the whole suite runs offline in
seconds with no model download. From the repo root:

```bash
pytest # all 98 tests
pytest -m unit # fast unit tests only
pytest -m "not slow" # skip corpus-building tests
```

Test tiers: `unit` (isolated components), `integration` (ingest→index→
retrieve and the full orchestrator), `api` (endpoints via TestClient).

## Project layout

```
RafikiAI/
├── backend/
│ ├── app/ # the application (see ARCHITECTURE.md)
│ ├── tests/ # unit / integration / api
│ └── scripts/ # run_benchmarks.py, profile_memory.py
├── scripts/ # build_index.py, setup_local.sh
├── data/ # corpus, index, cache (gitignored)
├── models/ # GGUF / ONNX weights (gitignored)
├── llama.cpp/ # inference engine (gitignored, built locally)
└── docs/
```

## Code quality

`ruff` (lint), `black` (format), and `mypy` (types) are configured in
`pyproject.toml` and enforced by pre-commit hooks and CI. Before
committing:

```bash
make format && make check && make test
```

Paths are anchored to the repo root (`app/config/paths.py`), so
everything works regardless of the directory you launch from.