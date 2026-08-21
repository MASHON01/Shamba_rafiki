# Deployment Guide

Farm Pal is designed to run **offline** on a modest kiosk machine:
an 8 GB RAM target, with a hard budget of **7 GB** to stay safely below
the ceiling. This guide covers deploying the backend and llama-server
together.

## Architecture on the target

Two processes on the same machine:

```
  Browser / kiosk UI
        │ HTTP
        ▼
  Backend (FastAPI) ──HTTP──▶ llama-server (llama.cpp):8000:8080
        │
        ▼
  Local corpus + vector index (built ahead of time, offline)
```

The backend calls llama-server over local HTTP (`LLM_SERVER_URL`). They
are separate processes so the model stays resident and warm while the
backend restarts freely.

## One-time setup on the target machine

1. **Clone and bootstrap:**
   ```bash
   git clone <your-repo-url> RafikiAI && cd RafikiAI./scripts/setup_local.sh
   ```
   This builds llama.cpp **on the target machine**, so the binary matches
   its CPU.

2. **Place a GGUF model** in `models/` (e.g. `models/llama.gguf`). Use a
   quantization that fits the budget (Q4_K_M is a good default for the
   8 GB target).

3. **Build the corpus** (offline, once), two steps:
   ```bash
   # a) put your KALRO / AFA / KAMIS documents (PDF/DOCX/TXT) in:
   # data/raw_documents/
   # b) ingest them into the processed corpus, then index that corpus:
   python scripts/ingest.py --source KALRO # raw_documents -> processed_documents
   python scripts/build_index.py # processed_documents -> vector_store
   # (or in one go: make corpus)
   ```
   Ingest each provider in its own run so the `--source` provenance label
   is accurate (e.g. `--source AFA`, `--source KAMIS`). All of this happens
   at build time, never on a live request.

## Running

1. **Start llama-server:**
   ```bash./llama.cpp/build/bin/llama-server \
     -m models/llama.gguf --host 0.0.0.0 --port 8080 \
     -c 4096 -t 8
   ```

2. **Start the backend:**
   ```bash
   make run-prod
   # or: uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
   ```

3. **Verify:**
   ```bash
   curl localhost:8000/health # should report llm_available: true
   ```

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `LLM_SERVER_URL` | `http://localhost:8080` | Where llama-server listens |
| `MODEL_PATH` | `models/llama.gguf` | Resolved against the repo root |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Backend bind address |
| `APP_ENV` | `development` | Set to `production` on the kiosk |
| `DEBUG` | `true` | Set to `false` in production |
| `SIMILARITY_THRESHOLD` | `0.60` | Tune against the real embedder |

## Docker (optional)

The provided `Dockerfile` containerizes the **backend only** (llama-server
runs separately, as above). `docker-compose.yml` wires the backend and
mounts `data/`, `models/`, `logs/` from the host; the llama-server
service is included as a commented template.

```bash
make docker-build
make docker-up
```

## Memory budget (the 7 GB ceiling)

Exceeding the target machine's RAM risks an OOM kill, a total failure, so peak memory matters as much as latency. **Measure it with the
provided script, not by eye:** a transient spike during model load or a
large batch can last a fraction of a second and be missed by a task
manager. The script samples RSS continuously and checks the peak
against the 7 GB ceiling.

On the target machine, with llama-server running and the real model and
index loaded:

```bash
python -m backend.scripts.profile_memory # peak RAM vs 7 GB ceiling
python -m backend.scripts.run_benchmarks # latency table
```

These write reports to `data/reports/`. **These numbers only mean
something on the actual 8 GB target machine**, a development machine
with more RAM or different hardware will report figures that do not
reflect the kiosk. Do not use dev-machine or `--mock` numbers as the
real budget figure.

Approximate budget (fill in real numbers from the profiler on target):

| Component | Peak RSS |
|---|---|
| llama-server + GGUF model (Q4_K_M) | _measure_ |
| Embedding model (MiniLM) | _measure_ |
| Vector index + backend | _measure_ |
| **Total peak** | _must be < 7168 MB_ |

## Troubleshooting

- **`/health` shows `llm_available: false`**, llama-server isn't running
  or `LLM_SERVER_URL` is wrong. The backend still answers (degraded, no
  generation).
- **"Model directory does not exist" at startup**, create `models/`
  (setup_local.sh does this) or point `MODEL_PATH` at the right place.
- **Retrieval returns nothing**, the index hasn't been built; run
  `python scripts/build_index.py` after placing documents in
  `data/raw_documents/`.
- **Answers get over-caveated / replaced**, `SIMILARITY_THRESHOLD` may be
  too high for your real embedder; tune it down during corpus QA.