# Phase 1 — Validation Checklist

Sign-off for the Phase 1 backend. Items are grouped by what can be
verified now (offline, on fakes) versus what requires the real model and
corpus on the target machine.

## ✅ Verified now (offline, in CI)

Code & structure
- [x] Entire `app/` package imports cleanly
- [x] All filesystem paths anchored to the repo root (CWD-independent)
- [x] `uvicorn app.main:app` boots from any working directory
- [x] `ruff` and `black` pass across the codebase
- [x] No duplicated / dead configuration (constants deduplicated)

Pipeline (on fake embedder + LLM)
- [x] Ingestion: PDF/DOCX/text → cleaned, chunked, metadata-tagged corpus
- [x] Retrieval: query → ranked chunks; NumPy and FAISS backends agree
- [x] Language: EN/SW detection, entity + intent extraction, SW→EN glossing
- [x] Prompt assembly: system prompt + numbered sources + history
- [x] Verification: evidence / citation / semantic / hallucination checks
- [x] Verification catches a fabricated dose and caveats/replaces it
- [x] Orchestrator: full Request→…→verified Response end to end

API
- [x] All endpoints (`/`, `/health`, `/version`, `/chat`, `/retrieve`, `/upload`)
- [x] Standard success/error envelope, request-ID tracing
- [x] Validation errors return 422; LLM-down degrades cleanly
- [x] OpenAPI docs generated

Tests & tooling
- [x] Full test suite passes (`pytest` from repo root) — 98 tests
- [x] Marker tiers work (`-m unit` / `integration` / `api`)
- [x] Makefile targets, pre-commit hooks, GitHub Actions CI
- [x] Dockerfile (backend) + docker-compose build

## ⏳ Requires the real model + corpus (on the target machine)

These cannot be validated on fakes or a dev machine — they need the
real GGUF model in llama-server, the real embedder, the real corpus,
and the 8 GB-spec target machine.

Corpus & retrieval quality
- [ ] Real KALRO / AFA / KAMIS documents ingested into `data/raw_documents/`
- [ ] Index built (`scripts/build_index.py`) from the real corpus
- [ ] Retrieval quality QA'd on real queries (English and Swahili)
- [ ] `SIMILARITY_THRESHOLD` tuned against the real MiniLM embedder

Model & generation
- [ ] GGUF model selected and placed in `models/` (fits the budget)
- [ ] llama-server serving; `/health` reports `llm_available: true`
- [ ] End-to-end `/chat` produces accurate, grounded answers
- [ ] Answer accuracy checked on the evaluation prompts (EN primary, SW bonus)

Performance & budget (the rubric numbers)
- [ ] Peak RAM measured on target — **must be < 7168 MB** (`profile_memory`)
- [ ] Latency measured on target (`run_benchmarks`) — p50 / p95 recorded
- [ ] Numbers recorded in REPORT.md (from the target machine, not dev/mock)

Reproducibility
- [ ] Fresh clone + `./scripts/setup_local.sh` reaches a runnable state
- [ ] A second person reproduces the setup from the docs alone

## Not in Phase 1 (later phases)

- [ ] Phase 2: real LLM integration hardening
- [ ] Phase 3: image classifier (cross-disciplinary integration / bonus)
- [ ] Frontend / kiosk UI

## Sign-off

Phase 1 delivers a complete, tested, HTTP-exposed offline advisory
backend. The engine is done; the remaining items above are **data,
model, and on-target measurement** — not code — and are the immediate
next steps before submission.