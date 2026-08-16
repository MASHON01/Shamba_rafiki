# Architecture

Shamba Rafiki is an **offline** agricultural advisory assistant for
smallholder farmers in Kenya. A farmer asks a question in English or
Swahili; the system retrieves grounding material from a local corpus,
generates an answer with a local LLM, verifies that answer against the
sources, and returns it with a confidence signal, all with no
internet and no per-query cost.

## Request lifecycle

A `POST /chat` request flows through the pipeline below. Every stage is
a separate, independently-testable component.

```
Farmer question
      │
      ▼
  Language layer detect language, normalize, extract intent +
  (app/language) entities, gloss Swahili domain words → English
      │
      ▼
  Retrieval embed the query, search the local vector index,
  (app/retrieval) return the most relevant corpus chunks
      │
      ▼
  Prompt assembly system prompt (per language + intent) +
  (app/orchestration numbered source context + conversation history
   /prompts) + the question
      │
      ▼
  LLM generation llama.cpp / llama-server over local HTTP
  (app/orchestration
   /llm)
      │
      ▼
  Verification score evidence, check citations, detect
  (app/verification) unsupported specifics → approve / caveat /
                        replace, with a confidence level
      │
      ▼
  Standardized response answer + sources + confidence + metadata
```

The orchestrator (`app/orchestration/orchestrator.py`) owns this
lifecycle end to end and is the single entry point the API layer calls.

## Layers

**Language intelligence** (`app/language/`), detector, normalizer,
entity/intent extractors, a bilingual terminology map, and a
dictionary-fallback translator. Turns a raw Swahili or English question
into a normalized, English-anchored retrieval query plus a detected
language and intent.

**Retrieval / RAG** (`app/retrieval/`), a multilingual MiniLM sentence
embedder (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`,
384-dim), a content-addressed embedding cache, and a vector store with
two interchangeable backends (a NumPy brute-force cosine search by
default, or FAISS). The corpus is embedded **at build time**, so the
kiosk only ever does retrieval, never indexing.

**Ingestion** (`app/ingestion/`), the build-time pipeline that turns
KALRO / AFA / KAMIS documents (PDF, DOCX, text) into clean, chunked,
metadata-tagged corpus records. Runs offline via `scripts/build_index.py`.

**Orchestration** (`app/orchestration/`), prompt assembly (system
prompts, context builder, prompt builder), the llama-server client,
per-session conversation memory, a request router, a dispatcher, and
the top-level orchestrator.

**Verification** (`app/verification/`), an evidence scorer, citation
checker, semantic validator, and hallucination detector, combined by a
confidence engine and a conservative decision policy. This is the
runtime half of hallucination control; grounding-first system prompts
are the generation-side half.

**API** (`app/api/`), a FastAPI app exposing the pipeline over HTTP,
with request-ID middleware, standardized error handling, and dependency
injection that builds the orchestrator and retriever once at startup.

**Configuration & core** (`app/config/`, `app/core/`), settings
(env-overridable), repo-root-anchored paths, application constants,
structured logging, the exception hierarchy, and the standard response
envelope.

**Profiling** (`app/profiling/`), latency and RAM measurement,
a benchmark suite, and report generation. These produce the numbers for
the efficiency score and the memory budget (see DEPLOYMENT.md).

## Design principles

**Grounded and honest.** The system answers from retrieved material and
says plainly when it is uncertain, rather than fabricating. A specific
invented claim (e.g. a pesticide dose with no source) is caught by the
hallucination detector and either caveated or replaced with a safe
fallback.

**Offline and cheap.** No cloud, no per-query cost. The corpus is built
and embedded ahead of time; the kiosk only loads a small model, a small
index, and answers.

**Memory-budget first.** The whole system is designed to run well under
a 7 GB ceiling on an 8 GB machine (see the memory budget in
DEPLOYMENT.md), leaving deliberate headroom.

**CWD-independent.** All filesystem paths are anchored to the repository
root (`app/config/paths.py`), so the app, tests, scripts, and container
all resolve the same locations regardless of the working directory.

## Technology

- **Inference:** llama.cpp / llama-server, GGUF weights (Q4_K_M)
- **Backend:** FastAPI (Python 3.12)
- **Retrieval:** NumPy or FAISS + multilingual MiniLM embeddings
- **Documents:** PyMuPDF (PDF), python-docx (DOCX)
- **Logging:** structlog (structured, JSON-capable)