# Farm Pal backend - Python API only (option B).
#
# This image contains the FastAPI backend ONLY. The LLM (llama.cpp /
# llama-server) runs as a SEPARATE process/service, exactly as it does
# on the kiosk: the backend calls it over HTTP via LLM_SERVER_URL. That
# keeps this image small and avoids compiling C++ inside it. See
# docker-compose.yml and docs/DEPLOYMENT.md for how the two connect.

FROM python:3.12-slim AS base

# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image
# - PYTHONUNBUFFERED: logs stream out immediately (structlog to stdout)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/backend

WORKDIR /app

# System deps: build-essential only where a wheel isn't available.
# Kept minimal; the heavy RAG/vision extras are optional and installed
# on demand, not baked into the base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching): copy only the
# dependency manifests before the source.
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Now the application source.
COPY backend/ ./backend/

# Runtime dirs the app expects (paths.py creates them too, but making
# them here means the image is ready even before first boot).
RUN mkdir -p data/raw_documents data/processed_documents \
    data/vector_store data/embeddings data/reports models logs

EXPOSE 8000

# Healthcheck hits the deep /health endpoint (reports LLM reachability).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Run from repo root; app is import-anchored so CWD doesn't matter, but
# root is the canonical launch point.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]