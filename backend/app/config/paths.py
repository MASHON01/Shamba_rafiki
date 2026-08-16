"""
Canonical filesystem paths, anchored to the repository root.

Every path here is absolute and derived from this file's own
location, so they resolve identically no matter which directory the
app, tests, or scripts are launched from. This is what lets
`uvicorn app.main:app`, the CLI scripts, Docker, and a fresh-checkout
reproducer all agree on where models, data, and artifacts live.

Layout (repo root = RafikiAI/):

    RafikiAI/                 <- ROOT_DIR
      backend/                <- BACKEND_DIR (the Python package root)
        app/                  <- APP_DIR
      data/                   <- DATA_DIR (corpus, index, cache)
      models/                 <- MODELS_DIR (GGUF weights, ONNX classifier)
      logs/                   <- LOGS_DIR
"""

from pathlib import Path

# This file is backend/app/config/paths.py:
#   parents[0] = config, [1] = app, [2] = backend, [3] = repo root.
ROOT_DIR = Path(__file__).resolve().parents[3]

BACKEND_DIR = ROOT_DIR / "backend"

# The static kiosk UI, served by FastAPI at /app.
FRONTEND_DIR = ROOT_DIR / "frontend"
APP_DIR = BACKEND_DIR / "app"
CONFIG_DIR = APP_DIR / "config"

DATA_DIR = ROOT_DIR / "data"
RAW_DOCUMENTS_DIR = DATA_DIR / "raw_documents"
PROCESSED_DOCUMENTS_DIR = DATA_DIR / "processed_documents"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
REPORTS_DIR = DATA_DIR / "reports"

MODELS_DIR = ROOT_DIR / "models"

LOGS_DIR = ROOT_DIR / "logs"
TESTS_DIR = BACKEND_DIR / "tests"
SCRIPTS_DIR = BACKEND_DIR / "scripts"


DIRECTORIES = [
    DATA_DIR,
    RAW_DOCUMENTS_DIR,
    PROCESSED_DOCUMENTS_DIR,
    VECTOR_STORE_DIR,
    EMBEDDINGS_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    LOGS_DIR,
]


def create_directories() -> None:
    """
    Create all required directories if they do not exist.

    TESTS_DIR and SCRIPTS_DIR are intentionally not created here - they
    are source directories that must already exist, not runtime
    artifacts.
    """
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)