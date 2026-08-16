"""
Vector-store persistence.

Saves and loads a vector store to/from disk. This is the one place
that understands the on-disk layout, so neither backend has to.

Why this is its own module: a raw FAISS index (or numpy matrix)
stores only vectors and their row order. The information that makes a
search *result* meaningful - which chunk, from which document, with
what metadata, each row corresponds to - is not in the index at all.
That id -> chunk mapping is identical regardless of backend, so it's
written here once, alongside whatever vector data the backend hands
over via its `_export_state()` hook.

On-disk layout (under a store directory):

    store_meta.json   backend, dimension, count, metric - validated on load
    mapping.json      row index -> serialized DocumentChunk (+ metadata)
    vectors.npy       raw matrix            (numpy backend only)
    index.faiss       serialized FAISS index (faiss backend only)

Load validates meta before trusting the index: a dimension or
backend mismatch raises rather than silently returning garbage
results.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import numpy as np

from app.config.constants import (
    VECTOR_INDEX_FILENAME,
    VECTOR_MAPPING_FILENAME,
    VECTOR_STORE_META_FILENAME,
    VECTOR_VECTORS_FILENAME,
)
from app.config.paths import VECTOR_STORE_DIR
from app.core.exceptions import ShambaRafikiError
from app.models.document import DocumentChunk
from app.retrieval.store.base import BaseVectorStore
from app.utils.logger import get_logger

logger = get_logger("VectorStorePersistence")


class VectorStorePersistenceError(ShambaRafikiError):
    """Raised when a vector store can't be saved or loaded."""


def _chunk_to_dict(chunk: DocumentChunk) -> dict:
    return {
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(chunk.document_id),
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(data: dict) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=UUID(data["chunk_id"]),
        document_id=UUID(data["document_id"]),
        chunk_index=data["chunk_index"],
        text=data["text"],
        token_count=data["token_count"],
        metadata=data.get("metadata", {}),
    )


def save_store(
    store: BaseVectorStore,
    backend: str,
    store_dir: Path | None = None,
) -> Path:
    """
    Persist `store` to `store_dir`. Returns the directory written to.
    """
    store_dir = store_dir or VECTOR_STORE_DIR

    try:
        store_dir.mkdir(parents=True, exist_ok=True)
        state = store._export_state()
        chunks = state["chunks"]

        # 1. Shared id -> chunk mapping (both backends).
        mapping = {
            "chunks": [_chunk_to_dict(c) for c in chunks],
        }
        (store_dir / VECTOR_MAPPING_FILENAME).write_text(
            json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
        )

        # 2. Backend-specific vector data.
        if "matrix" in state:  # numpy backend
            np.save(store_dir / VECTOR_VECTORS_FILENAME, state["matrix"])
        elif "index_bytes" in state:  # faiss backend
            (store_dir / VECTOR_INDEX_FILENAME).write_bytes(
                bytes(state["index_bytes"])
            )
        else:
            raise VectorStorePersistenceError(
                "Backend exported neither 'matrix' nor 'index_bytes'."
            )

        # 3. Store meta for validation on load.
        meta = {
            "backend": backend,
            "dimension": store.dimension,
            "count": len(chunks),
            "metric": "cosine",
        }
        (store_dir / VECTOR_STORE_META_FILENAME).write_text(
            json.dumps(meta), encoding="utf-8"
        )

    except OSError as exc:
        raise VectorStorePersistenceError(
            f"Could not save vector store to '{store_dir}': {exc}"
        ) from exc

    logger.info(
        "vector_store.saved",
        backend=backend,
        count=len(chunks),
        store_dir=str(store_dir),
    )
    return store_dir


def load_store(
    store: BaseVectorStore,
    backend: str,
    store_dir: Path | None = None,
) -> BaseVectorStore:
    """
    Load persisted state into `store` (a freshly-constructed backend
    of the matching type and dimension). Returns the same store,
    populated.
    """
    store_dir = store_dir or VECTOR_STORE_DIR

    meta_path = store_dir / VECTOR_STORE_META_FILENAME
    mapping_path = store_dir / VECTOR_MAPPING_FILENAME

    if not meta_path.exists() or not mapping_path.exists():
        raise VectorStorePersistenceError(
            f"No saved vector store found in '{store_dir}' "
            f"(missing meta or mapping file)."
        )

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorStorePersistenceError(
            f"Could not read store metadata: {exc}"
        ) from exc

    # Validate before trusting the vectors.
    if meta.get("backend") != backend:
        raise VectorStorePersistenceError(
            f"Saved store backend '{meta.get('backend')}' does not match "
            f"requested backend '{backend}'."
        )
    if meta.get("dimension") != store.dimension:
        raise VectorStorePersistenceError(
            f"Saved store dimension {meta.get('dimension')} does not match "
            f"store dimension {store.dimension}."
        )

    try:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        chunks = [_chunk_from_dict(d) for d in mapping["chunks"]]

        if "matrix" in _expected_state_keys(backend):
            matrix = np.load(store_dir / VECTOR_VECTORS_FILENAME)
            store._import_state({"matrix": matrix, "chunks": chunks})
        else:
            index_bytes = (store_dir / VECTOR_INDEX_FILENAME).read_bytes()
            store._import_state(
                {"index_bytes": np.frombuffer(index_bytes, dtype=np.uint8),
                 "chunks": chunks}
            )

    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise VectorStorePersistenceError(
            f"Could not load vector store from '{store_dir}': {exc}"
        ) from exc

    logger.info(
        "vector_store.loaded",
        backend=backend,
        count=len(chunks),
        store_dir=str(store_dir),
    )
    return store


def _expected_state_keys(backend: str) -> set[str]:
    """Which vector-data key a backend round-trips through persistence."""
    return {"matrix"} if backend == "numpy" else {"index_bytes"}