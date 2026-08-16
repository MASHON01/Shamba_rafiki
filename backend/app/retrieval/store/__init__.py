"""
Vector store: persisted, searchable embeddings.

    BaseVectorStore   the interface (add / search / persist hooks)
    NumpyStore        brute-force cosine over a flat matrix (default)
    FaissStore        exact inner-product FAISS index (opt-in)
    save_store/load_store   shared on-disk persistence, incl. the
                            critical id -> chunk mapping

Use `create_vector_store()` rather than importing a backend directly,
so switching backends is the single `VECTOR_STORE_BACKEND` constant
and never a code change at the call site.
"""

from __future__ import annotations

from app.config.constants import VECTOR_STORE_BACKEND
from app.core.exceptions import InitializationError
from app.retrieval.store.base import BaseVectorStore
from app.retrieval.store.numpy_store import NumpyStore
from app.retrieval.store.persistence import (
    VectorStorePersistenceError,
    load_store,
    save_store,
)

__all__ = [
    "BaseVectorStore",
    "NumpyStore",
    "FaissStore",
    "create_vector_store",
    "save_store",
    "load_store",
    "VectorStorePersistenceError",
]


def __getattr__(name: str):
    """
    Lazily expose FaissStore so importing this package doesn't require
    faiss-cpu to be installed unless FAISS is actually used.
    """
    if name == "FaissStore":
        from app.retrieval.store.faiss_store import FaissStore

        return FaissStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_vector_store(
    dimension: int,
    backend: str | None = None,
) -> BaseVectorStore:
    """
    Construct the configured vector-store backend.

    `backend` overrides the VECTOR_STORE_BACKEND constant when given
    (useful in tests). Unknown backends raise InitializationError.
    """
    backend = (backend or VECTOR_STORE_BACKEND).lower()

    if backend == "numpy":
        return NumpyStore(dimension)
    if backend == "faiss":
        from app.retrieval.store.faiss_store import FaissStore

        return FaissStore(dimension)

    raise InitializationError(
        f"Unknown vector store backend '{backend}'. "
        f"Use 'numpy' or 'faiss'."
    )