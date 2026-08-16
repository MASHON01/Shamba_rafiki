"""
Abstract vector-store interface.

A vector store holds embedded chunks and answers nearest-neighbour
queries over them. Two concrete backends implement this:

    NumpyStore   brute-force cosine over a flat matrix (default)
    FaissStore   a FAISS index (opt-in, for larger corpora)

Both speak the same currency as the rest of the system:
    add()     takes EmbeddedChunk objects (chunk + its vector)
    search()  returns RetrievalResult objects (chunk + similarity)

so nothing downstream needs to know which backend is in use. The
retriever (later group) depends only on this interface.

Persistence is deliberately NOT defined here as save/load logic -
that lives in `persistence.py`, shared by both backends, because the
hard part (the id -> chunk mapping) is identical regardless of how
the vectors themselves are stored. Backends expose their raw state
via the small hooks at the bottom of this class for persistence to
serialize.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.document import EmbeddedChunk, RetrievalResult


class BaseVectorStore(ABC):
    """
    Base class for vector-store backends.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    @abstractmethod
    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """
        Add embedded chunks to the store. Vectors whose length doesn't
        match `self.dimension` must be rejected with a ValueError -
        a wrong-width vector silently corrupts every future search.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Return up to `top_k` chunks most similar to `query_embedding`,
        ordered by descending cosine similarity. Results below
        `score_threshold` (when given) are dropped.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Number of chunks currently stored."""
        raise NotImplementedError

    def is_empty(self) -> bool:
        return len(self) == 0

    # ------------------------------------------------------------------
    # Persistence hooks - implemented by backends, consumed by
    # persistence.py. Kept minimal: the shared id->chunk mapping is
    # handled by persistence itself; these only cover the vector data
    # each backend stores differently.
    # ------------------------------------------------------------------

    @abstractmethod
    def _export_state(self) -> dict:
        """
        Return everything persistence.py needs to reconstruct this
        backend's vectors later (e.g. the raw matrix, or a serialized
        FAISS index path). Chunk metadata is handled separately.
        """
        raise NotImplementedError

    @abstractmethod
    def _import_state(self, state: dict) -> None:
        """Restore vector data previously produced by `_export_state`."""
        raise NotImplementedError