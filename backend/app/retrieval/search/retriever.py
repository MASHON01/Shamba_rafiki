"""
High-level retrieval interface.

The public face of the retrieval engine and the single object the
orchestrator (Output 6) will call. It ties the two halves together:

    query text --[embedder.embed_query]--> query vector
              --[store.search]-----------> ranked RetrievalResult[]

Everything below the Retriever (which embedder, which store backend,
how vectors are persisted) is hidden behind it. A caller only needs:

    retriever = Retriever.from_disk()      # load a pre-built index
    hits = retriever.retrieve("nyanya zina ugonjwa gani?")

Retrieval is READ-ONLY and runtime-cheap: the index is built offline
by the indexer, loaded once here, and queried. The kiosk never
embeds a corpus or builds an index.
"""

from __future__ import annotations

from pathlib import Path

from app.config.constants import VECTOR_STORE_BACKEND
from app.config.settings import settings
from app.models.document import RetrievalResult
from app.retrieval.embeddings.base import BaseEmbedder
from app.retrieval.embeddings.embedder import Embedder
from app.retrieval.store import (
    BaseVectorStore,
    create_vector_store,
    load_store,
)
from app.utils.logger import get_logger

logger = get_logger("Retriever")


class Retriever:
    """
    Embeds queries and searches the vector store for relevant chunks.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        default_top_k: int | None = None,
        default_threshold: float | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._default_top_k = default_top_k or settings.TOP_K
        # None is a valid explicit choice, so distinguish "not given".
        self._default_threshold = (
            settings.SIMILARITY_THRESHOLD
            if default_threshold is None
            else default_threshold
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Return the chunks most relevant to `query`, most similar
        first. An empty/whitespace query, or an empty index, yields an
        empty list rather than an error.
        """
        if not query or not query.strip():
            logger.debug("retriever.empty_query")
            return []

        if self._store.is_empty():
            logger.warning("retriever.empty_store")
            return []

        k = top_k or self._default_top_k
        threshold = (
            self._default_threshold
            if score_threshold is None
            else score_threshold
        )

        query_vector = self._embedder.embed_query(query)
        results = self._store.search(
            query_vector, top_k=k, score_threshold=threshold
        )

        logger.info(
            "retriever.retrieved",
            query_chars=len(query),
            hits=len(results),
            top_k=k,
            threshold=threshold,
        )
        return results

    def __len__(self) -> int:
        """Number of chunks in the underlying index."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_disk(
        cls,
        store_dir: Path | None = None,
        backend: str | None = None,
        embedder: BaseEmbedder | None = None,
        default_top_k: int | None = None,
        default_threshold: float | None = None,
    ) -> "Retriever":
        """
        Load a pre-built index from disk and wrap it in a Retriever.

        `embedder` defaults to the standard multilingual Embedder; its
        `dimension` must match the persisted store (validated by
        `load_store`). Backend defaults to VECTOR_STORE_BACKEND.
        """
        backend = backend or VECTOR_STORE_BACKEND
        embedder = embedder or Embedder()

        empty_store = create_vector_store(embedder.dimension, backend=backend)
        store = load_store(empty_store, backend=backend, store_dir=store_dir)

        logger.info(
            "retriever.loaded_from_disk",
            backend=backend,
            chunks=len(store),
        )
        return cls(
            embedder=embedder,
            store=store,
            default_top_k=default_top_k,
            default_threshold=default_threshold,
        )