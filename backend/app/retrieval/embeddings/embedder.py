"""
Concrete multilingual sentence embedder.

Wraps a MiniLM-class multilingual model
(paraphrase-multilingual-MiniLM-L12-v2 by default) so Swahili
queries retrieve against English source documents - the core reason
the product needs a multilingual embedder rather than an
English-only one.

Two things worth understanding here:

1. The heavy dependency (sentence-transformers + its torch backend)
   is imported lazily, inside `_ensure_model()`, NOT at module import
   time. So this module - and everything that imports it - stays
   importable on a machine without the model installed. The model
   only materializes the first time you actually embed something.
   This mirrors how the ingestion loaders lazy-import fitz/docx.

2. The embedding cache is consulted per-text on the corpus path, so
   re-indexing a mostly-unchanged corpus only encodes genuinely new
   chunks. Query embedding is never cached (queries are one-off).

Satisfies `BaseEmbedder`.
"""

from __future__ import annotations

from app.config.constants import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
)
from app.config.settings import settings
from app.core.exceptions import InitializationError
from app.retrieval.embeddings.base import BaseEmbedder
from app.retrieval.embeddings.cache import EmbeddingCache
from app.utils.logger import get_logger

logger = get_logger("Embedder")


class Embedder(BaseEmbedder):
    """
    Multilingual sentence embedder with an optional on-disk cache.
    """

    def __init__(
        self,
        model_id: str | None = None,
        dimension: int = EMBEDDING_DIMENSION,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        cache: EmbeddingCache | None = None,
        use_cache: bool = True,
    ) -> None:
        self._model_id = model_id or settings.EMBEDDING_MODEL
        self.dimension = dimension
        self._batch_size = batch_size
        self._model = None  # lazily loaded

        if use_cache:
            self._cache = cache or EmbeddingCache(
                model_id=self._model_id, dimension=dimension
            )
        else:
            self._cache = None

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of corpus texts, using the cache where possible
        and encoding only the misses. Order of the returned vectors
        matches the input order.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        to_encode: list[str] = []
        to_encode_positions: list[int] = []

        # 1. Serve what we can from cache.
        for i, text in enumerate(texts):
            cached = self._cache.get(text) if self._cache else None
            if cached is not None:
                results[i] = cached
            else:
                to_encode.append(text)
                to_encode_positions.append(i)

        # 2. Encode the misses (if any) and populate the cache.
        if to_encode:
            encoded = self._encode(to_encode)
            for position, text, vector in zip(
                to_encode_positions, to_encode, encoded
            ):
                results[position] = vector
                if self._cache:
                    self._cache.set(text, vector)

        # Every slot is filled by construction.
        return [vector for vector in results if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query. Not cached - queries are one-off.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed an empty query.")
        return self._encode([text])[0]

    # ------------------------------------------------------------------
    # Model handling (lazy)
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """
        Load the sentence-transformers model on first use. Raises a
        clear InitializationError if the dependency or the model
        itself isn't available, rather than a bare ImportError.
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise InitializationError(
                "sentence-transformers is required to compute embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        try:
            logger.info("embedder.loading_model", model_id=self._model_id)
            self._model = SentenceTransformer(self._model_id)
        except Exception as exc:
            raise InitializationError(
                f"Could not load embedding model '{self._model_id}': {exc}"
            ) from exc

        self._verify_dimension()
        logger.info(
            "embedder.model_ready",
            model_id=self._model_id,
            dimension=self.dimension,
        )

    def _verify_dimension(self) -> None:
        """
        Confirm the loaded model's real output width matches the
        configured `dimension` - a mismatch would silently corrupt
        the vector store, so fail loudly at load time instead.
        """
        try:
            reported = self._model.get_sentence_embedding_dimension()
        except Exception:  # pragma: no cover - model without the helper
            return
        if reported is not None and reported != self.dimension:
            raise InitializationError(
                f"Model '{self._model_id}' produces {reported}-dim vectors "
                f"but EMBEDDING_DIMENSION is {self.dimension}. Update the "
                f"constant to match the model."
            )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """
        Run the model over `texts` in batches and return plain
        list[float] vectors (JSON/Pydantic-friendly, matching
        EmbeddedChunk.embedding).
        """
        self._ensure_model()

        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]