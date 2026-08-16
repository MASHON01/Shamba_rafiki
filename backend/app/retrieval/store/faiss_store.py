"""
FAISS vector store (opt-in backend).

A thin wrapper over a FAISS index for corpora large enough that
brute-force numpy would lag. For Phase 1's few-thousand-chunk corpus
numpy is the default; this is here so scaling up is a one-constant
change, not a rewrite.

Cosine similarity via FAISS: FAISS has no cosine metric directly, so
we L2-normalize vectors and use an inner-product index
(IndexFlatIP). Inner product of normalized vectors == cosine
similarity. We keep the index "flat" (exact, no approximation) so
results are identical to the numpy backend - correctness first;
approximate indexes can come later if ever needed.

faiss is imported lazily so this module stays importable when
faiss-cpu isn't installed (mirrors the embedder's lazy model load).

Satisfies `BaseVectorStore`.
"""

from __future__ import annotations

import numpy as np

from app.core.exceptions import InitializationError
from app.models.document import DocumentChunk, EmbeddedChunk, RetrievalResult
from app.retrieval.store.base import BaseVectorStore
from app.utils.logger import get_logger

logger = get_logger("FaissStore")

_L2_EPSILON = 1e-12


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, _L2_EPSILON)


def _load_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise InitializationError(
            "faiss-cpu is required for the FAISS vector store. Install it "
            "with: pip install faiss-cpu - or set VECTOR_STORE_BACKEND to "
            "'numpy' to use the dependency-free backend."
        ) from exc
    return faiss


class FaissStore(BaseVectorStore):
    """
    Exact inner-product FAISS index over L2-normalized vectors.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        faiss = _load_faiss()
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[DocumentChunk] = []

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        rows = []
        for ec in embedded_chunks:
            if len(ec.embedding) != self.dimension:
                raise ValueError(
                    f"Embedding for chunk {ec.chunk.chunk_id} has dimension "
                    f"{len(ec.embedding)}, expected {self.dimension}."
                )
            rows.append(ec.embedding)
            self._chunks.append(ec.chunk)

        block = _normalize_rows(np.asarray(rows, dtype=np.float32))
        self._index.add(block)

        logger.debug(
            "faiss_store.added",
            added=len(embedded_chunks),
            total=len(self._chunks),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        if len(self._chunks) == 0:
            return []

        if len(query_embedding) != self.dimension:
            raise ValueError(
                f"Query embedding has dimension {len(query_embedding)}, "
                f"expected {self.dimension}."
            )

        query = _normalize_rows(
            np.asarray([query_embedding], dtype=np.float32)
        )
        k = min(top_k, len(self._chunks))

        scores, indices = self._index.search(query, k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than k found
                continue
            score = float(score)
            if score_threshold is not None and score < score_threshold:
                continue
            results.append(
                RetrievalResult(
                    chunk=self._chunks[int(idx)],
                    similarity_score=score,
                )
            )
        return results

    def __len__(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Persistence hooks
    # ------------------------------------------------------------------

    def _export_state(self) -> dict:
        """
        Serialize the FAISS index to bytes and hand over the ordered
        chunks. persistence.py writes the bytes to the index file and
        the chunks into the shared mapping.
        """
        index_bytes = self._faiss.serialize_index(self._index)
        return {"index_bytes": index_bytes, "chunks": self._chunks}

    def _import_state(self, state: dict) -> None:
        chunks = state["chunks"]
        self._index = self._faiss.deserialize_index(state["index_bytes"])
        if self._index.ntotal != len(chunks):
            raise ValueError(
                f"FAISS index vector count ({self._index.ntotal}) does not "
                f"match chunk count ({len(chunks)}) on load."
            )
        self._chunks = chunks