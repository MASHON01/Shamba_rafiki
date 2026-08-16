"""
Brute-force numpy vector store (default backend).

Holds all embeddings in a single float32 matrix and answers queries
by computing cosine similarity against every row. For a corpus of a
few thousand chunks this is effectively instant, adds negligible
RAM, needs no native dependencies beyond numpy, and can't fail to
install on the 8GB target machine - which is exactly why it's the
default per the product design.

Cosine similarity is implemented as a dot product over
L2-normalized vectors: normalize each stored vector once at add
time, normalize the query at search time, and a single matrix-vector
product yields all similarities at once.

Satisfies `BaseVectorStore`.
"""

from __future__ import annotations

import numpy as np

from app.models.document import DocumentChunk, EmbeddedChunk, RetrievalResult
from app.retrieval.store.base import BaseVectorStore
from app.utils.logger import get_logger

logger = get_logger("NumpyStore")

_L2_EPSILON = 1e-12  # guards against divide-by-zero on a zero vector


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows stay zero (cosine 0 with anything)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, _L2_EPSILON)


class NumpyStore(BaseVectorStore):
    """
    Flat-matrix cosine store.
    """

    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        # Normalized vectors, shape (n, dimension), float32.
        self._matrix = np.empty((0, dimension), dtype=np.float32)
        # Parallel list of the chunks behind each row.
        self._chunks: list[DocumentChunk] = []

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        new_rows = []
        for ec in embedded_chunks:
            if len(ec.embedding) != self.dimension:
                raise ValueError(
                    f"Embedding for chunk {ec.chunk.chunk_id} has dimension "
                    f"{len(ec.embedding)}, expected {self.dimension}."
                )
            new_rows.append(ec.embedding)
            self._chunks.append(ec.chunk)

        block = _normalize_rows(
            np.asarray(new_rows, dtype=np.float32)
        )
        self._matrix = np.vstack([self._matrix, block])

        logger.debug(
            "numpy_store.added",
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
        )[0]

        # Cosine similarity for every row in one matrix-vector product.
        scores = self._matrix @ query  # shape (n,)

        k = min(top_k, len(self._chunks))
        # argpartition for the top-k, then sort just those k descending.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        results: list[RetrievalResult] = []
        for idx in top_idx:
            score = float(scores[idx])
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
        Hand persistence.py the raw matrix and the ordered chunks.
        persistence writes the matrix as .npy and the chunks into the
        shared mapping file.
        """
        return {"matrix": self._matrix, "chunks": self._chunks}

    def _import_state(self, state: dict) -> None:
        matrix = state["matrix"]
        chunks = state["chunks"]
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"Vector count ({matrix.shape[0]}) does not match chunk "
                f"count ({len(chunks)}) on load."
            )
        self._matrix = matrix.astype(np.float32)
        self._chunks = chunks