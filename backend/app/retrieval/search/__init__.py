"""
Search layer: query -> ranked chunks.

    Retriever            the public retrieval interface (embed + search)
    similarity helpers   store-agnostic cosine scoring/ranking utilities

The Retriever is what the orchestrator calls; the similarity helpers
are for off-store scoring (re-ranking, de-duplication, diagnostics).
"""

from __future__ import annotations

from app.retrieval.search.retriever import Retriever
from app.retrieval.search.similarity import (
    apply_threshold,
    cosine_similarity,
    rank_by_similarity,
    top_k,
)

__all__ = [
    "Retriever",
    "cosine_similarity",
    "rank_by_similarity",
    "apply_threshold",
    "top_k",
]