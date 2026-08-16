"""
Embedding layer: chunk text -> dense vectors.

    BaseEmbedder      the abstract interface (embed_texts / embed_query)
    Embedder          concrete multilingual MiniLM-class embedder
    EmbeddingCache    on-disk cache so unchanged chunks aren't re-embedded

The cache is the embedding-layer analogue of ingestion's
DuplicateDetector: it's what makes re-running the indexer over a
mostly-unchanged corpus cheap, which matters because the corpus is
collected incrementally across several days.
"""

from __future__ import annotations

from app.retrieval.embeddings.base import BaseEmbedder
from app.retrieval.embeddings.cache import EmbeddingCache
from app.retrieval.embeddings.embedder import Embedder

__all__ = [
    "BaseEmbedder",
    "Embedder",
    "EmbeddingCache",
]