"""
Retrieval engine.

The RAG indexing and retrieval half of Farm Pal. Reads the
corpus manifest produced by `app.ingestion`, embeds each chunk, and
builds a searchable vector index:

    Chunks -> Embeddings -> Vector Store -> Retriever

All embedding and indexing happens at BUILD time. The kiosk only
ever runs retrieval against the pre-built index - it never embeds a
corpus or trains anything at runtime, which is what keeps the 8GB
RAM budget intact.

Sub-packages:
    embeddings/  chunk text -> vectors (+ on-disk cache)
    store/       vectors -> persisted, searchable index
    search/      query -> ranked chunks

Top-level:
    Indexer      build-time: manifest -> embed -> store -> persist
    Retriever    runtime: query -> ranked chunks
"""

from __future__ import annotations

from app.retrieval.indexer import Indexer, IndexingError
from app.retrieval.search import Retriever

__all__ = [
    "Indexer",
    "IndexingError",
    "Retriever",
]