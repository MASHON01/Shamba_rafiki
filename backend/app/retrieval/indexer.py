"""
Indexing orchestrator - manifest -> embed -> store -> persist.

The build-time counterpart to `app.ingestion.pipeline`. Where the
ingestion pipeline turns raw documents into a persisted corpus, the
indexer turns that persisted corpus into a persisted, searchable
vector index:

    manifest.json
        -> read each document record's chunks
        -> embed every chunk (via the cached Embedder)
        -> add to the vector store
        -> persist store (vectors + id->chunk mapping) to disk

Run once at build time (Day 8 in the plan), never on the kiosk. The
resulting index is what `Retriever.from_disk()` loads at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from app.config.constants import (
    EMBEDDING_BATCH_SIZE,
    VECTOR_STORE_BACKEND,
)
from app.config.paths import PROCESSED_DOCUMENTS_DIR, VECTOR_STORE_DIR
from app.core.exceptions import IngestionError, ShambaRafikiError
from app.core.responses import error_response, success_response
from app.models.document import DocumentChunk, EmbeddedChunk
from app.retrieval.embeddings.base import BaseEmbedder
from app.retrieval.embeddings.embedder import Embedder
from app.retrieval.store import (
    BaseVectorStore,
    create_vector_store,
    save_store,
)
from app.utils.logger import get_logger

logger = get_logger("Indexer")

MANIFEST_FILENAME = "manifest.json"


class IndexingError(ShambaRafikiError):
    """Raised when the corpus cannot be indexed."""


class Indexer:
    """
    Builds a vector index from the processed corpus manifest.
    """

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        store: BaseVectorStore | None = None,
        backend: str | None = None,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._embedder = embedder or Embedder()
        self._backend = backend or VECTOR_STORE_BACKEND
        self._store = store or create_vector_store(
            self._embedder.dimension, backend=self._backend
        )
        self._batch_size = batch_size

    def build(
        self,
        corpus_dir: Path | None = None,
        store_dir: Path | None = None,
    ) -> dict[str, Any]:
        """
        Read the corpus manifest, embed all chunks, populate and
        persist the vector store. Returns a standardized response.

        A document record that can't be read is skipped and reported
        in `data.failures` rather than aborting the whole build.
        """
        corpus_dir = corpus_dir or PROCESSED_DOCUMENTS_DIR
        store_dir = store_dir or VECTOR_STORE_DIR

        manifest_path = corpus_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            return error_response(
                f"No corpus manifest found at {manifest_path}. "
                f"Run ingestion first.",
                code="MANIFEST_NOT_FOUND",
            )

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return error_response(
                f"Could not read corpus manifest: {exc}",
                code="MANIFEST_UNREADABLE",
            )

        documents = manifest.get("documents", [])
        logger.info("indexer.build.start", documents=len(documents))

        all_chunks: list[DocumentChunk] = []
        failures: list[dict[str, str]] = []

        for entry in documents:
            try:
                chunks = self._load_chunks(corpus_dir, entry)
                all_chunks.extend(chunks)
            except IndexingError as exc:
                logger.warning(
                    "indexer.document.failed",
                    document=entry.get("filename"),
                    error=str(exc),
                )
                failures.append(
                    {
                        "document": entry.get("filename", "unknown"),
                        "error": str(exc),
                    }
                )

        if not all_chunks:
            return error_response(
                "No chunks available to index (corpus empty or all "
                "document records unreadable).",
                code="EMPTY_CORPUS",
                details={"failures": failures},
            )

        embedded = self._embed_chunks(all_chunks)
        self._store.add(embedded)

        try:
            save_store(self._store, backend=self._backend, store_dir=store_dir)
        except ShambaRafikiError as exc:
            return error_response(
                f"Indexing succeeded but persistence failed: {exc}",
                code="PERSIST_FAILED",
            )

        logger.info(
            "indexer.build.completed",
            indexed_chunks=len(embedded),
            documents_failed=len(failures),
            store_dir=str(store_dir),
        )

        return success_response(
            data={
                "indexed_chunks": len(embedded),
                "documents_indexed": len(documents) - len(failures),
                "documents_failed": len(failures),
                "store_dir": str(store_dir),
                "backend": self._backend,
                "embedding_cache_stats": getattr(
                    self._embedder, "_cache", None
                ).stats
                if getattr(self._embedder, "_cache", None)
                else None,
                "failures": failures,
            },
            message=(
                f"Indexed {len(embedded)} chunks from "
                f"{len(documents) - len(failures)} documents "
                f"using the {self._backend} backend."
            ),
        )

    def _load_chunks(
        self,
        corpus_dir: Path,
        entry: dict,
    ) -> list[DocumentChunk]:
        """
        Read one document record's chunks back into DocumentChunk
        objects. Mirrors the shape written by
        `app.ingestion.builder.CorpusBuilder`.
        """
        record_path = corpus_dir / entry["record_path"]

        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise IndexingError(
                f"Could not read record '{record_path}': {exc}"
            ) from exc

        # document_id is stored once at the record's top level (every
        # chunk in the record shares it), not repeated per chunk - so
        # read it here and apply it to each reconstructed chunk.
        try:
            document_id = UUID(record["document_id"])
        except (KeyError, ValueError) as exc:
            raise IndexingError(
                f"Record '{record_path}' has no valid document_id: {exc}"
            ) from exc

        chunks: list[DocumentChunk] = []
        for c in record.get("chunks", []):
            chunks.append(
                DocumentChunk(
                    chunk_id=UUID(c["chunk_id"]),
                    document_id=document_id,
                    chunk_index=c["chunk_index"],
                    text=c["text"],
                    token_count=c["token_count"],
                    metadata=c.get("metadata", {}),
                )
            )
        return chunks

    def _embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        """
        Embed all chunk texts (the embedder batches + caches
        internally) and pair each vector back with its chunk.
        """
        texts = [chunk.text for chunk in chunks]
        vectors = self._embedder.embed_texts(texts)

        return [
            EmbeddedChunk(chunk=chunk, embedding=vector)
            for chunk, vector in zip(chunks, vectors)
        ]