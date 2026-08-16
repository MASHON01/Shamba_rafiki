"""
Corpus builder.

Persists processed documents and their chunks to disk under
PROCESSED_DOCUMENTS_DIR, and maintains a cumulative manifest
describing the resulting corpus. Output 4 (embedding & vector
index) reads this manifest to know what to embed.

Directory layout produced:

    PROCESSED_DOCUMENTS_DIR/
        manifest.json
        documents/
            <document_id>.json   # document record + all its chunks

The manifest is cumulative and idempotent: since corpus documents
are collected across several days (Day 3 KALRO/AFA, Day 4 KAMIS,
Day 5 county context - see the build plan), each `build()` call
merges its results into any existing manifest rather than
overwriting it. Re-ingesting the same document (same document_id)
simply replaces its prior entry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config.paths import PROCESSED_DOCUMENTS_DIR
from app.core.exceptions import CorpusBuildError
from app.models.document import Document, DocumentChunk
from app.utils.logger import get_logger

logger = get_logger("CorpusBuilder")

MANIFEST_FILENAME = "manifest.json"


@dataclass(slots=True)
class BuildResult:
    """
    Summary of a single `CorpusBuilder.build()` call.
    """

    manifest_path: Path
    output_dir: Path
    documents_written: int
    chunks_written: int
    total_documents_in_corpus: int
    total_chunks_in_corpus: int


class CorpusBuilder:
    """
    Writes processed documents/chunks to disk and keeps the corpus
    manifest up to date.
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or PROCESSED_DOCUMENTS_DIR
        self._documents_dir = self._output_dir / "documents"

    def build(
        self,
        processed: list[tuple[Document, list[DocumentChunk]]],
    ) -> BuildResult:
        """
        Persist a batch of (Document, chunks) pairs and update the
        manifest. Safe to call with an empty list - the manifest is
        still (re)written so `total_*_in_corpus` stays accurate.
        """
        self._ensure_output_dirs()

        new_entries = []
        chunks_written = 0

        for document, chunks in processed:
            self._write_document_record(document, chunks)
            new_entries.append(self._manifest_entry(document, chunks))
            chunks_written += len(chunks)

        manifest, manifest_path = self._merge_and_write_manifest(new_entries)

        logger.info(
            "corpus.build.completed",
            documents_written=len(processed),
            chunks_written=chunks_written,
            total_documents=manifest["document_count"],
            total_chunks=manifest["chunk_count"],
        )

        return BuildResult(
            manifest_path=manifest_path,
            output_dir=self._output_dir,
            documents_written=len(processed),
            chunks_written=chunks_written,
            total_documents_in_corpus=manifest["document_count"],
            total_chunks_in_corpus=manifest["chunk_count"],
        )

    def _ensure_output_dirs(self) -> None:
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self._documents_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CorpusBuildError(
                f"Could not create corpus output directory "
                f"'{self._output_dir}': {exc}"
            ) from exc

    def _write_document_record(
        self,
        document: Document,
        chunks: list[DocumentChunk],
    ) -> Path:
        """
        Write one document's full processed record (metadata + all
        its chunk text) as JSON.
        """
        record = {
            "document_id": str(document.document_id),
            "filename": document.filename,
            "source": document.source,
            "file_type": document.file_type,
            "language": document.language,
            "checksum": document.checksum,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
        }

        record_path = self._documents_dir / f"{document.document_id}.json"

        try:
            record_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise CorpusBuildError(
                f"Could not write processed document "
                f"'{document.filename}': {exc}"
            ) from exc

        return record_path

    def _manifest_entry(
        self,
        document: Document,
        chunks: list[DocumentChunk],
    ) -> dict:
        record_path = self._documents_dir / f"{document.document_id}.json"

        return {
            "document_id": str(document.document_id),
            "filename": document.filename,
            "source": document.source,
            "file_type": document.file_type,
            "language": document.language,
            "checksum": document.checksum,
            "chunk_count": len(chunks),
            "record_path": str(record_path.relative_to(self._output_dir)),
            "ingested_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }

    def _merge_and_write_manifest(
        self,
        new_entries: list[dict],
    ) -> tuple[dict, Path]:
        """
        Merge `new_entries` into any existing manifest (keyed by
        document_id, new entries win) and write the result back.
        """
        manifest_path = self._output_dir / MANIFEST_FILENAME
        entries_by_id: dict[str, dict] = {}

        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                for entry in existing.get("documents", []):
                    entries_by_id[entry["document_id"]] = entry
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning(
                    "corpus.manifest.unreadable",
                    manifest_path=str(manifest_path),
                    error=str(exc),
                )

        for entry in new_entries:
            entries_by_id[entry["document_id"]] = entry

        documents = sorted(entries_by_id.values(), key=lambda e: e["filename"])

        manifest = {
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "document_count": len(documents),
            "chunk_count": sum(e["chunk_count"] for e in documents),
            "documents": documents,
        }

        try:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise CorpusBuildError(
                f"Could not write corpus manifest '{manifest_path}': {exc}"
            ) from exc

        return manifest, manifest_path