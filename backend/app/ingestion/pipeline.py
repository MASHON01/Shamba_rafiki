"""
Main ingestion orchestrator.

Coordinates a single document through the full pipeline:

    Path -> Extract -> Clean -> Validate -> Chunk -> Tag metadata
         -> Validate chunks -> Corpus

`IngestionPipeline` depends on its extraction/cleaning/chunking/
metadata/hashing/dedup components only through the `Protocol`
interfaces below - not concrete imports. The concrete
implementations (TextExtractor, Cleaner, Chunker,
MetadataGenerator, Hasher, DuplicateDetector) live under
`app/ingestion/{extractors,processors}/` and are built next; this
file needs no changes when they land - it just needs objects that
satisfy these method signatures. `create_default_pipeline()` below
is the wiring point once they exist.

`validator.py` and `builder.py` are concrete dependencies (built
alongside this file), so they're imported directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from app.config.constants import SUPPORTED_DOCUMENT_TYPES
from app.core.exceptions import DuplicateDocumentError, IngestionError
from app.core.responses import error_response, success_response
from app.ingestion.builder import CorpusBuilder
from app.ingestion.validator import DocumentValidator
from app.models.document import Document, DocumentChunk
from app.utils.logger import get_logger

logger = get_logger("IngestionPipeline")


# =============================================================================
# Component interfaces (structural typing - no inheritance required)
# =============================================================================


class Extractor(Protocol):
    """Turns a source file into raw text. Owns loader selection
    (pdf/docx/txt) and any OCR fallback internally."""

    def extract(self, path: Path) -> str: ...


class Cleaner(Protocol):
    """Strips headers/footers/OCR noise and normalizes whitespace."""

    def clean(self, text: str) -> str: ...


class Chunker(Protocol):
    """Splits clean text into ~200-400 token, paragraph-aware chunks."""

    def chunk(self, text: str) -> list[str]: ...


class MetadataGenerator(Protocol):
    """Produces per-chunk metadata (crop, county, document_type,
    language, etc.) as dict[str, str], matching DocumentChunk.metadata."""

    def generate(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str,
    ) -> dict[str, str]: ...


class Hasher(Protocol):
    """Computes a content checksum for duplicate detection."""

    def compute(self, path: Path) -> str: ...


class DuplicateDetector(Protocol):
    """Tracks checksums already ingested into the corpus."""

    def is_duplicate(self, checksum: str) -> bool: ...

    def register(self, checksum: str, document_id: UUID) -> None: ...


# =============================================================================
# Orchestrator
# =============================================================================


class IngestionPipeline:
    """
    Runs documents through extraction, cleaning, validation,
    chunking, metadata tagging, and corpus assembly.
    """

    def __init__(
        self,
        *,
        extractor: Extractor,
        cleaner: Cleaner,
        chunker: Chunker,
        metadata_generator: MetadataGenerator,
        hasher: Hasher,
        duplicate_detector: DuplicateDetector,
        validator: DocumentValidator | None = None,
        builder: CorpusBuilder | None = None,
    ) -> None:
        self._extractor = extractor
        self._cleaner = cleaner
        self._chunker = chunker
        self._metadata_generator = metadata_generator
        self._hasher = hasher
        self._duplicate_detector = duplicate_detector
        self._validator = validator or DocumentValidator()
        self._builder = builder or CorpusBuilder()

    def ingest_document(
        self,
        path: Path,
        *,
        source: str,
        language: str = "en",
    ) -> tuple[Document, list[DocumentChunk]]:
        """
        Run a single document through the full pipeline.

        Raises an `IngestionError` subclass on failure (see
        `app.core.exceptions`); raises rather than returning an
        error response, so callers processing many documents (see
        `run()`) can decide per-document whether to skip or abort.
        """
        logger.info("ingestion.document.start", path=str(path))

        checksum = self._hasher.compute(path)

        if self._duplicate_detector.is_duplicate(checksum):
            raise DuplicateDocumentError(
                f"'{path.name}' duplicates a document already in the "
                f"corpus (checksum={checksum})."
            )

        document = Document(
            filename=path.name,
            path=path,
            file_type=path.suffix.lower(),
            checksum=checksum,
            language=language,
            source=source,
        )

        self._validator.validate_document(document)

        raw_text = self._extractor.extract(path)
        clean_text = self._cleaner.clean(raw_text)

        self._validator.validate_text(clean_text, document=document)

        chunks = self._build_chunks(document, clean_text)

        self._validator.validate_chunks(chunks, document=document)

        self._duplicate_detector.register(checksum, document.document_id)

        logger.info(
            "ingestion.document.completed",
            path=str(path),
            chunks=len(chunks),
        )

        return document, chunks

    def _build_chunks(
        self,
        document: Document,
        clean_text: str,
    ) -> list[DocumentChunk]:
        chunk_texts = self._chunker.chunk(clean_text)

        chunks: list[DocumentChunk] = []

        for index, chunk_text in enumerate(chunk_texts):
            metadata = self._metadata_generator.generate(
                document, index, chunk_text
            )

            chunks.append(
                DocumentChunk(
                    document_id=document.document_id,
                    chunk_index=index,
                    text=chunk_text,
                    # Whitespace-split placeholder until the embedding
                    # model's real tokenizer is wired in (Output 4).
                    token_count=len(chunk_text.split()),
                    metadata=metadata,
                )
            )

        return chunks

    def run(
        self,
        directory: Path,
        *,
        source: str,
        language: str = "en",
    ) -> dict[str, Any]:
        """
        Ingest every supported document under `directory` (recursive),
        build/update the corpus, and return a standardized response
        (`success_response`/`error_response`).

        A single bad document does not abort the run - its failure is
        captured in the returned `data.failures` list so the whole
        batch (e.g. a folder of KALRO PDFs) doesn't need to be perfect
        to make progress.
        """
        if not directory.exists():
            return error_response(
                f"Directory does not exist: {directory}",
                code="DIRECTORY_NOT_FOUND",
            )

        if not directory.is_dir():
            return error_response(
                f"Not a directory: {directory}",
                code="NOT_A_DIRECTORY",
            )

        candidate_paths = sorted(
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_DOCUMENT_TYPES
        )

        logger.info(
            "ingestion.run.start",
            directory=str(directory),
            candidates=len(candidate_paths),
        )

        processed: list[tuple[Document, list[DocumentChunk]]] = []
        failures: list[dict[str, str]] = []

        for path in candidate_paths:
            try:
                document, chunks = self.ingest_document(
                    path, source=source, language=language
                )
                processed.append((document, chunks))
            except IngestionError as exc:
                logger.warning(
                    "ingestion.document.failed",
                    path=str(path),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                failures.append(
                    {
                        "path": str(path),
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
                logger.exception(
                    "ingestion.document.unexpected_error", path=str(path)
                )
                failures.append(
                    {
                        "path": str(path),
                        "error": str(exc),
                        "type": "UnexpectedError",
                    }
                )

        build_result = self._builder.build(processed)

        logger.info(
            "ingestion.run.completed",
            processed=len(processed),
            failed=len(failures),
            manifest=str(build_result.manifest_path),
        )

        return success_response(
            data={
                "manifest_path": str(build_result.manifest_path),
                "documents_processed": len(processed),
                "documents_failed": len(failures),
                "chunks_written": build_result.chunks_written,
                "total_documents_in_corpus": build_result.total_documents_in_corpus,
                "total_chunks_in_corpus": build_result.total_chunks_in_corpus,
                "failures": failures,
            },
            message=(
                f"Ingested {len(processed)}/{len(candidate_paths)} "
                f"documents from {directory}."
            ),
            metadata={
                "source": source,
                "language": language,
                "directory": str(directory),
            },
        )


def create_default_pipeline() -> IngestionPipeline:
    """
    Wire an `IngestionPipeline` to the concrete extractor, cleaner,
    chunker, metadata generator, hasher, and duplicate detector.

    Imports are deferred so `pipeline.py` stays importable before
    those components exist; this factory only needs to work once
    they're built (remaining Output 3 files).
    """
    from app.ingestion.extractors.text_extractor import TextExtractor
    from app.ingestion.processors.chunker import Chunker as TextChunker
    from app.ingestion.processors.cleaner import Cleaner as TextCleaner
    from app.ingestion.processors.duplicate_detector import (
        DuplicateDetector as CorpusDuplicateDetector,
    )
    from app.ingestion.processors.hashing import Hasher as DocumentHasher
    from app.ingestion.processors.metadata_generator import (
        MetadataGenerator as ChunkMetadataGenerator,
    )

    return IngestionPipeline(
        extractor=TextExtractor(),
        cleaner=TextCleaner(),
        chunker=TextChunker(),
        metadata_generator=ChunkMetadataGenerator(),
        hasher=DocumentHasher(),
        duplicate_detector=CorpusDuplicateDetector(),
    )