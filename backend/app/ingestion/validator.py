"""
Document and chunk validation for the ingestion pipeline.

Not to be confused with `app.core.validator`, which validates the
*application's* configuration at startup. This module validates the
*data* moving through the ingestion pipeline: is the format
supported, does the document contain real text, are the resulting
chunks well-formed.

Each `validate_*` method raises a typed exception (see
`app.core.exceptions`) on failure and returns `None` on success, so
`app.ingestion.pipeline.IngestionPipeline` can call them inline
without inspecting a return value.
"""

from __future__ import annotations

from app.config.constants import (
    MIN_CHUNK_TOKEN_COUNT,
    MIN_DOCUMENT_TEXT_LENGTH,
    REQUIRED_CHUNK_METADATA_KEYS,
    SUPPORTED_DOCUMENT_TYPES,
)
from app.core.exceptions import (
    CorruptedDocumentError,
    EmptyDocumentError,
    UnsupportedFormatError,
)
from app.models.document import Document, DocumentChunk
from app.utils.logger import get_logger

logger = get_logger("DocumentValidator")


class DocumentValidator:
    """
    Validates documents and chunks at each stage of ingestion.

    Usage mirrors the pipeline's own stages:

        validator.validate_document(document)   # before extraction
        validator.validate_text(text, document)  # after clean, before chunk
        validator.validate_chunks(chunks, document)  # after chunk + metadata
    """

    def validate_document(self, document: Document) -> None:
        """
        Validate a `Document` before any text has been extracted:
        format support and basic path/checksum sanity.
        """
        if document.file_type not in SUPPORTED_DOCUMENT_TYPES:
            raise UnsupportedFormatError(
                f"Unsupported document type '{document.file_type}' for "
                f"'{document.filename}'. Supported types: "
                f"{SUPPORTED_DOCUMENT_TYPES}"
            )

        if not document.path.exists():
            raise CorruptedDocumentError(
                f"Document path does not exist: {document.path}"
            )

        if not document.path.is_file():
            raise CorruptedDocumentError(
                f"Document path is not a file: {document.path}"
            )

        if not document.checksum:
            raise CorruptedDocumentError(
                f"Document '{document.filename}' has no checksum computed."
            )

    def validate_text(self, text: str, *, document: Document) -> None:
        """
        Validate extracted/cleaned text before it is chunked.
        """
        stripped = text.strip() if text else ""

        if not stripped:
            raise EmptyDocumentError(
                f"No extractable text found in '{document.filename}'. "
                f"This is often a scanned/image-only PDF awaiting OCR."
            )

        if len(stripped) < MIN_DOCUMENT_TEXT_LENGTH:
            raise EmptyDocumentError(
                f"'{document.filename}' produced only {len(stripped)} "
                f"characters of text (minimum is {MIN_DOCUMENT_TEXT_LENGTH}) "
                f"- likely a failed or partial extraction."
            )

    def validate_chunks(
        self,
        chunks: list[DocumentChunk],
        *,
        document: Document,
    ) -> None:
        """
        Validate the final list of chunks produced for a document:
        non-empty overall, no duplicate/mismatched IDs, each chunk has
        real text, and metadata coverage is logged if incomplete.
        """
        if not chunks:
            raise EmptyDocumentError(
                f"'{document.filename}' produced zero chunks after chunking."
            )

        seen_chunk_ids: set = set()

        for chunk in chunks:
            if chunk.chunk_id in seen_chunk_ids:
                raise CorruptedDocumentError(
                    f"Duplicate chunk_id '{chunk.chunk_id}' produced for "
                    f"'{document.filename}'."
                )
            seen_chunk_ids.add(chunk.chunk_id)

            if chunk.document_id != document.document_id:
                raise CorruptedDocumentError(
                    f"Chunk '{chunk.chunk_id}' references document_id "
                    f"'{chunk.document_id}', expected "
                    f"'{document.document_id}'."
                )

            if not chunk.text or not chunk.text.strip():
                raise EmptyDocumentError(
                    f"Empty chunk (index {chunk.chunk_index}) in "
                    f"'{document.filename}'."
                )

            if chunk.token_count < MIN_CHUNK_TOKEN_COUNT:
                logger.warning(
                    "validator.chunk.low_token_count",
                    document=document.filename,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count,
                )

            self._check_metadata_coverage(chunk, document=document)

    def _check_metadata_coverage(
        self,
        chunk: DocumentChunk,
        *,
        document: Document,
    ) -> None:
        """
        Log (don't fail) when a chunk is missing expected metadata
        keys. `DocumentChunk.metadata` is already type-enforced as
        dict[str, str] by the Pydantic model - this only checks
        *coverage*, since not every document will genuinely have a
        crop or county (e.g. general county-context pages).
        """
        missing = set(REQUIRED_CHUNK_METADATA_KEYS) - chunk.metadata.keys()

        if missing:
            logger.warning(
                "validator.chunk.missing_metadata",
                document=document.filename,
                chunk_index=chunk.chunk_index,
                missing_keys=sorted(missing),
            )