"""
Chunk metadata generation.

Tags each chunk with the fields retrieval and the LLM prompt will
lean on: crop, county, document_type, language, plus provenance
(source filename, chunk index, checksum). Detection is lightweight
and dependency-free - keyword/vocabulary matching over the chunk
text and the document's filename/source - which is the right weight
for Phase 1 and keeps the RAM budget untouched.

Output is strictly `dict[str, str]` to match
`DocumentChunk.metadata`; anything undetected becomes
`METADATA_UNKNOWN` rather than None, so the dict stays clean and the
validator's coverage check (crop/county/document_type/language) is
always satisfied with real string values.

Satisfies the `MetadataGenerator` protocol in
`app.ingestion.pipeline`
(`generate(document, chunk_index, chunk_text) -> dict[str, str]`).
"""

from __future__ import annotations

from app.config.constants import (
    DEFAULT_DOCUMENT_TYPE,
    DOCUMENT_TYPE_KEYWORDS,
    KNOWN_COUNTIES,
    KNOWN_CROPS,
    METADATA_UNKNOWN,
)
from app.models.document import Document


class MetadataGenerator:
    """
    Produces per-chunk metadata via vocabulary/keyword matching.
    """

    def generate(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str,
    ) -> dict[str, str]:
        haystack = chunk_text.lower()
        filename_hay = f"{document.filename} {document.source}".lower()

        metadata = {
            # Required-coverage keys (see REQUIRED_CHUNK_METADATA_KEYS).
            "crop": self._match_vocabulary(haystack, KNOWN_CROPS),
            "county": self._match_vocabulary(haystack, KNOWN_COUNTIES),
            "document_type": self._detect_document_type(filename_hay, haystack),
            "language": document.language or METADATA_UNKNOWN,
            # Provenance - all coerced to str for DocumentChunk.metadata.
            "source": document.source,
            "source_filename": document.filename,
            "file_type": document.file_type,
            "document_id": str(document.document_id),
            "chunk_index": str(chunk_index),
            "checksum": document.checksum,
        }

        return metadata

    def _match_vocabulary(
        self,
        haystack: str,
        vocabulary: dict[str, list[str]],
    ) -> str:
        """
        Return the canonical label whose surface forms appear in the
        text. If several match, the one whose surface form appears
        earliest wins (a document is usually 'about' whatever it names
        first); ties fall back to vocabulary order.
        """
        best_label = METADATA_UNKNOWN
        best_position = len(haystack) + 1

        for label, surface_forms in vocabulary.items():
            for form in surface_forms:
                position = haystack.find(form)
                if position != -1 and position < best_position:
                    best_position = position
                    best_label = label

        return best_label

    def _detect_document_type(self, filename_hay: str, text_hay: str) -> str:
        """
        Infer document type from filename/source first (most reliable),
        then fall back to scanning the chunk text.
        """
        for source in (filename_hay, text_hay):
            for doc_type, keywords in DOCUMENT_TYPE_KEYWORDS.items():
                if any(keyword in source for keyword in keywords):
                    return doc_type
        return DEFAULT_DOCUMENT_TYPE