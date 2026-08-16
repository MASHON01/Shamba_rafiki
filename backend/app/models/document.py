"""
Document models.

Used throughout ingestion, chunking,
embedding, and retrieval.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Original source document.
    """

    document_id: UUID = Field(default_factory=uuid4)

    filename: str

    path: Path

    file_type: str

    checksum: str

    language: str

    source: str


class DocumentChunk(BaseModel):
    """
    A chunk produced from a document.
    """

    chunk_id: UUID = Field(default_factory=uuid4)

    document_id: UUID

    chunk_index: int

    text: str

    token_count: int

    metadata: dict[str, str] = Field(
        default_factory=dict
    )


class EmbeddedChunk(BaseModel):
    """
    Chunk together with its embedding.
    """

    chunk: DocumentChunk

    embedding: list[float]


class RetrievalResult(BaseModel):
    """
    Result returned by vector retrieval.
    """

    chunk: DocumentChunk

    similarity_score: float