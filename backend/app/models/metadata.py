"""
Metadata models shared throughout the application.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ProcessingMetadata(BaseModel):
    """
    Processing statistics collected during
    request execution.
    """

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    finished_at: datetime | None = None

    duration_ms: float | None = None

    language: str | None = None

    model: str | None = None

    confidence: float | None = None


class RetrievalMetadata(BaseModel):
    """
    Retrieval information.
    """

    retrieved_chunks: int = 0

    similarity_score: float | None = None

    embedding_model: str | None = None

    vector_store: str | None = None


class ClassifierMetadata(BaseModel):
    """
    Image classifier metadata.
    """

    predicted_label: str | None = None

    confidence: float | None = None

    inference_time_ms: float | None = None


class DocumentMetadata(BaseModel):
    """
    Metadata describing a document.
    """

    filename: str

    source: str

    document_type: str

    language: str

    checksum: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )