"""
Request models used throughout the application.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class BaseRequest(BaseModel):
    """Base request shared by every application request."""

    request_id: UUID = Field(default_factory=uuid4)


class QueryRequest(BaseRequest):
    """Farmer text query."""

    query: str = Field(..., min_length=1)
    language: str = Field(default="en", description="ISO language code.")
    session_id: str | None = None


class ImageRequest(BaseRequest):
    """
    Image classification request: a leaf photo, optionally with a text
    question. When no text is given, the orchestrator synthesizes one
    from the predicted crop + condition.
    """

    image_path: str
    query: str | None = None
    language: str = "en"
    session_id: str | None = None


class RetrievalRequest(BaseRequest):
    """Vector retrieval request."""

    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] | None = None


class HealthRequest(BaseRequest):
    """Health endpoint request."""

    verbose: bool = False
