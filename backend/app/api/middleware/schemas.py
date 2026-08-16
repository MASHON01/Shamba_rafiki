"""
API-boundary request/response schemas.

These are the *transport* shapes - what an HTTP client sends and
receives - kept separate from the domain models in `app/models/`.
The boundary validates and shapes HTTP input (a JSON body) before it
becomes a domain `QueryRequest`, and documents the response envelope
for OpenAPI.

Responses themselves reuse the standard envelope from
`app.core.responses` (success/error), so these response models exist
mainly to give FastAPI's generated docs a concrete shape rather than
to re-serialize anything.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --- Requests ------------------------------------------------------------


class ChatRequest(BaseModel):
    """Body for POST /chat - a farmer's question."""

    query: str = Field(..., min_length=1, description="The farmer's question.")
    language: str = Field(
        default="en", description="ISO language hint ('en' or 'sw')."
    )
    session_id: str | None = Field(
        default=None,
        description="Opaque session id for follow-up context within one "
        "farmer's visit. Omit for a stateless one-off question.",
    )


class RetrieveRequest(BaseModel):
    """Body for POST /retrieve - raw retrieval, no LLM."""

    query: str = Field(..., min_length=1)
    top_k: int | None = Field(
        default=None, ge=1, le=20,
        description="How many chunks to return (defaults to the configured "
        "top-k).",
    )


# --- Response envelope (for OpenAPI docs) --------------------------------


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    """The standard error envelope (mirrors core.responses.error_response)."""

    success: bool = False
    request_id: str
    timestamp: str
    error: ErrorBody


class SuccessResponse(BaseModel):
    """The standard success envelope (mirrors core.responses.success_response)."""

    success: bool = True
    request_id: str
    timestamp: str
    message: str
    data: Any
    metadata: dict[str, Any] = Field(default_factory=dict)