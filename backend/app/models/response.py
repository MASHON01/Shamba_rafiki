"""
Standard response models.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ErrorModel(BaseModel):
    """
    Standard error payload.
    """

    code: str

    message: str

    details: Any | None = None


class BaseResponse(BaseModel):
    """
    Base application response.
    """

    request_id: UUID = Field(default_factory=uuid4)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    success: bool


class SuccessResponse(BaseResponse):
    """
    Successful operation.
    """

    success: bool = True

    message: str = "Success"

    data: Any | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseResponse):
    """
    Failed operation.
    """

    success: bool = False

    error: ErrorModel