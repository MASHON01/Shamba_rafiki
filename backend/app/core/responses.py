"""
Standardized response helpers.

Every module should return responses
using these helpers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    """
    Current UTC timestamp.
    """
    return (
        datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
    )


def request_id() -> str:
    """
    Generate a unique request identifier.
    """
    return str(uuid4())


def success_response(
    data: Any,
    *,
    message: str = "Success",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Standard success response.
    """

    return {
        "success": True,
        "request_id": request_id(),
        "timestamp": utc_now(),
        "message": message,
        "data": data,
        "metadata": metadata or {},
    }


def error_response(
    message: str,
    *,
    code: str = "APPLICATION_ERROR",
    details: Any = None,
) -> dict[str, Any]:
    """
    Standard error response.
    """

    return {
        "success": False,
        "request_id": request_id(),
        "timestamp": utc_now(),
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
    }