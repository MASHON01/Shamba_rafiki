"""
API exception handlers.

Maps the application's exception hierarchy (and FastAPI's own
validation errors) onto clean JSON using the standard
`error_response` envelope, so a raised ShambaRafikiError becomes a
proper HTTP status with a structured body - never a bare 500 stack
trace leaking to the kiosk.

The status mapping is intentional per exception family:
- validation / unsupported / empty document -> 400 (client's fault)
- resource / dependency not found            -> 404
- everything else under ShambaRafikiError     -> 500 (our fault)

A catch-all handler covers truly unexpected exceptions so the API
always returns the same envelope shape, which the frontend can rely
on.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse

from app.core.exceptions import (
    DependencyNotFoundError,
    ResourceNotFoundError,
    ShambaRafikiError,
    ValidationError,
)
from app.core.responses import error_response
from app.utils.logger import get_logger

logger = get_logger("APIErrors")

# Exception family -> HTTP status. Order matters: more specific
# (subclass) checks are handled by isinstance ordering below.
_NOT_FOUND = (ResourceNotFoundError, DependencyNotFoundError)


def _status_for(exc: ShambaRafikiError) -> int:
    if isinstance(exc, ValidationError):
        return 400
    if isinstance(exc, _NOT_FOUND):
        return 404
    return 500


def register_exception_handlers(app: FastAPI) -> None:
    """
    Attach all exception handlers to the app.
    """

    @app.exception_handler(ShambaRafikiError)
    async def _handle_app_error(
        request: Request, exc: ShambaRafikiError
    ) -> ORJSONResponse:
        status = _status_for(exc)
        # 5xx is our fault: log with stack. 4xx is the client's: log quietly.
        if status >= 500:
            logger.exception(
                "api.error.server",
                path=request.url.path,
                error_type=type(exc).__name__,
            )
        else:
            logger.info(
                "api.error.client",
                path=request.url.path,
                error_type=type(exc).__name__,
                detail=str(exc),
            )
        return ORJSONResponse(
            status_code=status,
            content=error_response(
                str(exc) or "Request could not be completed.",
                code=type(exc).__name__,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        request: Request, exc: RequestValidationError
    ) -> ORJSONResponse:
        return ORJSONResponse(
            status_code=422,
            content=error_response(
                "Request validation failed.",
                code="REQUEST_VALIDATION_ERROR",
                details=exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(
        request: Request, exc: Exception
    ) -> ORJSONResponse:
        logger.exception(
            "api.error.unexpected", path=request.url.path
        )
        return ORJSONResponse(
            status_code=500,
            content=error_response(
                "An unexpected error occurred.",
                code="INTERNAL_ERROR",
            ),
        )