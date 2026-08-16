"""
Request-ID and timing middleware.

Gives every HTTP request a trace id and logs how long it took. The id
is taken from the incoming REQUEST_ID_HEADER if the client sent one
(so a frontend can correlate its own logs), otherwise generated, and
echoed back on the response. This is the HTTP-level complement to the
request_id that already flows through the domain models.

Kept as pure ASGI-level middleware (no per-route wiring) so it
applies uniformly and stays out of the route handlers' way.
"""

from __future__ import annotations

import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from fastapi.responses import ORJSONResponse

from app.config.constants import REQUEST_ID_HEADER
from app.core.responses import error_response
from app.utils.logger import get_logger

logger = get_logger("APIRequest")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Attaches a request id, times the request, logs the outcome.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        request.state.request_id = request_id

        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            # A BaseHTTPMiddleware sits OUTSIDE FastAPI's exception-handler
            # layer, so an exception that no registered handler produced a
            # response for would otherwise escape as a raw, envelope-less
            # 500. Catch it here as a last resort and emit the standard
            # error envelope, so the frontend always gets the same shape.
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.exception(
                "api.request.unhandled",
                method=request.method,
                path=request.url.path,
                request_id=request_id,
                duration_ms=duration_ms,
            )
            response = ORJSONResponse(
                status_code=500,
                content=error_response(
                    "An unexpected error occurred.",
                    code="INTERNAL_ERROR",
                ),
            )
            response.headers[REQUEST_ID_HEADER] = request_id
            return response

        duration_ms = int((time.monotonic() - started) * 1000)
        response.headers[REQUEST_ID_HEADER] = request_id

        logger.info(
            "api.request.completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            request_id=request_id,
            duration_ms=duration_ms,
        )
        return response