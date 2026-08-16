"""
Streaming chat route - Server-Sent Events.

    POST /chat/stream { query, language?, session_id? } (fetch + reader)
    GET /chat/stream?query=...&language=...&session_id=... (EventSource)

Same answer as POST /chat, but streamed token-by-token so the kiosk shows
it forming immediately. The response is a text/event-stream of named
events:

    event: token data: {"text": "..."} # repeated, as tokens arrive
    event: done data: {answer, sources, confidence, metadata, partial}
    event: error data: {code, message} # if the pipeline fails early

Both verbs delegate to the same source. POST carries a JSON body (for a
`fetch` + ReadableStream client); GET takes query params so a plain
browser `EventSource` can consume it.

Cancellation: the endpoint is async and watches ``request.is_disconnected``
between events. When the farmer navigates away, it sets a cancel Event
that the underlying blocking generator checks between tokens, so the
model stops generating instead of running on for a client that has gone.
The sync pipeline generator is iterated in a threadpool so a slow
CPU generation never blocks the event loop.
"""

from __future__ import annotations

import json
import threading
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from app.api.dependencies import get_orchestrator
from app.api.middleware.schemas import ChatRequest
from app.models.request import QueryRequest
from app.orchestration.orchestrator import Orchestrator, StreamMessage

router = APIRouter(tags=["Chat"])

# Tell intermediaries not to buffer the stream (nginx honours the last).
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(message: StreamMessage) -> str:
    """Format one StreamMessage as an SSE event block."""
    return f"event: {message.event}\ndata: {json.dumps(message.data)}\n\n"


async def _event_source(
    orchestrator: Orchestrator,
    query_request: QueryRequest,
    request: Request,
) -> AsyncIterator[str]:
    """Bridge the sync pipeline generator to an async SSE stream."""
    cancel_event = threading.Event
    generator = orchestrator.handle_query_stream(query_request, cancel_event=cancel_event)

    # iterate_in_threadpool runs each blocking __next__ off the event loop.
    async for message in iterate_in_threadpool(generator):
        if await request.is_disconnected:
            cancel_event.set()
            break
        yield _sse(message)


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    query_request = QueryRequest(
        query=body.query, language=body.language, session_id=body.session_id
    )
    return StreamingResponse(
        _event_source(orchestrator, query_request, request),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.get("/chat/stream")
async def chat_stream_get(
    request: Request,
    query: str,
    language: str = "en",
    session_id: str | None = None,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    query_request = QueryRequest(query=query, language=language, session_id=session_id)
    return StreamingResponse(
        _event_source(orchestrator, query_request, request),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
