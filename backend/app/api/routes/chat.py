"""
Chat route - the main farmer query endpoint.

    POST /chat   { query, language?, session_id? }  ->  answer + sources

This is the endpoint the kiosk frontend calls. It's a thin adapter:
validate the HTTP body (ChatRequest), turn it into the domain
QueryRequest, hand it to the orchestrator, and return the
orchestrator's already-standardized response. All the real work -
language analysis, retrieval, prompt assembly, generation,
verification - lives behind orchestrator.handle_query().

The handler is a plain `def` (not `async def`) on purpose: the
orchestrator's LLM call is blocking, and FastAPI runs sync handlers
in a threadpool, so a slow generation doesn't stall the event loop.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_orchestrator
from app.api.middleware.schemas import ChatRequest
from app.models.request import QueryRequest
from app.orchestration.orchestrator import Orchestrator

router = APIRouter(tags=["Chat"])


@router.post("/chat")
def chat(
    body: ChatRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> dict:
    """
    Answer a farmer's question. Returns the standard response envelope
    with the (verified) answer, its sources, confidence, and metadata.
    """
    request = QueryRequest(
        query=body.query,
        language=body.language,
        session_id=body.session_id,
    )
    return orchestrator.handle_query(request)