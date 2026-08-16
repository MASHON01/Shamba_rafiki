"""
Health and system routes.

    GET /            liveness + basic identity
    GET /health      deep health - reports whether the LLM backend is
                     reachable, via the orchestrator
    GET /version     app version

/health is "deep": it doesn't just prove the process is up, it asks
the orchestrator whether llama-server is actually reachable, so the
kiosk operator can tell the difference between "API running" and
"API running but the model isn't answering". It still returns 200
(the API is up); the payload carries the real LLM status.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_orchestrator
from app.config.settings import settings
from app.core.responses import success_response
from app.orchestration.orchestrator import Orchestrator

router = APIRouter(tags=["Health"])


@router.get("/")
def root() -> dict:
    return success_response(
        data={
            "application": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "status": "running",
            "environment": settings.APP_ENV,
        },
        message="Shamba Rafiki backend is running.",
    )


@router.get("/health")
def health(orchestrator: Orchestrator = Depends(get_orchestrator)) -> dict:
    # orchestrator.health() already returns a standard response dict
    # reporting llm_available; pass it straight through.
    return orchestrator.health()


@router.get("/version")
def version() -> dict:
    return success_response(
        data={"version": settings.APP_VERSION},
        message="ok",
    )