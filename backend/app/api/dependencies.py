"""
API dependency providers.

FastAPI dependencies that hand route handlers a ready-to-use
Orchestrator. The heavy objects - the embedding model, the vector
index, the LLM client - are expensive to construct, so they're built
ONCE at application startup (via `init_dependencies`) and reused for
every request. Per-request construction would reload the index and
model on every call, which is exactly what the RAM/latency budget
can't afford.

Startup is deliberately tolerant: if no vector index exists yet (the
build script hasn't run), the orchestrator is still created, just
without a retriever, so the API comes up and answers in a degraded
"no corpus" mode rather than refusing to start. The /health endpoint
surfaces the real readiness.
"""

from __future__ import annotations

from app.orchestration.llm.llama_client import LlamaClient
from app.orchestration.orchestrator import Orchestrator
from app.orchestration.dispatcher import Dispatcher
from app.utils.logger import get_logger

logger = get_logger("APIDependencies")

# Module-level singletons, populated by init_dependencies() at startup.
_orchestrator: Orchestrator | None = None


def init_dependencies() -> None:
    """
    Build the singleton orchestrator (and its retriever, if an index
    exists). Called once from the app lifespan at startup.
    """
    global _orchestrator

    from app.config.settings import settings
    from app.orchestration.llm.factory import build_llm_client

    retriever = _try_load_retriever()
    # Compose the hardened client: LlamaClient -> resilience -> response cache.
    # The cache is what lets an identical repeat question return instantly.
    llm_client = build_llm_client()
    dispatcher = Dispatcher(llm_client=llm_client, retriever=retriever)
    _orchestrator = Orchestrator(llm_client=llm_client, dispatcher=dispatcher)

    # Pay the model's cold-start cost now, before the first farmer.
    if settings.WARMUP_ON_STARTUP:
        try:
            from app.orchestration.cache.warmup import warmup

            warmup(llm_client)
        except Exception as exc:  # noqa: BLE001 - warmup is best-effort
            logger.info("api.warmup.skipped", reason=str(exc))

    logger.info(
        "api.dependencies.ready",
        retriever_loaded=retriever is not None,
        response_cache=settings.LLM_RESPONSE_CACHE_ENABLED,
    )


def shutdown_dependencies() -> None:
    """Release singletons at shutdown."""
    global _orchestrator
    _orchestrator = None


def get_orchestrator() -> Orchestrator:
    """
    FastAPI dependency: the shared orchestrator. Raises if the app
    wasn't initialized (shouldn't happen in normal startup).
    """
    if _orchestrator is None:
        # Lazily initialize as a safety net (e.g. in a test that skipped
        # lifespan). Normal startup calls init_dependencies() first.
        init_dependencies()
    assert _orchestrator is not None
    return _orchestrator


def get_retriever():
    """
    FastAPI dependency: the shared retriever, or None if no index is
    loaded. Routes that need raw retrieval depend on this.
    """
    orch = get_orchestrator()
    return orch._dispatcher._retriever


def _try_load_retriever():
    """
    Load the pre-built vector index from disk. Returns None (not an
    error) if no index is present yet - the API should still start.
    """
    try:
        from app.retrieval.search import Retriever

        retriever = Retriever.from_disk()
        logger.info("api.retriever.loaded", chunks=len(retriever))
        return retriever
    except Exception as exc:  # noqa: BLE001 - startup must be tolerant
        logger.warning(
            "api.retriever.unavailable",
            reason=str(exc),
            hint="Run the build_index script to enable retrieval.",
        )
        return None