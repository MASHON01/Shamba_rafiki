"""
Application factory.

`create_app()` builds and wires the FastAPI instance: lifespan
(startup builds the singleton orchestrator/retriever, shutdown
releases them), middleware (request-id/timing, CORS), exception
handlers, and the API routers.

The factory pattern is what makes the API testable - Output 9's tests
call `create_app()` and drive it with FastAPI's TestClient, with no
live server and no real llama-server required (dependencies can be
overridden). `main.py` becomes a one-liner over this.

Routers are imported lazily inside `create_app()` so that importing
this module doesn't drag in the whole route/dependency graph until an
app is actually built - and so the routes group can be added without
touching the import surface here.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from app.api.dependencies import init_dependencies, shutdown_dependencies
from app.api.errors import register_exception_handlers
from app.api.middleware.request_id import RequestContextMiddleware
from app.config.logging import configure_logging
from app.config.settings import settings
from app.core.bootstrap import bootstrap
from app.utils.logger import get_logger

logger = get_logger("AppFactory")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("app.starting", env=settings.APP_ENV)
    bootstrap()             # validate config + ensure directories exist
    init_dependencies()     # build orchestrator + retriever singletons
    logger.info("app.ready")
    yield
    shutdown_dependencies()
    logger.info("app.shutdown_complete")


def create_app() -> FastAPI:
    """
    Build the fully-wired FastAPI application.
    """
    configure_logging()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Offline AI Agricultural Advisory Backend",
        default_response_class=ORJSONResponse,
        lifespan=_lifespan,
    )

    # Middleware. CORS is added last so it runs outermost (first to see
    # the request, last to touch the response).
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOW_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    _register_routers(app)
    _mount_frontend(app)

    logger.info("app.created")
    return app


def _register_routers(app: FastAPI) -> None:
    """
    Attach API routers. Imported lazily and guarded so the app can be
    built even while the routes group is still being added - each
    router is included only if present.
    """
    try:
        from app.api.routes import (
            chat,
            classify,
            health,
            retrieval,
            upload,
        )
    except ImportError as exc:
        logger.warning("app.routes_unavailable", reason=str(exc))
        return

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(retrieval.router)
    app.include_router(upload.router)
    app.include_router(classify.router)

def _mount_frontend(app) -> None:
    """Serve the static kiosk UI at /app, only if the frontend/ dir exists."""
    from app.config.paths import FRONTEND_DIR

    if not FRONTEND_DIR.is_dir():
        logger.info("app.frontend_absent", path=str(FRONTEND_DIR))
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info("app.frontend_mounted", path=str(FRONTEND_DIR))
