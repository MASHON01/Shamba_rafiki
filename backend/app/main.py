"""
Application entrypoint.

Thin over `app.api.app_factory.create_app()` - all wiring (routes,
middleware, error handlers, lifespan) lives in the factory, so this
file only exposes the ASGI `app` object and the dev-server runner.

Run in development:
    python -m app.main
Run in production (recommended):
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from app.api.app_factory import create_app
from app.config.settings import settings

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )