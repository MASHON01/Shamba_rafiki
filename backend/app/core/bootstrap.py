"""
Application bootstrap.

Responsible for preparing the application before it starts
serving requests.
"""

from app.config.paths import create_directories
from app.config.settings import settings
from app.utils.logger import get_logger
from app.core.validator import validate_configuration

logger = get_logger("Bootstrap")


def bootstrap() -> None:
    logger.info("Starting application bootstrap")

    try:
        validate_configuration()
        create_directories()

        logger.info(
            "Application initialized successfully",
            app=settings.APP_NAME,
            version=settings.APP_VERSION,
            environment=settings.APP_ENV,
        )

    except Exception:
        logger.exception("Application bootstrap failed")
        raise