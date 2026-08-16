"""
Application configuration validation.
"""

from app.config.settings import settings


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""


def validate_configuration() -> None:
    """
    Validate required configuration values.
    """

    if settings.API_PORT <= 0:
        raise ConfigurationError("Invalid API_PORT")

    if settings.MODEL_CONTEXT_SIZE <= 0:
        raise ConfigurationError("MODEL_CONTEXT_SIZE must be greater than zero")

    if not settings.model_path.parent.exists():
        raise ConfigurationError(
            f"Model directory does not exist: {settings.model_path.parent}"
        )

    if settings.TOP_K <= 0:
        raise ConfigurationError("TOP_K must be positive")

    if settings.MAX_CONTEXT_CHUNKS <= 0:
        raise ConfigurationError("MAX_CONTEXT_CHUNKS must be positive")

    if not 0 <= settings.SIMILARITY_THRESHOLD <= 1:
        raise ConfigurationError(
            "SIMILARITY_THRESHOLD must be between 0 and 1"
        )

    if settings.DEFAULT_LANGUAGE not in settings.SUPPORTED_LANGUAGES:
        raise ConfigurationError(
            "DEFAULT_LANGUAGE must exist in SUPPORTED_LANGUAGES"
        )