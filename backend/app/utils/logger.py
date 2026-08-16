"""
Centralized application logger.

This module provides a singleton logger used throughout the application.

Usage:

    from app.utils.logger import get_logger

    logger = get_logger(__name__)

    logger.info("Application started")
    logger.warning("Low confidence prediction")
    logger.error("Vector store unavailable")
"""

from __future__ import annotations

import logging
from typing import Optional

import structlog


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Return a structured logger.

    Parameters
    ----------
    name : str | None
        Name of the logger. Defaults to the root application logger.

    Returns
    -------
    structlog.stdlib.BoundLogger
        Configured structured logger.
    """

    logger_name = name or "shamba_rafiki"

    return structlog.get_logger(logger_name)


def bind_request_id(
    logger: structlog.stdlib.BoundLogger,
    request_id: str,
) -> structlog.stdlib.BoundLogger:
    """
    Bind a request ID to a logger.

    Parameters
    ----------
    logger:
        Existing logger.

    request_id:
        Unique request identifier.

    Returns
    -------
    BoundLogger
    """

    return logger.bind(request_id=request_id)


def bind_component(
    logger: structlog.stdlib.BoundLogger,
    component: str,
) -> structlog.stdlib.BoundLogger:
    """
    Attach the component name to log entries.

    Example
    -------
    component="Retriever"
    component="PromptBuilder"
    component="Vision"
    """

    return logger.bind(component=component)


def bind_session(
    logger: structlog.stdlib.BoundLogger,
    session_id: str,
) -> structlog.stdlib.BoundLogger:
    """
    Attach a session ID to all log events.
    """

    return logger.bind(session_id=session_id)


def set_log_level(level: str) -> None:
    """
    Update the global log level.

    Parameters
    ----------
    level:
        DEBUG
        INFO
        WARNING
        ERROR
        CRITICAL
    """

    logging.getLogger().setLevel(level.upper())