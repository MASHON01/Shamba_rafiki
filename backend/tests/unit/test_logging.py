"""Unit tests for logging configuration and structured output."""

from __future__ import annotations

import pytest
import structlog

from app.config.logging import configure_logging
from app.utils.logger import bind_component, bind_request_id, get_logger

pytestmark = pytest.mark.unit


def test_get_logger_returns_bound_logger():
    logger = get_logger("test_component")
    # structlog loggers expose the standard level methods
    assert hasattr(logger, "info")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


def test_configure_logging_runs_idempotently():
    # Should be safe to call more than once (app + tests both call it).
    configure_logging()
    configure_logging()
    logger = get_logger("x")
    assert logger is not None


def test_structured_logging_captures_event_and_kwargs():
    """Structured logs should carry the event name plus bound key/values.
    Uses structlog's capture to assert on the emitted entries."""
    from structlog.testing import capture_logs

    configure_logging()
    logger = get_logger("capture_test")

    with capture_logs() as entries:
        logger.info("ingestion.completed", documents=3, chunks=42)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["event"] == "ingestion.completed"
    assert entry["documents"] == 3
    assert entry["chunks"] == 42
    assert entry["log_level"] == "info"


def test_bind_helpers_attach_context():
    """bind_request_id / bind_component should attach fields that then
    appear on subsequent log entries."""
    from structlog.testing import capture_logs

    configure_logging()
    logger = get_logger("bind_test")
    logger = bind_request_id(logger, "req-123")
    logger = bind_component(logger, "Retriever")

    with capture_logs() as entries:
        logger.info("did.something")

    entry = entries[0]
    assert entry.get("request_id") == "req-123"
    assert entry.get("component") == "Retriever"


def test_warning_and_error_levels_recorded():
    from structlog.testing import capture_logs

    configure_logging()
    logger = get_logger("levels_test")
    with capture_logs() as entries:
        logger.warning("something.odd")
        logger.error("something.bad")

    levels = {e["log_level"] for e in entries}
    assert "warning" in levels and "error" in levels