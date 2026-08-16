"""
Application lifecycle management.

Responsible for coordinated startup and shutdown
of every registered component.
"""

from __future__ import annotations

from collections.abc import Callable

from app.utils.logger import get_logger

logger = get_logger(__name__)


class LifecycleManager:
    """
    Coordinates application startup and shutdown.
    """

    def __init__(self) -> None:
        self._startup_tasks: list[Callable[[], None]] = []
        self._shutdown_tasks: list[Callable[[], None]] = []

    def on_startup(self, task: Callable[[], None]) -> None:
        """
        Register startup callback.
        """
        self._startup_tasks.append(task)

    def on_shutdown(self, task: Callable[[], None]) -> None:
        """
        Register shutdown callback.
        """
        self._shutdown_tasks.append(task)

    def startup(self) -> None:
        """
        Execute startup callbacks.
        """
        logger.info("Application startup initiated.")

        for task in self._startup_tasks:
            task()

        logger.info("Application startup completed.")

    def shutdown(self) -> None:
        """
        Execute shutdown callbacks.
        """
        logger.info("Application shutdown initiated.")

        for task in reversed(self._shutdown_tasks):
            task()

        logger.info("Application shutdown completed.")


lifecycle = LifecycleManager()