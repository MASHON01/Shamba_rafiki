"""
Application Dependency Injection Container.

Responsible for creating and resolving singleton
services across the application.
"""

from __future__ import annotations

from typing import Any, Callable

from app.core.exceptions import DependencyNotFoundError


class DependencyContainer:
    """
    Lightweight singleton dependency container.

    Services are lazily instantiated the first time
    they are requested.
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Any]] = {}
        self._instances: dict[str, Any] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], Any],
    ) -> None:
        """
        Register a service factory.
        """
        self._factories[name] = factory

    def resolve(self, name: str) -> Any:
        """
        Resolve a singleton service.

        If it hasn't been created yet,
        instantiate and cache it.
        """

        if name in self._instances:
            return self._instances[name]

        if name not in self._factories:
            raise DependencyNotFoundError(
                f"Dependency '{name}' not registered."
            )

        instance = self._factories[name]()
        self._instances[name] = instance

        return instance

    def exists(self, name: str) -> bool:
        return (
            name in self._factories
            or name in self._instances
        )

    def clear(self) -> None:
        """
        Clear all instantiated singletons.

        Registered factories remain intact.
        """
        self._instances.clear()

    def registered(self) -> list[str]:
        return sorted(self._factories.keys())


container = DependencyContainer()