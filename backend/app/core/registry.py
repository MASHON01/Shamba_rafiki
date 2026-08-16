"""
Simple in-memory service registry.

Acts as the application's dependency lookup table.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import RegistryError


class ServiceRegistry:
    """
    Stores singleton instances used across the application.
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def register(self, name: str, instance: Any) -> None:
        """
        Register a singleton service.
        """
        if name in self._services:
            raise RegistryError(
                f"Service '{name}' already registered."
            )

        self._services[name] = instance

    def get(self, name: str) -> Any:
        """
        Retrieve a registered service.
        """
        if name not in self._services:
            raise RegistryError(
                f"Service '{name}' is not registered."
            )

        return self._services[name]

    def remove(self, name: str) -> None:
        """
        Remove a service.
        """
        self._services.pop(name, None)

    def exists(self, name: str) -> bool:
        return name in self._services

    def clear(self) -> None:
        self._services.clear()

    def list_services(self) -> list[str]:
        return sorted(self._services.keys())


registry = ServiceRegistry()