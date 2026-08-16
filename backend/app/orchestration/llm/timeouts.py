"""
Timeout policy for llama-server calls.

A single timeout number can't express what this system needs. Connecting
to a dead server should fail *fast* - if llama-server isn't listening, we
want to know in a second or two, not wait two minutes. But once
connected, a real generation on a CPU-only 8 GB machine can legitimately
take a while, so the *read* timeout has to be generous.

``TimeoutPolicy`` splits the two and hands ``requests`` the ``(connect,
read)`` tuple it already understands. Generation and health use different
policies: health is a cheap ping, so its read timeout is short too.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import settings


@dataclass(frozen=True, slots=True)
class TimeoutPolicy:
    """Connect and read timeouts, in seconds."""

    connect_s: float
    read_s: float

    def as_requests_timeout(self) -> tuple[float, float]:
        """The ``(connect, read)`` tuple the requests library expects."""
        return (self.connect_s, self.read_s)

    @classmethod
    def for_generation(cls) -> "TimeoutPolicy":
        """Fast connect, generous read - a real generation may be slow."""
        return cls(
            settings.LLM_CONNECT_TIMEOUT_SECONDS,
            settings.LLM_READ_TIMEOUT_SECONDS,
        )

    @classmethod
    def for_health(cls) -> "TimeoutPolicy":
        """Fast connect and fast read - health is a cheap ping."""
        return cls(
            settings.LLM_CONNECT_TIMEOUT_SECONDS,
            settings.LLM_HEALTH_TIMEOUT_SECONDS,
        )
