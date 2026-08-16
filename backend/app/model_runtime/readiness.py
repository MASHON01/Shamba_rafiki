"""
Server readiness probe.

Starting llama-server returns immediately, but the model is not resident
and answering for some seconds afterward (loading and mmapping a
multi-GB GGUF on a CPU-only 8 GB machine is not instant). Sending a real
query before then just fails. This module answers one question: is the
server actually up and serving *yet*?

It polls the server's health endpoint on a fixed interval until it
succeeds or a timeout elapses, and reports how long that took (a useful
"cold start" number for the reports). The default probe reuses the
 ``LlamaClient.health`` so there is exactly one HTTP path to
llama-server, not a second, subtly-different one. The probe is injectable
so tests need no real server and callers can supply their own check.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from app.config.constants import (
    MODEL_SERVER_POLL_INTERVAL_S,
    MODEL_SERVER_STARTUP_TIMEOUT_S,
)
from app.utils.logger import get_logger

logger = get_logger("Readiness")

# A probe returns True when the server is ready, False otherwise, and
# never raises - transient connection errors are just "not ready yet".
ProbeFn = Callable[[], bool]


@dataclass(slots=True)
class ReadinessResult:
    """Outcome of a readiness wait."""

    ready: bool
    elapsed_s: float
    attempts: int

    def __bool__(self) -> bool:
        return self.ready


def default_probe(server_url: str) -> ProbeFn:
    """
    A probe backed by ``LlamaClient.health`` (reuses 's one HTTP
    path). Imported lazily so this module stays importable without the
    LLM stack wired up.
    """

    def _probe() -> bool:
        from app.orchestration.llm.llama_client import LlamaClient

        try:
            return LlamaClient(server_url=server_url).health
        except Exception:  # noqa: BLE001 - a failed probe is just "not ready".
            return False

    return _probe


def wait_for_ready(
    server_url: str,
    timeout_s: float | None = None,
    interval_s: float | None = None,
    probe: ProbeFn | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ReadinessResult:
    """
    Poll until the server is ready or ``timeout_s`` elapses.

    Parameters
    ----------
    server_url:
        Base URL of llama-server (used by the default probe).
    timeout_s:
        Total budget. Defaults to ``MODEL_SERVER_STARTUP_TIMEOUT_S``.
    interval_s:
        Delay between polls. Defaults to ``MODEL_SERVER_POLL_INTERVAL_S``.
    probe:
        Readiness check. Defaults to a LlamaClient-backed health probe.
    sleep, clock:
        Injectable for deterministic tests; leave at defaults in prod.

    Returns
    -------
    ReadinessResult
        ``ready`` plus how long it took and how many probes it ran. Always
        attempts at least once, even with a zero timeout.
    """
    timeout = MODEL_SERVER_STARTUP_TIMEOUT_S if timeout_s is None else timeout_s
    interval = MODEL_SERVER_POLL_INTERVAL_S if interval_s is None else interval_s
    check = probe or default_probe(server_url)

    start = clock
    attempts = 0

    while True:
        attempts += 1
        if check:
            elapsed = clock - start
            logger.info(
                "readiness.ready",
                url=server_url,
                elapsed_s=round(elapsed, 2),
                attempts=attempts,
            )
            return ReadinessResult(True, elapsed, attempts)

        elapsed = clock - start
        if elapsed >= timeout:
            logger.warning(
                "readiness.timeout",
                url=server_url,
                elapsed_s=round(elapsed, 2),
                attempts=attempts,
            )
            return ReadinessResult(False, elapsed, attempts)

        sleep(interval)
