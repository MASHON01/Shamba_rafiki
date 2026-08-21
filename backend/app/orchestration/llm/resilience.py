"""
Resilience for the LLM layer: retries, backoff, and a circuit breaker.

Three failure modes, three defences:

  - A *transient* blip (a dropped connection, a momentary timeout) should
    be retried a few times with exponential backoff, because it will
    often succeed on the second try.
  - A *persistently dead* server should NOT be retried on every request -
    that just piles latency onto every farmer while the server is down.
    After enough consecutive failures the circuit breaker OPENS and calls
    fail fast, until a cooldown lets one trial request test the water.
  - A *non-transient* error (a malformed response) is a bug, not a blip,
    and is raised immediately - retrying it only wastes time.

``ResilientLLMClient`` wraps any BaseLLMClient with this behaviour, so the
orchestrator and the rest of the pipeline are unchanged: they still call
``generate`` / ``generate_stream`` / ``health`` and still see only typed
LLM errors. Streaming is deliberately NOT retried mid-flight (you can't
un-send half an answer); the breaker still gates it, and the orchestrator
already falls back from a failed stream to a blocking call - which IS
retried.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, TypeVar

from app.config.constants import (
    LLM_RETRY_BACKOFF_MULTIPLIER,
    LLM_RETRY_BASE_DELAY_SECONDS,
    LLM_RETRY_MAX_DELAY_SECONDS,
)
from app.config.settings import settings
from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.llm.llama_client import LLMConnectionError, LLMError
from app.orchestration.llm.streaming import StreamEvent
from app.utils.logger import get_logger

logger = get_logger("LLMResilience")

T = TypeVar("T")


# ===========================================================================
# Retry with exponential backoff
# ===========================================================================


@dataclass(slots=True)
class RetryPolicy:
    """Bounded exponential backoff."""

    max_attempts: int = 3
    base_delay_s: float = LLM_RETRY_BASE_DELAY_SECONDS
    max_delay_s: float = LLM_RETRY_MAX_DELAY_SECONDS
    multiplier: float = LLM_RETRY_BACKOFF_MULTIPLIER
    jitter: bool = True

    @classmethod
    def default(cls) -> "RetryPolicy":
        return cls(max_attempts=settings.LLM_RETRY_MAX_ATTEMPTS)

    def delay_for(self, attempt: int) -> float:
        """
        Backoff before the (attempt+1)-th try. ``attempt`` is 1-based.
        Jitter spreads retries over 50-100% of the computed delay so
        concurrent clients don't retry in lockstep.
        """
        raw = self.base_delay_s * (self.multiplier ** (attempt - 1))
        delay = min(self.max_delay_s, raw)
        if self.jitter:
            delay *= 0.5 + random.random * 0.5
        return delay


def retry_call(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    retryable: tuple[type[Exception], ...] = (LLMConnectionError,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """
    Call ``fn`` with bounded retries on ``retryable`` exceptions.

    Non-retryable exceptions propagate immediately. After the last
    attempt, the final retryable exception is re-raised.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except retryable as exc:
            if attempt >= policy.max_attempts:
                logger.warning("llm.retry.exhausted", attempts=attempt, error=str(exc))
                raise
            delay = policy.delay_for(attempt)
            logger.warning(
                "llm.retry",
                attempt=attempt,
                delay_s=round(delay, 2),
                error=str(exc),
            )
            sleep(delay)

            # ===========================================================================
            # Circuit breaker
            # ===========================================================================


class CircuitState(str, Enum):
    CLOSED = "closed"  # healthy: calls flow through
    OPEN = "open"  # dead: calls fail fast
    HALF_OPEN = "half_open"  # cooldown elapsed: allow one trial


class CircuitOpenError(LLMConnectionError):
    """Raised when the breaker is open and a call is refused fast."""


class CircuitBreaker:
    """
    Trips OPEN after ``failure_threshold`` consecutive failures, then
    fails fast until ``reset_timeout_s`` elapses, at which point one
    HALF_OPEN trial decides whether to close again or re-open.

    Thread-safe: the API runs sync handlers in a threadpool, so several
    requests may touch one breaker at once.
    """

    def __init__(
        self,
        failure_threshold: int,
        reset_timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = failure_threshold
        self._reset_timeout = reset_timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at = 0.0

    @classmethod
    def default(cls) -> "CircuitBreaker":
        return cls(
            failure_threshold=settings.LLM_CIRCUIT_FAILURE_THRESHOLD,
            reset_timeout_s=settings.LLM_CIRCUIT_RESET_SECONDS,
        )

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._current_state

    def _current_state(self) -> CircuitState:
        # Promote OPEN -> HALF_OPEN once the cooldown has passed.
        if (
            self._state == CircuitState.OPEN
            and (self._clock - self._opened_at) >= self._reset_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            logger.info("llm.circuit.half_open")
        return self._state

    def allow(self) -> None:
        """Raise CircuitOpenError if calls are currently refused."""
        with self._lock:
            if self._current_state == CircuitState.OPEN:
                raise CircuitOpenError(
                    "llama-server circuit is open (server appears down); " "failing fast."
                )

    def record_success(self) -> None:
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info("llm.circuit.closed")
            self._failures = 0
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self._threshold:
                if self._state != CircuitState.OPEN:
                    logger.warning("llm.circuit.open", consecutive_failures=self._failures)
                self._state = CircuitState.OPEN
                self._opened_at = self._clock

                # ===========================================================================
                # The resilient client
                # ===========================================================================


class ResilientLLMClient(BaseLLMClient):
    """
    Wraps a BaseLLMClient with retries + a circuit breaker.

    Blocking ``generate`` is retried (transient failures) and gated by the
    breaker. ``generate_stream`` is gated but not retried mid-stream. Both
    record success/failure so the breaker tracks the real server state.
    """

    def __init__(
        self,
        inner: BaseLLMClient,
        retry_policy: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._inner = inner
        self._retry = retry_policy or RetryPolicy.default()
        self._breaker = breaker or CircuitBreaker.default()

    @property
    def supports_streaming(self) -> bool:
        return getattr(self._inner, "supports_streaming", False)

    @property
    def circuit_state(self) -> CircuitState:
        return self._breaker.state

    def generate(self, prompt, config=None) -> GenerationResult:
        self._breaker.allow()
        try:
            result = retry_call(
                lambda: self._inner.generate(prompt, config),
                policy=self._retry,
            )
        except LLMError:
            self._breaker.record_failure()
            raise
        self._breaker.record_success()
        return result

    def generate_stream(self, prompt, config=None, cancel_event=None) -> Iterator[StreamEvent]:
        # Fail fast if the breaker is open; do not retry a stream in flight.
        self._breaker.allow()
        try:
            yield from self._inner.generate_stream(prompt, config, cancel_event=cancel_event)
        except LLMError:
            self._breaker.record_failure()
            raise
        self._breaker.record_success()

    def health(self) -> bool:
        try:
            return self._inner.health()
        except Exception:  # noqa: BLE001 - health never raises.
            return False
