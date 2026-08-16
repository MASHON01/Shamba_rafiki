"""
Unit tests for LLM resilience: timeouts, retries, circuit
breaker, the resilient client wrapper, and orchestrator degradation.

All offline and deterministic - clocks and sleeps are injected, and the
"server" is a counting fake.
"""

from __future__ import annotations

import pytest
from app.models.request import QueryRequest
from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.llm.llama_client import LLMConnectionError, LLMError
from app.orchestration.llm.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ResilientLLMClient,
    RetryPolicy,
    retry_call,
)
from app.orchestration.llm.streaming import StreamEvent
from app.orchestration.llm.timeouts import TimeoutPolicy

pytestmark = pytest.mark.unit


class _Prompt:
    full_prompt = "SYSTEM\n\nQuestion: hi"

    # ---------------------------------------------------------------------------
    # Timeout policy
    # ---------------------------------------------------------------------------


def test_timeout_policy_tuple():
    assert TimeoutPolicy(5, 120).as_requests_timeout == (5, 120)


def test_timeout_policies_from_settings():
    gen = TimeoutPolicy.for_generation
    health = TimeoutPolicy.for_health
    # Health read is shorter than generation read (a ping vs a generation).
    assert health.read_s <= gen.read_s
    assert gen.connect_s == health.connect_s

    # ---------------------------------------------------------------------------
    # retry_call
    # ---------------------------------------------------------------------------


def _policy(attempts=3):
    return RetryPolicy(
        max_attempts=attempts, base_delay_s=0.01, max_delay_s=0.02, multiplier=2, jitter=False
    )


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMConnectionError("blip")
        return "ok"

    result = retry_call(fn, policy=_policy(3), sleep=lambda _s: None)
    assert result == "ok" and calls["n"] == 3


def test_retry_exhausts_and_raises():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise LLMConnectionError("still down")

    with pytest.raises(LLMConnectionError):
        retry_call(fn, policy=_policy(3), sleep=lambda _s: None)
    assert calls["n"] == 3


def test_retry_does_not_retry_non_transient():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise LLMError("malformed response")  # not a connection error

    with pytest.raises(LLMError):
        retry_call(fn, policy=_policy(3), sleep=lambda _s: None)
    assert calls["n"] == 1  # raised immediately, no retries


def test_backoff_grows_and_caps():
    p = RetryPolicy(max_attempts=5, base_delay_s=1.0, max_delay_s=3.0, multiplier=2, jitter=False)
    assert p.delay_for(1) == 1.0
    assert p.delay_for(2) == 2.0
    assert p.delay_for(3) == 3.0  # capped (would be 4.0)
    assert p.delay_for(4) == 3.0

    # ---------------------------------------------------------------------------
    # Circuit breaker
    # ---------------------------------------------------------------------------


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_opens_after_threshold_and_fails_fast():
    clock = _Clock
    cb = CircuitBreaker(failure_threshold=2, reset_timeout_s=10, clock=clock)
    assert cb.state == CircuitState.CLOSED

    cb.record_failure()
    cb.allow()  # still closed after one failure
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        cb.allow()


def test_breaker_half_opens_then_closes_on_success():
    clock = _Clock
    cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=10, clock=clock)
    cb.record_failure()  # opens immediately (threshold 1)
    assert cb.state == CircuitState.OPEN

    clock.t = 10  # cooldown elapsed
    assert cb.state == CircuitState.HALF_OPEN
    cb.allow()  # a trial is permitted
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_breaker_reopens_on_failed_trial():
    clock = _Clock
    cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=5, clock=clock)
    cb.record_failure()
    clock.t = 5
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_failure()  # trial failed
    assert cb.state == CircuitState.OPEN

    # ---------------------------------------------------------------------------
    # ResilientLLMClient
    # ---------------------------------------------------------------------------


class _FlakyLLM(BaseLLMClient):
    """Fails its first ``fail_times`` generate calls, then succeeds."""

    supports_streaming = True

    def __init__(self, fail_times=0, always_fail=False):
        self.calls = 0
        self._fail_times = fail_times
        self._always = always_fail

    def generate(self, prompt, config=None):
        self.calls += 1
        if self._always or self.calls <= self._fail_times:
            raise LLMConnectionError("down")
        return GenerationResult(text="ok", completion_tokens=1)

    def generate_stream(self, prompt, config=None, cancel_event=None):
        self.calls += 1
        if self._always:
            raise LLMConnectionError("down")
        yield StreamEvent(content="ok", done=True, completion_tokens=1)

    def health(self):
        return not self._always


def _resilient(inner, attempts=3, threshold=2):
    return ResilientLLMClient(
        inner,
        retry_policy=RetryPolicy(
            max_attempts=attempts, base_delay_s=0.0, max_delay_s=0.0, multiplier=1, jitter=False
        ),
        breaker=CircuitBreaker(failure_threshold=threshold, reset_timeout_s=999),
    )


def test_resilient_generate_retries_then_succeeds():
    inner = _FlakyLLM(fail_times=1)
    client = _resilient(inner)
    result = client.generate(_Prompt)
    assert result.text == "ok"
    assert inner.calls == 2  # one failure retried
    assert client.circuit_state == CircuitState.CLOSED


def test_resilient_opens_circuit_then_fails_fast():
    inner = _FlakyLLM(always_fail=True)
    client = _resilient(inner, attempts=1, threshold=2)

    # Two failed calls (attempts=1 each) trip the breaker (threshold=2).
    for _ in range(2):
        with pytest.raises(LLMError):
            client.generate(_Prompt)
    assert client.circuit_state == CircuitState.OPEN

    calls_before = inner.calls
    with pytest.raises(CircuitOpenError):
        client.generate(_Prompt)
    assert inner.calls == calls_before  # failed fast, inner NOT called


def test_resilient_stream_passthrough_and_failure_records():
    ok = _resilient(_FlakyLLM)
    events = list(ok.generate_stream(_Prompt))
    assert events[-1].done is True
    assert ok.circuit_state == CircuitState.CLOSED

    bad_inner = _FlakyLLM(always_fail=True)
    bad = _resilient(bad_inner, threshold=1)
    with pytest.raises(LLMConnectionError):
        list(bad.generate_stream(_Prompt))
    assert bad.circuit_state == CircuitState.OPEN


def test_supports_streaming_delegates():
    assert _resilient(_FlakyLLM).supports_streaming is True

    # ---------------------------------------------------------------------------
    # Orchestrator graceful degradation carries retrieved sources
    # ---------------------------------------------------------------------------


class _DeadLLM(BaseLLMClient):
    def generate(self, prompt, config=None):
        raise LLMConnectionError("llama-server down")

    def health(self):
        return False


def test_degradation_includes_retrieved_sources(retriever):
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.orchestrator import Orchestrator

    orch = Orchestrator(
        llm_client=_DeadLLM,
        dispatcher=Dispatcher(llm_client=_DeadLLM, retriever=retriever),
    )
    resp = orch.handle_query(
        QueryRequest(query="How do I treat maize blight in Nakuru?", language="en")
    )
    assert not resp["success"]
    assert resp["error"]["code"] == "LLM_UNAVAILABLE"
    details = resp["error"]["details"]
    assert details["retrieval_only"] is True
    assert len(details["sources"]) >= 1  # retrieval-only degradation


def test_degradation_without_retriever_has_no_sources(fake_llm):
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.orchestrator import Orchestrator

    orch = Orchestrator(
        llm_client=_DeadLLM,
        dispatcher=Dispatcher(llm_client=_DeadLLM, retriever=None),
    )
    resp = orch.handle_query(QueryRequest(query="hello", language="en"))
    assert resp["error"]["code"] == "LLM_UNAVAILABLE"
    assert resp["error"]["details"]["retrieval_only"] is False
    assert resp["error"]["details"]["sources"] == []
