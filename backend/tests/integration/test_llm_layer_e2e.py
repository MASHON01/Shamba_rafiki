"""
Integration test for the hardened LLM layer (, ).

Proves the whole stack works together through the orchestrator,
on fakes and offline: the resilience + cache client chain (/8),
per-intent generation presets, streaming, graceful
degradation, and the extended /health surface (/8/9).

The point is composition: each piece was unit-tested in its own output;
here they run as one wired system, the way the app assembles them via
build_llm_client.
"""

from __future__ import annotations

import pytest
from app.models.request import QueryRequest
from app.orchestration.dispatcher import Dispatcher
from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.llm.factory import build_llm_client
from app.orchestration.llm.llama_client import LLMConnectionError
from app.orchestration.llm.streaming import StreamEvent
from app.orchestration.orchestrator import Orchestrator

pytestmark = pytest.mark.integration


class _RecordingLLM(BaseLLMClient):
    """A fake that records how it was called and streams word-by-word."""

    supports_streaming = True

    def __init__(self):
        self.gen_calls = 0
        self.stream_calls = 0
        self.configs = []
        self._answer = (
            "Apply mancozeb fungicide at 40 g per 20 litres and rotate maize " "with beans."
        )

    def generate(self, prompt, config=None):
        self.gen_calls += 1
        self.configs.append(config)
        return GenerationResult(
            text=self._answer, prompt_tokens=100, completion_tokens=16, latency_ms=20
        )

    def generate_stream(self, prompt, config=None, cancel_event=None):
        self.stream_calls += 1
        self.configs.append(config)
        for i, word in enumerate(self._answer.split(" ")):
            yield StreamEvent(content=word if i == 0 else " " + word)
        yield StreamEvent(done=True, prompt_tokens=100, completion_tokens=16)

    def health(self):
        return True


def _orchestrator(inner, retriever):
    """Wrap `inner` in the real hardened chain, behind the orchestrator."""
    client = build_llm_client(inner)
    return Orchestrator(
        llm_client=client,
        dispatcher=Dispatcher(llm_client=client, retriever=retriever),
    )

    # ---------------------------------------------------------------------------
    # Blocking path: chain + presets + cache
    # ---------------------------------------------------------------------------


def test_blocking_query_grounded_through_full_chain(retriever):
    inner = _RecordingLLM
    orch = _orchestrator(inner, retriever)
    resp = orch.handle_query(
        QueryRequest(query="How do I treat maize blight in Nakuru?", language="en", session_id="s1")
    )
    assert resp["success"]
    assert resp["data"]["answer"]
    assert resp["data"]["grounded"] is True
    assert len(resp["data"]["sources"]) >= 1
    # Diagnosis intent -> cold factual preset reached the model.
    assert inner.configs[0].temperature == 0.20


def test_identical_query_is_cache_served(retriever):
    inner = _RecordingLLM
    orch = _orchestrator(inner, retriever)
    q = QueryRequest(query="How do I treat maize blight?", language="en")
    orch.handle_query(q)
    orch.handle_query(QueryRequest(query="How do I treat maize blight?", language="en"))
    # Second identical query served from cache: inner generated once.
    assert inner.gen_calls == 1

    # ---------------------------------------------------------------------------
    # Streaming path through the chain
    # ---------------------------------------------------------------------------


def test_streaming_query_tokens_then_done(retriever):
    inner = _RecordingLLM
    orch = _orchestrator(inner, retriever)
    messages = list(
        orch.handle_query_stream(QueryRequest(query="How do I treat maize blight?", language="en"))
    )
    kinds = [m.event for m in messages]
    assert kinds.count("token") >= 1 and kinds[-1] == "done"
    streamed = "".join(m.data["text"] for m in messages if m.event == "token")
    assert "mancozeb" in streamed
    assert messages[-1].data["grounded"] is True
    assert messages[-1].data["partial"] is False


def test_stream_cache_hit_after_blocking(retriever):
    inner = _RecordingLLM
    orch = _orchestrator(inner, retriever)
    q = "How do I treat maize blight?"
    orch.handle_query(QueryRequest(query=q, language="en"))  # fills cache
    list(orch.handle_query_stream(QueryRequest(query=q, language="en")))  # hit
    # The stream was served from cache: inner never streamed.
    assert inner.stream_calls == 0

    # ---------------------------------------------------------------------------
    # Resilience: graceful degradation through the chain
    # ---------------------------------------------------------------------------


class _DeadLLM(BaseLLMClient):
    def generate(self, prompt, config=None):
        raise LLMConnectionError("llama-server down")

    def health(self):
        return False


def test_degrades_cleanly_when_model_down(retriever):
    orch = _orchestrator(_DeadLLM, retriever)
    resp = orch.handle_query(
        QueryRequest(query="How do I treat maize blight in Nakuru?", language="en")
    )
    assert not resp["success"]
    assert resp["error"]["code"] == "LLM_UNAVAILABLE"
    # Retrieval-only degradation still hands back the reference material.
    assert resp["error"]["details"]["retrieval_only"] is True
    assert len(resp["error"]["details"]["sources"]) >= 1

    # ---------------------------------------------------------------------------
    # /health surface: model + cache + config
    # ---------------------------------------------------------------------------


def test_health_reports_model_cache_and_config(retriever):
    inner = _RecordingLLM
    orch = _orchestrator(inner, retriever)
    orch.handle_query(QueryRequest(query="How do I treat maize blight?", language="en"))
    health = orch.health["data"]

    assert health["llm_available"] is True
    assert health["model"]["configured"] is True
    assert "hit_rate" in health["cache"]  # cache metrics present
    cfg = health["config"]
    assert cfg["prompt_version"] and cfg["model_id"]
    assert "response_cache_enabled" in cfg
