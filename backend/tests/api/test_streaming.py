"""
Tests for streaming & response handling.

Covers the SSE transport parser, the assembler, the client stream path,
the /chat/stream endpoint over a TestClient, cancellation, and the
fallback to blocking generation - all offline (no llama-server).
"""

from __future__ import annotations

import json
import threading
import types

import pytest
from app.models.request import QueryRequest
from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.llm.llama_client import LlamaClient
from app.orchestration.llm.stream_assembler import StreamAssembler
from app.orchestration.llm.streaming import StreamEvent, iter_completion_stream

pytestmark = pytest.mark.api


# ---------------------------------------------------------------------------
# Fake SSE transport
# ---------------------------------------------------------------------------


def _fake_requests(lines: list[bytes], status: int = 200):
    """A stand-in `requests` whose POST returns a canned SSE line stream."""

    class _Resp:
        status_code = status

        def iter_lines(self):
            yield from lines

        def close(self):
            pass

    def post(url, json=None, timeout=None, stream=False):
        return _Resp

    exceptions = types.SimpleNamespace(
        Timeout=type("Timeout", (Exception,), {}),
        ConnectionError=type("ConnectionError", (Exception,), {}),
        RequestException=type("RequestException", (Exception,), {}),
    )
    return types.SimpleNamespace(post=post, exceptions=exceptions)


_SSE_LINES = [
    b'data: {"content": "Apply", "stop": false}',
    b"",
    b'data: {"content": " mancozeb", "stop": false}',
    b'data: {"content": "", "stop": true, "tokens_evaluated": 40, "tokens_predicted": 2}',
]


def test_iter_completion_stream_parses_deltas_and_final():
    events = list(
        iter_completion_stream(_fake_requests(_SSE_LINES), "http://x/completion", {}, timeout=5)
    )
    contents = [e.content for e in events if e.content]
    assert contents == ["Apply", " mancozeb"]
    final = events[-1]
    assert final.done is True
    assert final.prompt_tokens == 40 and final.completion_tokens == 2


def test_iter_completion_stream_honours_cancel():
    cancel = threading.Event
    cancel.set()
    events = list(
        iter_completion_stream(
            _fake_requests(_SSE_LINES),
            "http://x/completion",
            {},
            timeout=5,
            cancel_event=cancel,
        )
    )
    assert events == []


def test_iter_completion_stream_non_200_raises():
    from app.orchestration.llm.llama_client import LLMError

    with pytest.raises(LLMError):
        list(
            iter_completion_stream(
                _fake_requests([], status=503),
                "http://x/completion",
                {},
                timeout=5,
            )
        )

        # ---------------------------------------------------------------------------
        # Assembler
        # ---------------------------------------------------------------------------


def test_assembler_builds_text_and_counts():
    asm = StreamAssembler
    for e in [
        StreamEvent(content="Apply"),
        StreamEvent(content=" mancozeb"),
        StreamEvent(done=True, prompt_tokens=40, completion_tokens=2),
    ]:
        asm.feed(e)
    result = asm.finalize(latency_ms=123)
    assert result.text == "Apply mancozeb"
    assert result.completion_tokens == 2
    assert result.latency_ms == 123
    assert asm.done is True


def test_assembler_partial_counts_pieces():
    asm = StreamAssembler
    asm.feed(StreamEvent(content="Apply"))
    asm.feed(StreamEvent(content=" mancozeb"))
    asm.mark_interrupted()  # never got a done event
    result = asm.finalize
    assert result.text == "Apply mancozeb"
    assert asm.interrupted is True
    assert result.completion_tokens == 2  # fell back to piece count

    # ---------------------------------------------------------------------------
    # Client stream path
    # ---------------------------------------------------------------------------


def test_client_generate_stream(monkeypatch):
    client = LlamaClient(server_url="http://x")
    monkeypatch.setattr(client, "_load_requests", lambda: _fake_requests(_SSE_LINES))

    class _P:
        full_prompt = "SYSTEM\n\nQuestion: hi"

    events = list(client.generate_stream(_P))
    text = "".join(e.content for e in events)
    assert "mancozeb" in text
    assert events[-1].done is True

    # ---------------------------------------------------------------------------
    # SSE endpoint (TestClient)
    # ---------------------------------------------------------------------------


def _collect_sse(response) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event_name = None
    data = None
    for line in response.iter_lines:
        if line == "":
            if event_name is not None:
                events.append((event_name, json.loads(data) if data else None))
            event_name = data = None
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data = line[len("data:") :].strip()
    return events


def test_chat_stream_post_emits_tokens_then_done(api_client):
    with api_client.stream(
        "POST",
        "/chat/stream",
        json={
            "query": "How do I treat maize blight in Nakuru?",
            "language": "en",
            "session_id": "s1",
        },
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        events = _collect_sse(r)

    kinds = [e for e, _ in events]
    assert kinds.count("token") >= 1
    assert kinds[-1] == "done"

    done_data = events[-1][1]
    assert done_data["answer"]
    assert done_data["grounded"] is True
    assert done_data["partial"] is False
    assert len(done_data["sources"]) >= 1


def test_chat_stream_get_eventsource(api_client):
    with api_client.stream(
        "GET",
        "/chat/stream",
        params={"query": "How do I treat maize blight?", "language": "en"},
    ) as r:
        assert r.status_code == 200
        events = _collect_sse(r)
    assert any(e == "token" for e, _ in events)
    assert events[-1][0] == "done"

    # ---------------------------------------------------------------------------
    # Fallback to blocking when streaming is unavailable
    # ---------------------------------------------------------------------------


class _BlockingLLM(BaseLLMClient):
    """A backend that cannot stream - forces the blocking fallback."""

    supports_streaming = False

    def generate(self, prompt, config=None):
        return GenerationResult(
            text="Apply mancozeb fungicide and rotate maize with beans.",
            prompt_tokens=10,
            completion_tokens=8,
            latency_ms=5,
        )

    def health(self):
        return True


def test_stream_falls_back_to_blocking():
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.orchestrator import Orchestrator

    llm = _BlockingLLM
    orch = Orchestrator(llm_client=llm, dispatcher=Dispatcher(llm_client=llm, retriever=None))
    messages = list(
        orch.handle_query_stream(QueryRequest(query="How do I treat maize blight?", language="en"))
    )
    kinds = [m.event for m in messages]
    assert "token" in kinds and kinds[-1] == "done"
    # The single blocking chunk carried the whole answer.
    token_text = "".join(m.data["text"] for m in messages if m.event == "token")
    assert "mancozeb" in token_text
    assert messages[-1].data["answer"]
