"""
Unit tests for LLM caching & warm-path optimization.

Bounded LRU eviction, exact-match keying, the caching client wrapper
(generate + stream, hit/miss, no-cache-on-partial), warm-up tolerance,
and stats - all offline.
"""

from __future__ import annotations

import pytest
from app.orchestration.cache import (
    BoundedLRUCache,
    CachePolicy,
    CachingLLMClient,
    ResponseCache,
    warmup,
)
from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.llm.generation_config import GenerationConfig
from app.orchestration.llm.llama_client import LLMConnectionError
from app.orchestration.llm.streaming import StreamEvent

pytestmark = pytest.mark.unit


class _Prompt:
    def __init__(self, text="SYSTEM\n\nQuestion: hi"):
        self.full_prompt = text

        # ---------------------------------------------------------------------------
        # BoundedLRUCache
        # ---------------------------------------------------------------------------


def test_lru_evicts_by_entry_count():
    cache = BoundedLRUCache(CachePolicy(max_entries=2, max_bytes=10_000))
    cache.put("a", 1, 10)
    cache.put("b", 2, 10)
    cache.get("a")  # 'a' now most-recent
    cache.put("c", 3, 10)  # evicts LRU -> 'b'
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3
    assert cache.stats.evictions == 1


def test_lru_evicts_by_bytes():
    cache = BoundedLRUCache(CachePolicy(max_entries=100, max_bytes=100))
    cache.put("a", 1, 60)
    cache.put("b", 2, 60)  # total 120 > 100 -> evict 'a'
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.stats.bytes <= 100


def test_stats_hits_and_misses():
    cache = BoundedLRUCache(CachePolicy(max_entries=10, max_bytes=10_000))
    cache.put("a", 1, 10)
    cache.get("a")  # hit
    cache.get("z")  # miss
    s = cache.stats
    assert s.hits == 1 and s.misses == 1 and s.hit_rate == 0.5

    # ---------------------------------------------------------------------------
    # ResponseCache keying
    # ---------------------------------------------------------------------------


def test_key_sensitive_to_prompt_and_config():
    rc = ResponseCache
    k1 = rc.make_key("prompt A", GenerationConfig(temperature=0.2))
    k2 = rc.make_key("prompt B", GenerationConfig(temperature=0.2))
    k3 = rc.make_key("prompt A", GenerationConfig(temperature=0.9))
    assert k1 != k2  # different prompt
    assert k1 != k3  # different config
    assert k1 == rc.make_key("prompt A", GenerationConfig(temperature=0.2))

    # ---------------------------------------------------------------------------
    # CachingLLMClient - generate
    # ---------------------------------------------------------------------------


class _CountingLLM(BaseLLMClient):
    supports_streaming = True

    def __init__(self):
        self.gen_calls = 0
        self.stream_calls = 0

    def generate(self, prompt, config=None):
        self.gen_calls += 1
        return GenerationResult(
            text=f"answer {self.gen_calls}", prompt_tokens=10, completion_tokens=2
        )

    def generate_stream(self, prompt, config=None, cancel_event=None):
        self.stream_calls += 1
        yield StreamEvent(content="answer ")
        yield StreamEvent(content="streamed")
        yield StreamEvent(done=True, prompt_tokens=10, completion_tokens=2)

    def health(self):
        return True


def test_generate_caches_identical_calls():
    inner = _CountingLLM
    client = CachingLLMClient(inner)
    r1 = client.generate(_Prompt)
    r2 = client.generate(_Prompt)
    assert r1.text == r2.text  # same answer served
    assert inner.gen_calls == 1  # inner called once
    assert client.stats.hits == 1


def test_generate_miss_on_different_config():
    inner = _CountingLLM
    client = CachingLLMClient(inner)
    client.generate(_Prompt, GenerationConfig(temperature=0.2))
    client.generate(_Prompt, GenerationConfig(temperature=0.9))
    assert inner.gen_calls == 2  # different config -> not a hit

    # ---------------------------------------------------------------------------
    # CachingLLMClient - streaming
    # ---------------------------------------------------------------------------


def test_stream_caches_then_replays():
    inner = _CountingLLM
    client = CachingLLMClient(inner)

    first = list(client.generate_stream(_Prompt))
    assert "".join(e.content for e in first) == "answer streamed"
    assert inner.stream_calls == 1

    # Second stream is a cache hit: inner NOT streamed again, text replayed.
    second = list(client.generate_stream(_Prompt))
    assert inner.stream_calls == 1
    assert "".join(e.content for e in second) == "answer streamed"
    assert second[-1].done is True


def test_generate_reuses_streamed_cache():
    inner = _CountingLLM
    client = CachingLLMClient(inner)
    list(client.generate_stream(_Prompt))  # populates cache
    result = client.generate(_Prompt)  # same key -> hit
    assert inner.gen_calls == 0
    assert result.text == "answer streamed"


class _PartialLLM(BaseLLMClient):
    supports_streaming = True

    def generate_stream(self, prompt, config=None, cancel_event=None):
        yield StreamEvent(content="half an ")
        yield StreamEvent(content="answer")
        # No done event: an interrupted / partial stream.

    def generate(self, prompt, config=None):
        return GenerationResult(text="full answer")

    def health(self):
        return True


def test_partial_stream_not_cached():
    inner = _PartialLLM
    client = CachingLLMClient(inner)
    list(client.generate_stream(_Prompt))  # partial, must not cache
    assert client.stats.entries == 0
    # A following generate is therefore a miss (not served from cache).
    assert client.generate(_Prompt).text == "full answer"

    # ---------------------------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------------------------


def test_warmup_success():
    assert warmup(_CountingLLM) is True


def test_warmup_tolerates_dead_server():
    class _Dead(BaseLLMClient):
        def generate(self, prompt, config=None):
            raise LLMConnectionError("server down")

        def health(self):
            return False

    assert warmup(_Dead) is False
