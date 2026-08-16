"""
Exact-match LLM response cache.

The safest cache is an exact one: if the *identical* prompt (same system
prompt, same retrieved context, same question, same history) is generated
again with the *identical* sampling config, the answer can only be the
same useful answer - so return the stored one instead of paying for
generation again. Two farmers asking the same question at the same kiosk,
or a repeated tap, cost one generation, not two.

This is deliberately conservative. The key is the whole flattened prompt
plus the sampling signature; any difference in context, phrasing, or
config is a different key and a fresh generation. Nothing "similar" is
ever served - only identical. A pleasant side effect: identical inputs
become deterministic by reuse, which is exactly what you want from an
advisor.

``CachingLLMClient`` wraps any backend and is meant to sit OUTERMOST in
the client chain (outside resilience), so a hit costs no server call at
all. On the streaming path a hit replays the cached text as a single
chunk; a miss streams normally and caches only once the stream has fully
completed (never a partial answer).
"""

from __future__ import annotations

import hashlib
from typing import Iterator

from app.orchestration.cache.cache_policy import BoundedLRUCache, CacheStats
from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.llm.generation_config import GenerationConfig
from app.orchestration.llm.stream_assembler import StreamAssembler
from app.orchestration.llm.streaming import StreamEvent
from app.utils.logger import get_logger

logger = get_logger("ResponseCache")


def _config_signature(config: GenerationConfig | None) -> str:
    """A stable string for the sampling params that affect the output."""
    if config is None:
        return "default"
    return (
        f"t={config.temperature};p={config.top_p};k={config.top_k};"
        f"n={config.max_tokens};rp={config.repeat_penalty};"
        f"seed={config.seed};stop={config.stop}"
    )


class ResponseCache:
    """Keys prompts+configs to generated results over a bounded LRU."""

    def __init__(self, cache: BoundedLRUCache | None = None) -> None:
        self._cache = cache or BoundedLRUCache()

    @staticmethod
    def make_key(prompt_text: str, config: GenerationConfig | None) -> str:
        raw = f"{prompt_text}\x00{_config_signature(config)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, prompt_text: str, config) -> GenerationResult | None:
        return self._cache.get(self.make_key(prompt_text, config))

    def put(self, prompt_text: str, config, result: GenerationResult) -> None:
        key = self.make_key(prompt_text, config)
        # Size ~ the answer text (the only large field); a small floor so
        # empty answers still count against the entry cap.
        size = len(result.text.encode("utf-8")) + 64
        self._cache.put(key, result, size)

    def stats(self) -> CacheStats:
        return self._cache.stats

    def clear(self) -> None:
        self._cache.clear()


class CachingLLMClient(BaseLLMClient):
    """
    Decorates a backend with an exact-match response cache.

    Place outermost: CachingLLMClient(ResilientLLMClient(LlamaClient)).
    A cache hit returns immediately without touching the inner client.
    """

    def __init__(self, inner: BaseLLMClient, cache: ResponseCache | None = None) -> None:
        self._inner = inner
        self._cache = cache or ResponseCache()

    @property
    def supports_streaming(self) -> bool:
        return getattr(self._inner, "supports_streaming", False)

    def stats(self) -> CacheStats:
        return self._cache.stats

    def generate(self, prompt, config=None) -> GenerationResult:
        cached = self._cache.get(prompt.full_prompt, config)
        if cached is not None:
            logger.info("cache.hit", mode="generate")
            return cached
        result = self._inner.generate(prompt, config)
        self._cache.put(prompt.full_prompt, config, result)
        return result

    def generate_stream(self, prompt, config=None, cancel_event=None) -> Iterator[StreamEvent]:
        cached = self._cache.get(prompt.full_prompt, config)
        if cached is not None:
            logger.info("cache.hit", mode="stream")
            # Replay the cached answer as one chunk + a final done event.
            yield StreamEvent(content=cached.text)
            yield StreamEvent(
                done=True,
                prompt_tokens=cached.prompt_tokens,
                completion_tokens=cached.completion_tokens,
            )
            return

        assembler = StreamAssembler
        for event in self._inner.generate_stream(prompt, config, cancel_event=cancel_event):
            assembler.feed(event)
            yield event

            # Cache only a fully-completed stream, never a partial/cancelled one.
        if assembler.done and assembler.text:
            self._cache.put(prompt.full_prompt, config, assembler.finalize)

    def health(self) -> bool:
        return self._inner.health
