"""
LLM client factory - the one place the hardened client chain is composed.

 built the inference layer as a stack of decorators around the raw
llama-server client, each adding one capability:

    LlamaClient raw HTTP + streaming + typed errors (/4)
      wrapped by
    ResilientLLMClient retries + circuit breaker
      wrapped by
    CachingLLMClient exact-match response cache

Order matters: caching is outermost (a hit costs no server call, so it
sits outside resilience), resilience is inside it (retries/breaker guard
the actual HTTP), and the raw client is innermost. This function is the
single, documented source of that composition, driven by settings, so
the app, scripts, and tests all build the same chain the same way instead
of re-assembling it by hand.
"""

from __future__ import annotations

from app.config.settings import settings
from app.orchestration.cache import CachingLLMClient
from app.orchestration.llm.base import BaseLLMClient
from app.orchestration.llm.llama_client import LlamaClient
from app.orchestration.llm.resilience import ResilientLLMClient


def build_llm_client(inner: BaseLLMClient | None = None) -> BaseLLMClient:
    """
    Compose the hardened LLM client chain from settings.

    ``inner`` defaults to a fresh ``LlamaClient``; pass a fake to build the
    same resilience + cache chain around a test double. Resilience is
    always applied; the cache layer is added only when
    ``LLM_RESPONSE_CACHE_ENABLED`` is set.
    """
    client: BaseLLMClient = ResilientLLMClient(inner or LlamaClient())
    if settings.LLM_RESPONSE_CACHE_ENABLED:
        client = CachingLLMClient(client)
    return client
