"""
LLM caching & warm-path optimization (, ).

Cuts latency and redundant work without threatening the RAM budget:

    ResponseCache / CachingLLMClient exact-match (prompt+config -> answer)
    BoundedLRUCache / CachePolicy size + byte bounds, LRU eviction
    warmup one startup generation, model warm

The cache is exact-match only (safe), bounded and LRU-evicted (never an
OOM risk), and sits outermost in the client chain so a hit costs nothing.
Warm-up pays the cold-start cost before the first farmer does.
"""

from __future__ import annotations

from app.orchestration.cache.cache_policy import (
    BoundedLRUCache,
    CachePolicy,
    CacheStats,
)
from app.orchestration.cache.response_cache import (
    CachingLLMClient,
    ResponseCache,
)
from app.orchestration.cache.warmup import warmup

__all__ = [
    "BoundedLRUCache",
    "CachePolicy",
    "CacheStats",
    "ResponseCache",
    "CachingLLMClient",
    "warmup",
]
