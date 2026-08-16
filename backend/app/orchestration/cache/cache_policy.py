"""
Cache sizing and eviction policy.

A cache that grows without bound is an OOM risk, and on this system an
OOM is a total failure (the 7 GB ceiling). So the cache is bounded on two
axes - a maximum number of entries and a maximum total bytes - and evicts
least-recently-used entries when either limit is reached. Answers are
short strings, so in practice the byte total stays tiny; the point of the
bounds is that a full day of kiosk traffic can never make it otherwise.

``BoundedLRUCache`` is a small, thread-safe, generic string-keyed store
(the API runs sync handlers in a threadpool, so concurrent access is
real). ``response_cache`` builds the LLM-specific keys and values on top.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

from app.config.constants import (
    RESPONSE_CACHE_MAX_BYTES,
)
from app.config.settings import settings


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """The two bounds; whichever is hit first triggers eviction."""

    max_entries: int
    max_bytes: int

    @classmethod
    def default(cls) -> "CachePolicy":
        return cls(
            max_entries=settings.LLM_RESPONSE_CACHE_MAX_ENTRIES,
            max_bytes=RESPONSE_CACHE_MAX_BYTES,
        )


@dataclass(slots=True)
class CacheStats:
    """A point-in-time view of cache behaviour, for metrics/reports."""

    entries: int
    bytes: int
    hits: int
    misses: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def as_dict(self) -> dict:
        return {
            "entries": self.entries,
            "bytes": self.bytes,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
        }


class BoundedLRUCache:
    """
    Thread-safe LRU cache with entry-count and byte bounds.

    Values are stored alongside a caller-supplied byte size; the cache
    tracks the running total and evicts the least-recently-used entries
    until both bounds are satisfied.
    """

    def __init__(self, policy: CachePolicy | None = None) -> None:
        self._policy = policy or CachePolicy.default()
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, tuple[object, int]]" = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str):
        """Return the value for ``key`` (marking it most-recent), or None."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return entry[0]

    def put(self, key: str, value, size_bytes: int) -> None:
        """Insert/replace ``key``, then evict until within both bounds."""
        with self._lock:
            if key in self._store:
                old_size = self._store[key][1]
                self._bytes -= old_size
                self._store.move_to_end(key)
            self._store[key] = (value, size_bytes)
            self._bytes += size_bytes
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        # Caller holds the lock.
        while self._store and (
            len(self._store) > self._policy.max_entries or self._bytes > self._policy.max_bytes
        ):
            _key, (_value, size) = self._store.popitem(last=False)
            self._bytes -= size
            self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._bytes = 0

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                entries=len(self._store),
                bytes=self._bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )
