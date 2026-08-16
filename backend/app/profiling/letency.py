"""
Latency measurement helpers.

Small, dependency-free timing tools the benchmark suite and any ad-hoc
profiling use. A context manager for one-shot timing, a repeated-run
timer that returns per-run samples, and a percentile summary so a
result is reported as p50/p95/mean rather than a single number that a
lucky or unlucky run could distort.

Latency is half of the efficiency score (the other half is RAM), so
these are deliberately simple and honest: wall-clock time via
`time.perf_counter`, no hidden smoothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class LatencyStats:
    """Summary of a set of timed runs, all in milliseconds."""

    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    samples_ms: list[float]

    def as_dict(self) -> dict:
        return {
            "runs": self.runs,
            "mean_ms": round(self.mean_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
        }


class Timer:
    """
    Context manager for one-shot timing.

        with Timer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def time_once(fn: Callable[[], object]) -> float:
    """Run `fn` once and return its wall-clock duration in ms."""
    with Timer() as t:
        fn()
    return t.elapsed_ms


def time_runs(
    fn: Callable[[], object],
    runs: int,
    warmup: int = 0,
) -> LatencyStats:
    """
    Run `fn` `warmup` untimed times, then `runs` timed times, and
    summarize. `fn` takes no args - close over inputs at the call site.
    """
    for _ in range(max(0, warmup)):
        fn()

    samples: list[float] = [time_once(fn) for _ in range(max(1, runs))]
    return summarize(samples)


def summarize(samples_ms: list[float]) -> LatencyStats:
    """Compute mean/p50/p95/min/max over a list of millisecond samples."""
    if not samples_ms:
        return LatencyStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, [])

    ordered = sorted(samples_ms)
    return LatencyStats(
        runs=len(ordered),
        mean_ms=sum(ordered) / len(ordered),
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        min_ms=ordered[0],
        max_ms=ordered[-1],
        samples_ms=samples_ms,
    )


def _percentile(ordered: list[float], pct: float) -> float:
    """
    Nearest-rank percentile over an already-sorted list. Simple and
    unambiguous for the small sample counts benchmarks use.
    """
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = max(1, int(round((pct / 100.0) * len(ordered))))
    return ordered[min(rank, len(ordered)) - 1]