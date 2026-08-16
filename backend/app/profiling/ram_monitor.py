"""
RAM monitoring.

Measures process memory (RSS) via psutil, tracks the peak during an
operation, and checks it against the 7GB budget. This is the tool
behind the Budget Profile bonus and the OOM-safety guarantee - an
answer that's correct but blows the memory ceiling is still a
failure, so peak RAM matters as much as latency.

Two ways to use it:

- snapshot: `current_rss_mb()` for a single reading.
- peak-during-operation: a background sampler thread reads RSS every
  RAM_SAMPLE_INTERVAL_SECONDS while your code runs, capturing the
  true peak (a single before/after reading would miss a transient
  spike during, say, model loading or a big batch embed).

    with RamMonitor() as m:
        run_a_full_query()
    print(m.peak_mb, m.within_budget)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from app.config.constants import (
    RAM_CEILING_MB,
    RAM_SAMPLE_INTERVAL_SECONDS,
    RAM_WARN_MB,
)
from app.utils.logger import get_logger

logger = get_logger("RamMonitor")


def _load_psutil():
    try:
        import psutil

        return psutil
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "psutil is required for RAM monitoring. Install it with: "
            "pip install psutil"
        ) from exc


def current_rss_mb() -> float:
    """Current resident set size of this process, in MB."""
    psutil = _load_psutil()
    rss_bytes = psutil.Process().memory_info().rss
    return rss_bytes / (1024 * 1024)


@dataclass(slots=True)
class RamResult:
    """Outcome of a monitored operation."""

    baseline_mb: float
    peak_mb: float
    delta_mb: float
    ceiling_mb: int
    within_budget: bool
    near_limit: bool

    def as_dict(self) -> dict:
        return {
            "baseline_mb": round(self.baseline_mb, 1),
            "peak_mb": round(self.peak_mb, 1),
            "delta_mb": round(self.delta_mb, 1),
            "ceiling_mb": self.ceiling_mb,
            "within_budget": self.within_budget,
            "near_limit": self.near_limit,
        }


class RamMonitor:
    """
    Context manager that samples RSS in the background and reports the
    peak against the budget.
    """

    def __init__(
        self,
        ceiling_mb: int = RAM_CEILING_MB,
        warn_mb: int = RAM_WARN_MB,
        interval: float = RAM_SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self._ceiling_mb = ceiling_mb
        self._warn_mb = warn_mb
        self._interval = interval
        self._baseline_mb = 0.0
        self._peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.result: RamResult | None = None

    def __enter__(self) -> "RamMonitor":
        self._baseline_mb = current_rss_mb()
        self._peak_mb = self._baseline_mb
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # One final reading in case the peak landed after the last sample.
        self._peak_mb = max(self._peak_mb, current_rss_mb())

        self.result = RamResult(
            baseline_mb=self._baseline_mb,
            peak_mb=self._peak_mb,
            delta_mb=self._peak_mb - self._baseline_mb,
            ceiling_mb=self._ceiling_mb,
            within_budget=self._peak_mb <= self._ceiling_mb,
            near_limit=self._peak_mb >= self._warn_mb,
        )

        if not self.result.within_budget:
            logger.warning(
                "ram.over_budget",
                peak_mb=round(self._peak_mb, 1),
                ceiling_mb=self._ceiling_mb,
            )
        elif self.result.near_limit:
            logger.warning(
                "ram.near_limit",
                peak_mb=round(self._peak_mb, 1),
                warn_mb=self._warn_mb,
            )

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    @property
    def within_budget(self) -> bool:
        return self._peak_mb <= self._ceiling_mb

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak_mb = max(self._peak_mb, current_rss_mb())
            except Exception:  # pragma: no cover - sampling must never crash
                pass
            time.sleep(self._interval)