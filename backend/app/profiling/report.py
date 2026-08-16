"""
Performance report.

Turns a BenchmarkReport into two outputs:

- a machine-readable JSON file (for diffing runs, CI thresholds, or
  pasting figures into the submission), and
- a human-readable text summary - the per-stage latency table and the
  peak-RAM-against-budget line that go straight into REPORT.md's
  memory-budget section.

This module does no measuring itself; it only formats what the
benchmark suite produced. That keeps "how we measure" and "how we
present" separate, so the report layout can change without touching
the timing code.

Reminder: numbers are only meaningful when the benchmark ran on the
target machine with the real model. A report generated in a dev
sandbox with fakes formats correctly but the figures are not the
rubric numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config.constants import RAM_CEILING_MB, RAM_WARN_MB
from app.profiling.benchmark import BenchmarkReport
from app.utils.logger import get_logger

logger = get_logger("PerfReport")


@dataclass(slots=True)
class PerformanceReport:
    """A benchmark report plus metadata, renderable to JSON or text."""

    benchmark: BenchmarkReport
    generated_at: str
    label: str = ""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_benchmark(
        cls, benchmark: BenchmarkReport, label: str = ""
    ) -> "PerformanceReport":
        return cls(
            benchmark=benchmark,
            generated_at=datetime.now(timezone.utc).isoformat(),
            label=label,
        )

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        peak = self._overall_peak_mb()
        return {
            "label": self.label,
            "generated_at": self.generated_at,
            "ram_ceiling_mb": RAM_CEILING_MB,
            "ram_warn_mb": RAM_WARN_MB,
            "overall_peak_mb": round(peak, 1),
            "within_budget": peak <= RAM_CEILING_MB,
            "headroom_mb": round(RAM_CEILING_MB - peak, 1),
            "benchmark": self.benchmark.as_dict(),
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2), encoding="utf-8"
        )
        logger.info("perf_report.json_written", path=str(path))
        return path

    # ------------------------------------------------------------------
    # Human-readable output
    # ------------------------------------------------------------------

    def render_text(self) -> str:
        b = self.benchmark
        peak = self._overall_peak_mb()
        within = peak <= RAM_CEILING_MB
        headroom = RAM_CEILING_MB - peak

        lines: list[str] = []
        title = "Shamba Rafiki - Performance Report"
        if self.label:
            title += f" ({self.label})"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append(f"Generated: {self.generated_at}")
        lines.append("")

        # --- Latency table ---
        lines.append("Latency by stage (milliseconds)")
        lines.append("-" * 64)
        lines.append(
            f"{'stage':<22}{'p50':>10}{'p95':>10}{'mean':>10}{'runs':>8}"
        )
        lines.append("-" * 64)
        for case in b.cases:
            lat = case.latency
            lines.append(
                f"{case.name:<22}{lat.p50_ms:>10.1f}{lat.p95_ms:>10.1f}"
                f"{lat.mean_ms:>10.1f}{lat.runs:>8}"
            )
        lines.append("")

        # --- RAM / budget ---
        lines.append("Memory (peak RSS)")
        lines.append("-" * 64)
        lines.append(f"{'Baseline RSS':<28}{b.baseline_rss_mb:>10.1f} MB")
        lines.append(f"{'Overall peak RSS':<28}{peak:>10.1f} MB")
        lines.append(f"{'Budget ceiling':<28}{RAM_CEILING_MB:>10} MB")
        lines.append(f"{'Headroom':<28}{headroom:>10.1f} MB")
        status = "WITHIN BUDGET" if within else "OVER BUDGET (!)"
        if within and peak >= RAM_WARN_MB:
            status = "WITHIN BUDGET - NEAR LIMIT"
        lines.append(f"{'Status':<28}{status:>10}")
        lines.append("")

        # --- Per-stage peak (where the memory goes) ---
        lines.append("Peak RSS by stage")
        lines.append("-" * 64)
        for case in b.cases:
            flag = "" if case.ram.within_budget else "  <-- OVER"
            lines.append(
                f"{case.name:<28}{case.ram.peak_mb:>10.1f} MB{flag}"
            )

        return "\n".join(lines)

    def write_text(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render_text(), encoding="utf-8")
        logger.info("perf_report.text_written", path=str(path))
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _overall_peak_mb(self) -> float:
        """
        The highest peak across all benchmarked cases - the number that
        decides whether the whole system fits under the ceiling. Falls
        back to the baseline if no cases ran.
        """
        peaks = [c.ram.peak_mb for c in self.benchmark.cases]
        return max(peaks) if peaks else self.benchmark.baseline_rss_mb