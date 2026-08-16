"""
Profiling & benchmarking.

The tools that measure what the rubric scores: latency (half the
efficiency score) and peak RAM against the 7GB budget (the other half
and the Budget Profile bonus).

    Timer, time_runs, LatencyStats      latency primitives
    RamMonitor, current_rss_mb          peak-RSS sampling vs. the ceiling
    BenchmarkSuite,                     per-stage + end-to-end benchmarks
      benchmark_pipeline_stages,
      benchmark_end_to_end
    PerformanceReport                   JSON + human-readable report

Every benchmark takes its components as arguments: the same harness
runs with fakes in CI (proving it works) and with the real model on
the 8GB target machine (producing the numbers that count). A run in a
dev sandbox formats correctly but its figures are not the real ones -
measure on target.
"""

from __future__ import annotations

from app.profiling.benchmark import (
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkSuite,
    benchmark_end_to_end,
    benchmark_pipeline_stages,
)
from app.profiling.latency import (
    LatencyStats,
    Timer,
    summarize,
    time_once,
    time_runs,
)
from app.profiling.ram_monitor import (
    RamMonitor,
    RamResult,
    current_rss_mb,
)
from app.profiling.report import PerformanceReport

__all__ = [
    # latency
    "Timer",
    "time_once",
    "time_runs",
    "summarize",
    "LatencyStats",
    # ram
    "RamMonitor",
    "RamResult",
    "current_rss_mb",
    # benchmark
    "BenchmarkSuite",
    "BenchmarkCase",
    "BenchmarkReport",
    "benchmark_pipeline_stages",
    "benchmark_end_to_end",
    # report
    "PerformanceReport",
]