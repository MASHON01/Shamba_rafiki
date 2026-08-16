"""
Benchmark suite.

Times the pipeline - both each stage in isolation and the full
end-to-end query - and records peak RAM alongside each measurement.
The result is the raw material for the performance report that feeds
the efficiency score and the Budget Profile bonus.

Design (per the chosen approach): every benchmark takes its
components as arguments. Nothing here constructs a real embedder or
LLM. That means the same code runs:

- here / in CI with fakes, proving the harness itself works, and
- on the 8GB target machine with the real embedder, index, and
  llama-server, producing the actual numbers that matter.

So the numbers this produces in a dev sandbox are meaningless (fake
components); the numbers it produces on the target machine are the
real ones. Run it there.

Each benchmarked operation is a zero-arg callable, run `warmup` times
untimed then `runs` times timed, with a RamMonitor wrapped around the
timed runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from app.config.constants import (
    BENCHMARK_MEASURED_RUNS,
    BENCHMARK_WARMUP_RUNS,
)
from app.models.request import QueryRequest
from app.profiling.latency import LatencyStats, time_runs
from app.profiling.ram_monitor import RamMonitor, RamResult, current_rss_mb
from app.utils.logger import get_logger

logger = get_logger("Benchmark")


@dataclass(slots=True)
class BenchmarkCase:
    """One benchmarked operation's latency + RAM result."""

    name: str
    latency: LatencyStats
    ram: RamResult

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "latency": self.latency.as_dict(),
            "ram": self.ram.as_dict(),
        }


@dataclass(slots=True)
class BenchmarkReport:
    """All benchmarked cases plus a captured environment baseline."""

    cases: list[BenchmarkCase] = field(default_factory=list)
    baseline_rss_mb: float = 0.0

    def as_dict(self) -> dict:
        return {
            "baseline_rss_mb": round(self.baseline_rss_mb, 1),
            "cases": [c.as_dict() for c in self.cases],
        }


class BenchmarkSuite:
    """
    Runs a set of named operations and collects latency + RAM for each.
    """

    def __init__(
        self,
        runs: int = BENCHMARK_MEASURED_RUNS,
        warmup: int = BENCHMARK_WARMUP_RUNS,
    ) -> None:
        self._runs = runs
        self._warmup = warmup
        self._report = BenchmarkReport(baseline_rss_mb=current_rss_mb())

    def measure(self, name: str, operation: Callable[[], object]) -> BenchmarkCase:
        """
        Benchmark a single zero-arg operation: latency over repeated
        runs, and peak RAM across those runs.
        """
        logger.info("benchmark.case.start", name=name)

        with RamMonitor() as ram:
            latency = time_runs(operation, runs=self._runs, warmup=self._warmup)

        assert ram.result is not None
        case = BenchmarkCase(name=name, latency=latency, ram=ram.result)
        self._report.cases.append(case)

        logger.info(
            "benchmark.case.done",
            name=name,
            p50_ms=case.latency.p50_ms,
            p95_ms=case.latency.p95_ms,
            peak_mb=case.ram.peak_mb,
        )
        return case

    def report(self) -> BenchmarkReport:
        return self._report


def benchmark_pipeline_stages(
    *,
    analyzer,
    retriever,
    context_builder,
    prompt_builder,
    llm_client,
    verifier,
    query: str = "How do I treat maize blight in Nakuru?",
    runs: int = BENCHMARK_MEASURED_RUNS,
    warmup: int = BENCHMARK_WARMUP_RUNS,
) -> BenchmarkReport:
    """
    Benchmark each pipeline stage in isolation, using whatever
    components are passed in (fakes here, real on target).

    Stages: language analysis, retrieval, prompt build, generation,
    verification. Each is timed on its own so the report shows where
    latency and memory actually go - the input to any optimization.
    """
    from app.models.language import LanguageCode

    suite = BenchmarkSuite(runs=runs, warmup=warmup)

    # Stage 1: language analysis
    suite.measure("language_analysis", lambda: analyzer.analyze_full(query))

    # Precompute an analysis to feed later stages realistically.
    analysis = analyzer.analyze_full(query)

    # Stage 2: retrieval
    if retriever is not None:
        suite.measure(
            "retrieval",
            lambda: retriever.retrieve(analysis.retrieval_query),
        )
        sources = retriever.retrieve(analysis.retrieval_query)
    else:
        sources = []

    # Stage 3: context + prompt build
    context = context_builder.build(sources)
    suite.measure(
        "prompt_build",
        lambda: prompt_builder.build(
            question=query, context=context,
            language=LanguageCode.ENGLISH, intent=analysis.intent,
        ),
    )
    prompt = prompt_builder.build(
        question=query, context=context,
        language=LanguageCode.ENGLISH, intent=analysis.intent,
    )

    # Stage 4: generation
    suite.measure("generation", lambda: llm_client.generate(prompt))
    generation = llm_client.generate(prompt)

    # Stage 5: verification
    suite.measure(
        "verification",
        lambda: verifier.verify(generation.text, sources,
                                language=LanguageCode.ENGLISH),
    )

    return suite.report()


def benchmark_end_to_end(
    *,
    orchestrator,
    query: str = "How do I treat maize blight in Nakuru?",
    runs: int = BENCHMARK_MEASURED_RUNS,
    warmup: int = BENCHMARK_WARMUP_RUNS,
) -> BenchmarkReport:
    """
    Benchmark a full query through the orchestrator - the number that
    represents real end-user latency and peak RAM per request.
    """
    suite = BenchmarkSuite(runs=runs, warmup=warmup)
    suite.measure(
        "end_to_end_query",
        lambda: orchestrator.handle_query(
            QueryRequest(query=query, language="en")
        ),
    )
    return suite.report()