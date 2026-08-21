#!/usr/bin/env python3
"""
Run the benchmark suite and write a performance report.

    python -m scripts.run_benchmarks               # real components (default)
    python -m scripts.run_benchmarks --mock        # fakes, for a smoke test
    python -m scripts.run_benchmarks --stages       # per-stage only
    python -m scripts.run_benchmarks --e2e          # end-to-end only
    python -m scripts.run_benchmarks --runs 10      # more measured runs

By default this benchmarks the REAL stack - the SentenceTransformer
embedder, the on-disk index, and llama-server - so it must be run on
the target machine with llama-server up and an index built. That is
the whole point: these are the numbers behind the Efficiency score
and the Budget Profile bonus, and they only mean anything on target.

`--mock` swaps in the deterministic fake embedder/LLM so you can
verify the harness runs end to end without a model. The report it
writes then is structurally correct but the figures are meaningless -
don't put mock numbers in REPORT.md.

Reports are written to data/reports/ as a timestamped .json (for
diffing / CI) and .txt (the human-readable table for REPORT.md).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make `app` importable when run as a plain script too.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.logging import configure_logging  # noqa: E402
from app.config.paths import DATA_DIR  # noqa: E402
from app.profiling import (  # noqa: E402
    PerformanceReport,
    benchmark_end_to_end,
    benchmark_pipeline_stages,
)
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("RunBenchmarks")

REPORTS_DIR = DATA_DIR / "reports"


def _build_real_stack() -> dict:
    """
    Assemble the real pipeline components. Reuses the API's dependency
    wiring so the benchmarked stack is exactly what serves requests.
    """
    from app.api.dependencies import get_orchestrator, get_retriever, init_dependencies
    from app.language.analyzer import LanguageAnalyzer
    from app.orchestration.llm.llama_client import LlamaClient
    from app.orchestration.prompts import ContextBuilder, PromptBuilder
    from app.verification.verifier import Verifier

    init_dependencies()
    orchestrator = get_orchestrator()
    retriever = get_retriever()

    if retriever is None:
        logger.warning(
            "benchmark.no_index",
            hint="No vector index found - retrieval stage will be skipped. "
            "Build the index first for complete numbers.",
        )

    return {
        "analyzer": LanguageAnalyzer(),
        "retriever": retriever,
        "context_builder": ContextBuilder(),
        "prompt_builder": PromptBuilder(),
        "llm_client": LlamaClient(),
        "verifier": Verifier(),
        "orchestrator": orchestrator,
    }


def _build_mock_stack() -> dict:
    """Deterministic fakes - harness smoke test only, numbers are fake."""
    import tempfile

    from app.ingestion.builder import CorpusBuilder
    from app.ingestion.extractors import TextExtractor
    from app.ingestion.pipeline import IngestionPipeline
    from app.ingestion.processors import (
        Chunker, Cleaner, DuplicateDetector, Hasher, MetadataGenerator,
    )
    from app.language.analyzer import LanguageAnalyzer
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.llm.base import BaseLLMClient, GenerationResult
    from app.orchestration.orchestrator import Orchestrator
    from app.orchestration.prompts import ContextBuilder, PromptBuilder
    from app.retrieval.embeddings.base import BaseEmbedder
    from app.retrieval.indexer import Indexer
    from app.retrieval.search import Retriever
    from app.retrieval.store import create_vector_store
    from app.verification.verifier import Verifier

    vocab = ["maize", "beans", "tomato", "blight", "fungicide", "mancozeb", "rotation", "nakuru"]

    class FakeEmbedder(BaseEmbedder):
        def __init__(self) -> None:
            self.dimension = len(vocab)

        def _v(self, t: str) -> list[float]:
            low = t.lower()
            return [float(low.count(w)) + 0.01 for w in vocab]

        def embed_texts(self, ts):
            return [self._v(t) for t in ts]

        def embed_query(self, t):
            return self._v(t)

    class FakeLLM(BaseLLMClient):
        def generate(self, prompt):
            return GenerationResult(
                text="Apply mancozeb fungicide at 40 grams per 20 litres every 7 days.",
                prompt_tokens=100, completion_tokens=15, latency_ms=30)

        def health(self):
            return True

    tmp = Path(tempfile.mkdtemp(prefix="bench_mock_"))
    raw = tmp / "raw"
    raw.mkdir()
    (raw / "m.txt").write_text(
        "Maize leaf blight in Nakuru. Apply mancozeb fungicide at 40 grams "
        "per 20 litres. Spray every 7 days. Rotation with beans helps. " * 3,
        encoding="utf-8",
    )
    corpus, store = tmp / "proc", tmp / "vs"
    IngestionPipeline(
        extractor=TextExtractor(), cleaner=Cleaner(), chunker=Chunker(),
        metadata_generator=MetadataGenerator(), hasher=Hasher(),
        duplicate_detector=DuplicateDetector(corpus_dir=corpus),
        builder=CorpusBuilder(output_dir=corpus),
    ).run(raw, source="KALRO", language="en")
    Indexer(
        embedder=FakeEmbedder(),
        store=create_vector_store(len(vocab), backend="numpy"),
        backend="numpy",
    ).build(corpus_dir=corpus, store_dir=store)
    retriever = Retriever.from_disk(
        store_dir=store, backend="numpy",
        embedder=FakeEmbedder(), default_threshold=0.1,
    )

    llm = FakeLLM()
    return {
        "analyzer": LanguageAnalyzer(),
        "retriever": retriever,
        "context_builder": ContextBuilder(),
        "prompt_builder": PromptBuilder(),
        "llm_client": llm,
        "verifier": Verifier(),
        "orchestrator": Orchestrator(
            llm_client=llm,
            dispatcher=Dispatcher(llm_client=llm, retriever=retriever)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Farm Pal benchmark suite.")
    parser.add_argument("--mock", action="store_true",
                        help="Use fake components (harness smoke test; numbers are meaningless).")
    parser.add_argument("--stages", action="store_true", help="Per-stage benchmark only.")
    parser.add_argument("--e2e", action="store_true", help="End-to-end benchmark only.")
    parser.add_argument("--runs", type=int, default=None, help="Measured runs per operation.")
    parser.add_argument("--query", type=str,
                        default="How do I treat maize blight in Nakuru?",
                        help="Query to benchmark with.")
    args = parser.parse_args()

    configure_logging()

    run_stages = args.stages or not args.e2e
    run_e2e = args.e2e or not args.stages
    label = "mock" if args.mock else "target"

    logger.info("benchmark.start", mock=args.mock, stages=run_stages, e2e=run_e2e)
    stack = _build_mock_stack() if args.mock else _build_real_stack()

    runs_kwargs = {"runs": args.runs} if args.runs else {}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if run_stages:
        report = benchmark_pipeline_stages(
            analyzer=stack["analyzer"], retriever=stack["retriever"],
            context_builder=stack["context_builder"],
            prompt_builder=stack["prompt_builder"],
            llm_client=stack["llm_client"], verifier=stack["verifier"],
            query=args.query, **runs_kwargs)
        perf = PerformanceReport.from_benchmark(report, label=f"{label}-stages")
        _write(perf, f"benchmark-stages-{timestamp}")

    if run_e2e:
        report = benchmark_end_to_end(
            orchestrator=stack["orchestrator"], query=args.query, **runs_kwargs)
        perf = PerformanceReport.from_benchmark(report, label=f"{label}-e2e")
        _write(perf, f"benchmark-e2e-{timestamp}")

    if args.mock:
        print("\n[!] These are MOCK numbers - do not use them in REPORT.md. "
              "Run without --mock on the 8GB target machine for real figures.")
    return 0


def _write(perf: PerformanceReport, stem: str) -> None:
    json_path = perf.write_json(REPORTS_DIR / f"{stem}.json")
    text_path = perf.write_text(REPORTS_DIR / f"{stem}.txt")
    print(f"\n{perf.render_text()}\n")
    print(f"Written:\n  {json_path}\n  {text_path}")


if __name__ == "__main__":
    raise SystemExit(main())