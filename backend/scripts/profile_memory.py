#!/usr/bin/env python3
"""
Measure peak RAM of a full query against the 7GB ceiling.

    python -m scripts.profile_memory              # real stack (default)
    python -m scripts.profile_memory --mock       # fakes, smoke test
    python -m scripts.profile_memory --queries 5  # average over N queries

This is the single most important number for the Budget Profile bonus
and OOM safety: the peak resident memory the process reaches while
serving a real query, and whether it stays under RAM_CEILING_MB
(7168 MB). Run it on the 8GB target machine with llama-server up and
the real model loaded - that is the only place the number is real.

It reports the baseline RSS (idle, after the model is resident), the
peak during the query, the delta, and the headroom below the ceiling.
A peak over the ceiling is flagged loudly: exceeding 7GB risks an OOM
kill, which is a total failure.

`--mock` runs the whole thing with fakes so you can confirm the tool
works; the resulting MB figures are meaningless (no real model).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.constants import RAM_CEILING_MB, RAM_WARN_MB  # noqa: E402
from app.config.logging import configure_logging  # noqa: E402
from app.models.request import QueryRequest  # noqa: E402
from app.profiling.ram_monitor import RamMonitor, current_rss_mb  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("ProfileMemory")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile peak RAM of a full query vs. the 7GB ceiling.")
    parser.add_argument("--mock", action="store_true",
                        help="Use fake components (smoke test; MB figures meaningless).")
    parser.add_argument("--queries", type=int, default=3,
                        help="Number of queries to run under the monitor.")
    parser.add_argument("--query", type=str,
                        default="How do I treat maize blight in Nakuru?")
    args = parser.parse_args()

    configure_logging()

    orchestrator = _mock_orchestrator() if args.mock else _real_orchestrator()

    # Baseline AFTER the model/index are resident, so the delta reflects
    # per-query cost, not one-time load.
    baseline = current_rss_mb()
    logger.info("profile.baseline", rss_mb=round(baseline, 1))

    request = QueryRequest(query=args.query, language="en")

    with RamMonitor() as monitor:
        for _ in range(max(1, args.queries)):
            orchestrator.handle_query(request)

    result = monitor.result
    assert result is not None

    _print_report(result, baseline, args.mock)

    # Non-zero exit if over budget, so CI / a shell check can catch it.
    return 0 if result.within_budget else 2


def _print_report(result, baseline: float, mock: bool) -> None:
    peak = result.peak_mb
    within = result.within_budget
    headroom = RAM_CEILING_MB - peak

    print("\nMemory profile - full query")
    print("=" * 44)
    print(f"{'Baseline RSS (model resident)':<32}{baseline:>8.1f} MB")
    print(f"{'Peak RSS during query':<32}{peak:>8.1f} MB")
    print(f"{'Per-query delta':<32}{result.delta_mb:>8.1f} MB")
    print(f"{'Warn line':<32}{RAM_WARN_MB:>8} MB")
    print(f"{'Ceiling (hard budget)':<32}{RAM_CEILING_MB:>8} MB")
    print(f"{'Headroom below ceiling':<32}{headroom:>8.1f} MB")
    print("-" * 44)
    if not within:
        print(f"{'STATUS':<32}{'OVER BUDGET (!)':>8}")
        print("\n[!!] Peak RAM exceeds the 7GB ceiling. This risks an OOM "
              "kill - a total failure. Reduce model size or quantization.")
    elif peak >= RAM_WARN_MB:
        print(f"{'STATUS':<32}{'NEAR LIMIT':>8}")
        print("\n[!] Within budget but little headroom left.")
    else:
        print(f"{'STATUS':<32}{'WITHIN BUDGET':>8}")

    if mock:
        print("\n[!] MOCK run - these MB figures are meaningless. Run without "
              "--mock on the 8GB target machine with the real model for the "
              "actual Budget Profile number.")


def _real_orchestrator():
    from app.api.dependencies import get_orchestrator, init_dependencies

    init_dependencies()
    return get_orchestrator()


def _mock_orchestrator():
    import tempfile

    from app.ingestion.builder import CorpusBuilder
    from app.ingestion.extractors import TextExtractor
    from app.ingestion.pipeline import IngestionPipeline
    from app.ingestion.processors import (
        Chunker, Cleaner, DuplicateDetector, Hasher, MetadataGenerator,
    )
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.llm.base import BaseLLMClient, GenerationResult
    from app.orchestration.orchestrator import Orchestrator
    from app.retrieval.embeddings.base import BaseEmbedder
    from app.retrieval.indexer import Indexer
    from app.retrieval.search import Retriever
    from app.retrieval.store import create_vector_store

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
            return GenerationResult(text="Use fungicide and rotate crops.",
                                    prompt_tokens=100, completion_tokens=8, latency_ms=30)

        def health(self):
            return True

    tmp = Path(tempfile.mkdtemp(prefix="profmem_mock_"))
    raw = tmp / "raw"
    raw.mkdir()
    (raw / "m.txt").write_text(
        "Maize leaf blight in Nakuru. Apply mancozeb fungicide. " * 5, encoding="utf-8")
    corpus, store = tmp / "proc", tmp / "vs"
    IngestionPipeline(
        extractor=TextExtractor(), cleaner=Cleaner(), chunker=Chunker(),
        metadata_generator=MetadataGenerator(), hasher=Hasher(),
        duplicate_detector=DuplicateDetector(corpus_dir=corpus),
        builder=CorpusBuilder(output_dir=corpus),
    ).run(raw, source="KALRO", language="en")
    Indexer(embedder=FakeEmbedder(),
            store=create_vector_store(len(vocab), backend="numpy"),
            backend="numpy").build(corpus_dir=corpus, store_dir=store)
    retriever = Retriever.from_disk(store_dir=store, backend="numpy",
                                    embedder=FakeEmbedder(), default_threshold=0.1)
    llm = FakeLLM()
    return Orchestrator(llm_client=llm,
                        dispatcher=Dispatcher(llm_client=llm, retriever=retriever))


if __name__ == "__main__":
    raise SystemExit(main())