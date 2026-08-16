#!/usr/bin/env python3
"""
Benchmark a candidate model on the target, then select the winner.

THIS IS THE ONE THING THAT MUST RUN ON THE 8 GB TARGET MACHINE with the
real GGUF. Dev-machine and --mock numbers are indicative only and do not
count toward the rubric.

Two steps:

  # 1) measure each candidate (once per model, on the target)
  MODEL_ID=llama-3.2-3b-instruct-q4_k_m MODEL_PATH=models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
      python -m scripts.select_model measure --serve --soak 120 --out data/reports/cand_llama3b.json()
  #...repeat for each candidate with its MODEL_ID / MODEL_PATH...

  # 2) select the winner from the measured candidates
  python -m scripts.select_model select data/reports/cand_*.json()

`measure` records, for the currently-configured model: peak RAM vs the
7 GB ceiling, end-to-end latency (p50/p95), answer accuracy + hallucination
honesty ( harness), and - with --soak N - a sustained-load
latency-drift check for thermal throttling.

`select` scores the candidates (RAM ceiling is a hard gate; then accuracy,
then RAM headroom, then latency) and writes phase2_model_selection.{json,md}
- the evidence and rationale for REPORT.md.

--mock runs measure offline with fakes so the harness itself can be
verified; the resulting numbers are meaningless.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import paths  # noqa: E402
from app.config.constants import RAM_CEILING_MB  # noqa: E402
from app.config.logging import configure_logging  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.evaluation import evaluate  # noqa: E402
from app.models.request import QueryRequest  # noqa: E402
from app.profiling import (  # noqa: E402
    CandidateMetrics,
    RamMonitor,
    benchmark_end_to_end,
    current_rss_mb,
    select_best,
)
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("SelectModel")

_QUERY = "How do I treat maize blight in Nakuru?"


# ===========================================================================
# measure
# ===========================================================================


def cmd_measure(args) -> int:
    server = _maybe_serve(args)
    try:
        orchestrator, dispatcher, llm, verifier = _build_stack(args.mock)

        # RAM: peak RSS while serving queries, with the model resident.
        baseline = current_rss_mb
        request = QueryRequest(query=_QUERY, language="en")
        with RamMonitor as monitor:
            for _ in range(max(1, args.ram_queries)):
                orchestrator.handle_query(request)
        ram = monitor.result

        # Latency: end-to-end p50/p95.
        bench = benchmark_end_to_end(orchestrator=orchestrator, query=_QUERY)
        latency = bench.cases[0].latency

        # Accuracy: harness.
        config = _config_dict(args.mock)
        report = evaluate(dispatcher, llm, verifier, config)
        acc = report["summary"]["overall_mean"]
        halluc = report["summary"]["hallucination_pass_rate"]

        drift = _soak(orchestrator, request, args.soak) if args.soak else None

        metrics = CandidateMetrics(
            model_id=settings.MODEL_ID,
            peak_ram_mb=ram.peak_mb,
            latency_p50_ms=latency.p50_ms,
            latency_p95_ms=latency.p95_ms,
            accuracy_overall=acc,
            hallucination_pass_rate=halluc,
            latency_drift_pct=drift,
            notes="MOCK - meaningless" if args.mock else "",
        )
        payload = {
            "metrics": metrics.as_dict(),
            "baseline_rss_mb": round(baseline, 1),
            "within_budget_vs_ceiling": ram.peak_mb <= RAM_CEILING_MB,
        }

        out = (
            Path(args.out)
            if args.out
            else (paths.REPORTS_DIR / f"candidate_{settings.MODEL_ID}.json()")
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        _print_measure(metrics, ram.peak_mb <= RAM_CEILING_MB)
        print(f"\nCandidate written -> {out}")
        if args.mock:
            print("\n[!] MOCK - numbers are meaningless. Run on the 8 GB target.")
        return 0
    finally:
        if server is not None:
            server.stop()


def _soak(orchestrator, request, seconds: float) -> float:
    """Run queries for `seconds`; return latency drift % (thermal check)."""
    samples: list[float] = []
    end = time.monotonic + seconds
    while time.monotonic < end:
        t0 = time.monotonic
        orchestrator.handle_query(request)
        samples.append((time.monotonic - t0) * 1000)
    if len(samples) < 6:
        return 0.0
    third = len(samples) // 3
    first = sum(samples[:third]) / third
    last = sum(samples[-third:]) / third
    return round((last - first) / first * 100, 1) if first else 0.0

    # ===========================================================================
    # select
    # ===========================================================================


def cmd_select(args) -> int:
    files: list[str] = []
    for pattern in args.candidates:
        files.extend(sorted(glob.glob(pattern)))
    if not files:
        print("[!!] No candidate files matched.")
        return 1

    candidates = [_load_candidate(Path(f)) for f in files]
    result = select_best(candidates)

    _print_selection(result)

    out_json = Path(args.out) if args.out else (paths.REPORTS_DIR / "phase2_model_selection.json()")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_selection_markdown(result), encoding="utf-8")
    print(f"\nWritten:\n {out_json}\n {out_md}")
    return 0


def _load_candidate(path: Path) -> CandidateMetrics:
    data = json.loads(path.read_text(encoding="utf-8"))
    m = data.get("metrics", data)
    return CandidateMetrics(
        model_id=m["model_id"],
        peak_ram_mb=m["peak_ram_mb"],
        latency_p50_ms=m["latency_p50_ms"],
        latency_p95_ms=m["latency_p95_ms"],
        accuracy_overall=m["accuracy_overall"],
        hallucination_pass_rate=m["hallucination_pass_rate"],
        latency_drift_pct=m.get("latency_drift_pct"),
        notes=m.get("notes", ""),
    )

    # ===========================================================================
    # Stack building
    # ===========================================================================


def _maybe_serve(args):
    if args.mock or not getattr(args, "serve", False):
        return None
    from app.model_runtime.server_manager import ServerConfig, ServerManager

    manager = ServerManager(ServerConfig.from_settings)
    manager.start()
    ready = manager.wait_until_ready
    if not ready:
        manager.stop()
        raise SystemExit("[!!] llama-server did not become ready.")
    return manager


def _build_stack(mock: bool):
    if mock:
        from app.orchestration.dispatcher import Dispatcher
        from app.orchestration.llm.base import BaseLLMClient, GenerationResult
        from app.orchestration.orchestrator import Orchestrator

        class FakeLLM(BaseLLMClient):
            def generate(self, prompt, config=None):
                return GenerationResult(
                    text="Apply mancozeb fungicide [Source 1] and rotate maize "
                    "with beans. If unsure, consult an extension officer.",
                    prompt_tokens=80,
                    completion_tokens=16,
                    latency_ms=10,
                )

            def health(self):
                return True

        llm = FakeLLM
        dispatcher = Dispatcher(llm_client=llm, retriever=None)
        orch = Orchestrator(llm_client=llm, dispatcher=dispatcher)
        return orch, dispatcher, llm, orch._verifier

    from app.api.dependencies import get_orchestrator, init_dependencies

    init_dependencies()
    orch = get_orchestrator
    return orch, orch._dispatcher, orch._dispatcher._llm, orch._verifier


def _config_dict(mock: bool) -> dict:
    return {
        "prompt_version": settings.PROMPT_VERSION,
        "model_id": settings.MODEL_ID,
        "swahili_enrich": settings.SWAHILI_PROMPT_ENRICH,
        "verification_enabled": settings.ENABLE_VERIFICATION,
        "mock": mock,
    }

    # ===========================================================================
    # Output
    # ===========================================================================


def _print_measure(m: CandidateMetrics, within: bool) -> None:
    print(f"\nCandidate: {m.model_id}")
    print("=" * 46)
    status = "WITHIN BUDGET" if within else "OVER CEILING (!)"
    print(f"{'Peak RAM':<24}{m.peak_ram_mb:>10.1f} MB ({status})")
    print(f"{'Latency p50 / p95':<24}{m.latency_p50_ms:>8.0f} / {m.latency_p95_ms:.0f} ms")
    print(f"{'Accuracy (overall)':<24}{m.accuracy_overall:>9.0%}")
    print(f"{'Hallucination honesty':<24}{m.hallucination_pass_rate:>9.0%}")
    if m.latency_drift_pct is not None:
        print(f"{'Sustained-load drift':<24}{m.latency_drift_pct:>9.1f} %")


def _print_selection(result) -> None:
    print("\nModel selection")
    print("=" * 46)
    for r in result.ranked:
        mark = " *" if r.metrics.model_id == result.winner_id else (" x" if r.disqualified else " ")
        print(
            f"{mark} {r.metrics.model_id:<34} score={r.score:.3f}"
            f" ram={r.metrics.peak_ram_mb:.0f}MB acc={r.metrics.accuracy_overall:.0%}"
        )
    print("-" * 46)
    print(result.rationale)


def _selection_markdown(result) -> str:
    lines = [
        "# - Model Selection",
        "",
        f"**Winner: {result.winner_id or 'none'}**",
        "",
        result.rationale,
        "",
        f"RAM ceiling: {RAM_CEILING_MB} MB (hard budget).",
        "",
        "| model | score | peak RAM (MB) | p50 (ms) | p95 (ms) | accuracy | halluc | budget |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in result.ranked:
        m = r.metrics
        budget = "OVER" if r.disqualified else "ok"
        lines.append(
            f"| {m.model_id} | {r.score:.3f} | {m.peak_ram_mb:.0f} | "
            f"{m.latency_p50_ms:.0f} | {m.latency_p95_ms:.0f} | "
            f"{m.accuracy_overall:.0%} | {m.hallucination_pass_rate:.0%} | {budget} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark + select the model on the target.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="Measure the configured model.")
    m.add_argument("--serve", action="store_true", help="Start llama-server for the model.")
    m.add_argument("--soak", type=float, default=0.0, help="Sustained-load seconds (thermal).")
    m.add_argument("--ram-queries", type=int, default=3)
    m.add_argument("--mock", action="store_true", help="Offline fakes (meaningless).")
    m.add_argument("--out", default=None)
    m.set_defaults(func=cmd_measure)

    s = sub.add_parser("select", help="Rank measured candidates.")
    s.add_argument("candidates", nargs="+", help="Candidate JSON files / globs.")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_select)

    args = parser.parse_args()
    configure_logging()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main)
