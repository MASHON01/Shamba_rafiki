#!/usr/bin/env python3
"""
Tune generation config knobs against the answer-quality harness.

    python -m scripts.tune_config # real stack, full sweep
    python -m scripts.tune_config --versions v1 v2
    python -m scripts.tune_config --thresholds 0.5 0.6 0.7
    python -m scripts.tune_config --mock # offline smoke test

Sweeps the levers that move the accuracy score - the system-prompt
version, Swahili enrichment on/off, and the retrieval SIMILARITY_THRESHOLD
- runs harness for each combination, and reports which
configuration scores best. This is how the earlier "keep v1 until the
eval proves v2" and "enrichment on if it helps" decisions get settled
with evidence, and how the threshold is tuned against the REAL embedder
and corpus (which is why the meaningful run is on the target).

The threshold only bites with a real retriever + corpus; under --mock
(no retriever) it is a no-op and only one threshold is tried.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import paths  # noqa: E402
from app.config.logging import configure_logging  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.evaluation import evaluate  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("TuneConfig")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune prompt/enrich/threshold on the eval harness."
    )
    parser.add_argument("--versions", nargs="+", default=["v1", "v2"])
    parser.add_argument(
        "--enrich",
        nargs="+",
        type=_bool,
        default=[True, False],
        help="Swahili enrichment values to try (true/false).",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    parser.add_argument("--mock", action="store_true", help="Offline fakes (meaningless).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    configure_logging()

    thresholds = [settings.SIMILARITY_THRESHOLD] if args.mock else args.thresholds
    results = []

    for threshold in thresholds:
        settings.SIMILARITY_THRESHOLD = threshold
        dispatcher, llm, verifier = _stack(args.mock)
        for version in args.versions:
            for enrich in args.enrich:
                settings.PROMPT_VERSION = version
                settings.SWAHILI_PROMPT_ENRICH = enrich
                config = {
                    "prompt_version": version,
                    "swahili_enrich": enrich,
                    "similarity_threshold": threshold,
                    "model_id": settings.MODEL_ID,
                    "verification_enabled": settings.ENABLE_VERIFICATION,
                    "mock": args.mock,
                }
                report = evaluate(dispatcher, llm, verifier, config)
                s = report["summary"]
                results.append(
                    {
                        "config": config,
                        "overall_mean": s["overall_mean"],
                        "hallucination_pass_rate": s["hallucination_pass_rate"],
                        "by_language": s["by_language"],
                    }
                )

    results.sort(key=lambda r: r["overall_mean"], reverse=True)
    _print(results)

    out = Path(args.out) if args.out else paths.REPORTS_DIR / "phase2_tuning.json()"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten -> {out}")
    if args.mock:
        print(
            "\n[!] MOCK - scores meaningless. Tune on the target with the real " "model + corpus."
        )
    return 0


def _stack(mock: bool):
    if mock:
        from app.orchestration.dispatcher import Dispatcher
        from app.orchestration.llm.base import BaseLLMClient, GenerationResult
        from app.verification.verifier import Verifier

        class FakeLLM(BaseLLMClient):
            def generate(self, prompt, config=None):
                return GenerationResult(
                    text="Apply mancozeb fungicide [Source 1]; if unsure consult "
                    "an extension officer.",
                    prompt_tokens=1,
                    completion_tokens=1,
                )

            def health(self):
                return True

        llm = FakeLLM
        return Dispatcher(llm_client=llm, retriever=None), llm, Verifier

        # Rebuild the retriever so the new SIMILARITY_THRESHOLD takes effect.
    from app.api.dependencies import get_orchestrator, init_dependencies

    init_dependencies()
    orch = get_orchestrator
    return orch._dispatcher, orch._dispatcher._llm, orch._verifier


def _print(results) -> None:
    print("\nConfig tuning (sorted by overall accuracy)")
    print("=" * 60)
    for r in results[:12]:
        c = r["config"]
        print(
            f"v={c['prompt_version']:<3} enrich={str(c['swahili_enrich']):<5} "
            f"thr={c['similarity_threshold']:<4} -> overall={r['overall_mean']:.0%} "
            f"halluc={r['hallucination_pass_rate']:.0%}"
        )
    if results:
        best = results[0]["config"]
        print("-" * 60)
        print(
            f"Best: PROMPT_VERSION={best['prompt_version']}, "
            f"SWAHILI_PROMPT_ENRICH={best['swahili_enrich']}, "
            f"SIMILARITY_THRESHOLD={best['similarity_threshold']}"
        )


def _bool(value: str) -> bool:
    return str(value).lower() in ("1", "true", "yes", "on")


if __name__ == "__main__":
    raise SystemExit(main)
