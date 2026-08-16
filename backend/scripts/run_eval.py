#!/usr/bin/env python3
"""
Score a configuration end to end on the answer-quality harness.

    python -m scripts.run_eval # real stack
    python -m scripts.run_eval --mock # offline smoke test
    python -m scripts.run_eval --prompt-version v2 # score a prompt version
    python -m scripts.run_eval --no-enrich # Swahili enrichment off
    python -m scripts.run_eval --out report.json() # save the report

Runs the golden question set and the hallucination probes through the
real pipeline (language -> retrieval -> prompt -> generation ->
verification), grades every answer, and writes a config-tagged scored
report. Because the report records the prompt version, model, and
toggles, two runs are directly comparable - which is the whole point:
change one thing, re-run, watch the number move.

Real accuracy needs llama-server, the real model, and the real corpus.
--mock proves the harness runs offline; its scores are not meaningful.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.logging import configure_logging  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.evaluation import evaluate, format_report  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from app.verification.verifier import Verifier  # noqa: E402

logger = get_logger("RunEval")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a config on the answer-quality harness.")
    parser.add_argument("--mock", action="store_true", help="Offline fakes (scores meaningless).")
    parser.add_argument("--prompt-version", default=None, help="Prompt version to score (e.g. v2).")
    parser.add_argument("--no-enrich", action="store_true", help="Disable Swahili enrichment.")
    parser.add_argument("--out", default=None, help="Write the report JSON to this path.")
    args = parser.parse_args()

    configure_logging()

    if args.prompt_version:
        settings.PROMPT_VERSION = args.prompt_version
    if args.no_enrich:
        settings.SWAHILI_PROMPT_ENRICH = False

    dispatcher, llm = _build_pipeline(args.mock)
    verifier = Verifier

    config = {
        "prompt_version": settings.PROMPT_VERSION,
        "model_id": settings.MODEL_ID,
        "swahili_enrich": settings.SWAHILI_PROMPT_ENRICH,
        "verification_enabled": settings.ENABLE_VERIFICATION,
        "mock": args.mock,
    }
    report = evaluate(dispatcher, llm, verifier, config)
    print("\n" + format_report(report))

    out = Path(args.out) if args.out else _default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written -> {out}")

    if args.mock:
        print(
            "\n[!] MOCK run - scores are meaningless. Run against the real "
            "model + corpus for real accuracy numbers."
        )
    return 0


def _default_out() -> Path:
    from app.config import paths

    tag = f"{settings.PROMPT_VERSION}_{settings.MODEL_ID}"
    return paths.REPORTS_DIR / f"eval_{tag}.json()"


def _build_pipeline(mock: bool):
    from app.orchestration.dispatcher import Dispatcher

    if mock:
        llm = _fake_llm
        return Dispatcher(llm_client=llm, retriever=None), llm

    from app.api.dependencies import get_retriever, init_dependencies

    init_dependencies()
    from app.orchestration.llm.llama_client import LlamaClient

    llm = LlamaClient
    if not llm.health:
        print("[!!] llama-server not reachable. Start it or pass --mock.")
        raise SystemExit(1)
    return Dispatcher(llm_client=llm, retriever=get_retriever), llm


def _fake_llm():
    from app.orchestration.llm.base import BaseLLMClient, GenerationResult

    class FakeLLM(BaseLLMClient):
        def generate(self, prompt, config=None):
            return GenerationResult(
                text="I'm not sure - please consult a local agricultural "
                "officer. Based on general guidance, remove affected leaves.",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=1,
            )

        def health(self):
            return True

    return FakeLLM


if __name__ == "__main__":
    raise SystemExit(main)
