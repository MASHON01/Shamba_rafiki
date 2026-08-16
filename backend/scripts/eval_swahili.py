#!/usr/bin/env python3
"""
Measure Swahili answer quality - the zero-shot baseline and the effect of
prompt-side enrichment.

    python -m scripts.eval_swahili # enrichment ON (real server)
    python -m scripts.eval_swahili --no-enrich # baseline, enrichment OFF
    python -m scripts.eval_swahili --compare # ON vs OFF, side by side
    python -m scripts.eval_swahili --mock # offline smoke test

For each question in the Swahili eval set it builds the prompt (with or
without the glossary + few-shot enrichment), generates an answer, and
scores two things automatically:

  - term coverage: did the answer engage the expected domain concepts
    (the English canonical term OR any of its Swahili forms both count)?
  - language: did the answer come back in Kiswahili?

This is the evidence for two decisions: whether the base model's Swahili
is good enough as-is, and whether the fine-tune (training/) is even worth
attempting. Real numbers need llama-server and the real model; --mock
just proves the harness runs (its scores are meaningless).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.constants import AGRI_TERMS, KNOWN_CROPS  # noqa: E402
from app.config.logging import configure_logging  # noqa: E402
from app.language.swahili import SWAHILI_EVAL_SET  # noqa: E402
from app.models.language import LanguageCode  # noqa: E402
from app.orchestration.prompts import ContextBuilder, PromptBuilder  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("EvalSwahili")

_SURFACE = {**KNOWN_CROPS, **AGRI_TERMS}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score Swahili answer quality (baseline + enrichment)."
    )
    parser.add_argument(
        "--no-enrich", action="store_true", help="Disable prompt-side Swahili enrichment."
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run enrichment ON and OFF and compare."
    )
    parser.add_argument(
        "--mock", action="store_true", help="Fake LLM (offline; scores meaningless)."
    )
    args = parser.parse_args()

    configure_logging()
    llm = _make_llm(args.mock)

    if args.compare:
        off = _run(llm, enrich=False)
        on = _run(llm, enrich=True)
        print("\nSummary (term coverage / Swahili rate)")
        print("=" * 46)
        print(f"{'enrichment OFF':<24}{off[0]:>6.0%} {off[1]:>6.0%}")
        print(f"{'enrichment ON':<24}{on[0]:>6.0%} {on[1]:>6.0%}")
        print(
            "\nShip prompt-side enrichment if ON >= OFF. Only consider the "
            "fine-tune (training/) if ON is still too low."
        )
    else:
        coverage, sw_rate = _run(llm, enrich=not args.no_enrich, verbose=True)
        print(f"\nMean term coverage: {coverage:.0%} Swahili rate: {sw_rate:.0%}")

    if args.mock:
        print(
            "\n[!] MOCK run - scores are meaningless. Run against a real "
            "llama-server with the real model for the actual baseline."
        )
    return 0


def _run(llm, enrich: bool, verbose: bool = False) -> tuple[float, float]:
    builder = PromptBuilder
    empty = ContextBuilder.build([])
    detector = _language_detector

    coverages: list[float] = []
    sw_hits = 0
    for case in SWAHILI_EVAL_SET:
        prompt = builder.build(
            question=case.question,
            context=empty,
            language=LanguageCode.SWAHILI,
            intent=case.intent,
            enrich_swahili=enrich,
        )
        answer = llm.generate(prompt).text
        cov = _term_coverage(answer, case.expected_terms)
        is_sw = detector.detect(answer).language == LanguageCode.SWAHILI if answer else False
        coverages.append(cov)
        sw_hits += int(is_sw)
        if verbose:
            print(f"[{case.id}] coverage={cov:.0%} sw={is_sw}")

    mean_cov = sum(coverages) / len(coverages) if coverages else 0.0
    sw_rate = sw_hits / len(SWAHILI_EVAL_SET)
    return mean_cov, sw_rate


def _term_coverage(answer: str, expected_terms) -> float:
    """Fraction of expected concepts the answer engages (EN or SW form)."""
    if not expected_terms:
        return 1.0
    low = answer.lower()
    hits = 0
    for canonical in expected_terms:
        forms = _SURFACE.get(canonical, [canonical])
        if any(form.lower() in low for form in forms) or canonical.lower() in low:
            hits += 1
    return hits / len(expected_terms)


def _language_detector():
    from app.language.detector import LanguageDetector

    return LanguageDetector


def _make_llm(mock: bool):
    if mock:
        from app.orchestration.llm.base import BaseLLMClient, GenerationResult

        class FakeLLM(BaseLLMClient):
            def generate(self, prompt, config=None):
                return GenerationResult(
                    text="Tumia dawa ya ukungu na uondoe majani yaliyoathirika.",
                    prompt_tokens=1,
                    completion_tokens=1,
                    latency_ms=1,
                )

            def health(self):
                return True

        return FakeLLM

    from app.orchestration.llm.llama_client import LlamaClient

    client = LlamaClient
    if not client.health:
        print("[!!] llama-server is not reachable. Start it or pass --mock.")
        raise SystemExit(1)
    return client


if __name__ == "__main__":
    raise SystemExit(main)
