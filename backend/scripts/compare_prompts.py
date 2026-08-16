#!/usr/bin/env python3
"""
A/B two system-prompt versions on the same questions.

    python -m scripts.compare_prompts                    # show v1 vs v2 prompts
    python -m scripts.compare_prompts --a v1 --b v2      # pick versions
    python -m scripts.compare_prompts --generate         # also generate answers
    python -m scripts.compare_prompts --generate --mock  # fake LLM (offline)

Prompt wording is the single highest-leverage lever on the accuracy
score, so it should be changed with evidence, not by feel. This harness
renders the SAME questions under two prompt versions side by side, and -
with --generate - asks the model for an answer under each, so you can
eyeball the difference before the Output 7 grader scores it numerically.

Without --generate it is fully offline: it just assembles and prints the
two prompts. With --generate it needs a running llama-server (or --mock
for a smoke test whose text is meaningless).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.logging import configure_logging  # noqa: E402
from app.models.language import LanguageCode  # noqa: E402
from app.orchestration.prompts import ContextBuilder, PromptBuilder  # noqa: E402
from app.orchestration.prompts.template_registry import list_versions  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402

logger = get_logger("ComparePrompts")

# (question, intent) pairs spanning the intents. Swap in real golden
# questions once the corpus lands; these are placeholders to exercise the
# harness offline.
_SAMPLE_QUESTIONS: list[tuple[str, str]] = [
    ("My maize has yellow spots and is wilting - what is wrong?", "diagnosis"),
    ("What is the price of beans in Nakuru and should I sell now?", "price"),
    ("How and when should I plant tomatoes in Kiambu?", "how_to"),
    ("Tell me about crop rotation.", "general"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B compare two system-prompt versions on the same questions.")
    parser.add_argument("--a", default="v1", help="First prompt version.")
    parser.add_argument("--b", default="v2", help="Second prompt version.")
    parser.add_argument("--language", default="en", choices=["en", "sw"])
    parser.add_argument("--generate", action="store_true",
                        help="Also generate an answer under each version.")
    parser.add_argument("--mock", action="store_true",
                        help="Use a fake LLM (offline smoke test; text meaningless).")
    args = parser.parse_args()

    configure_logging()

    versions = list_versions()
    for v in (args.a, args.b):
        if v not in versions:
            print(f"[!!] Unknown version {v!r}. Registered: {versions}")
            return 1

    language = (
        LanguageCode.SWAHILI if args.language == "sw" else LanguageCode.ENGLISH
    )
    builder = PromptBuilder()
    # No retriever here: an empty context exercises the prompt scaffolding
    # itself (the grounding/uncertainty language), which is what differs
    # between versions. Real context comes from the Output 7 eval run.
    empty_context = ContextBuilder().build([])

    llm = _make_llm(args.mock) if args.generate else None

    for question, intent in _SAMPLE_QUESTIONS:
        print("\n" + "=" * 70)
        print(f"Q ({intent}): {question}")
        for version in (args.a, args.b):
            prompt = builder.build(
                question=question, context=empty_context,
                language=language, intent=intent, version=version,
            )
            print("\n" + "-" * 70)
            print(f"[{version}] system prompt:\n{prompt.system_prompt}")
            if llm is not None:
                result = llm.generate(prompt)
                print(f"\n[{version}] answer:\n{result.text}")

    print("\nDone. Score these with scripts/run_eval.py (Output 7) for numbers.")
    return 0


def _make_llm(mock: bool):
    if mock:
        from app.orchestration.llm.base import BaseLLMClient, GenerationResult

        class FakeLLM(BaseLLMClient):
            def generate(self, prompt, config=None):
                return GenerationResult(
                    text="[mock answer - text is meaningless without a real model]",
                    prompt_tokens=1, completion_tokens=1, latency_ms=1,
                )

            def health(self):
                return True

        return FakeLLM()

    from app.orchestration.llm.llama_client import LlamaClient

    client = LlamaClient()
    if not client.health():
        print("[!!] llama-server is not reachable. Start it or pass --mock.")
        raise SystemExit(1)
    return client


if __name__ == "__main__":
    raise SystemExit(main())
