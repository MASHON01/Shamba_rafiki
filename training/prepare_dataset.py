#!/usr/bin/env python3
"""
Build a small Swahili agricultural instruction dataset for fine-tuning.

    python prepare_dataset.py # -> training/data/swahili_agri.jsonl
    python prepare_dataset.py --out my.jsonl

Emits chat-style instruction/response pairs (JSONL) grounded in the
project's own vocabulary and eval set, so a fine-tune reinforces exactly
the terms and answer shapes the system uses. This is a SEED set - expand
it with real KALRO/AFA-derived Q/A before a serious run; a few dozen
hand-checked examples beat a large noisy set.

Standalone: run from the training/ directory. It reads the committed
vocabulary and Swahili eval set from the backend package; it writes only
to training/data/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the backend package importable when run from training/.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from app.language.swahili import SWAHILI_EVAL_SET  # noqa: E402
from app.language.swahili.fewshot_examples import fewshot_block  # noqa: E402

_SYSTEM = (
    "Wewe ni Shamba Rafiki, mshauri wa kilimo kwa wakulima wadogo nchini "
    "Kenya. Jibu kwa Kiswahili wazi, kifupi na cha vitendo, ukieleza hatua "
    "za kuchukua."
)


def build_examples() -> list[dict]:
    """
    Seed instruction/response pairs from the eval questions.

    The 'response' here is a STYLE scaffold (the intent exemplar), not a
    verified answer - replace these with real, source-checked answers
    before training a model you intend to ship.
    """
    examples: list[dict] = []
    for case in SWAHILI_EVAL_SET:
        response = fewshot_block(case.intent)
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": case.question},
                    {"role": "assistant", "content": response},
                ],
                "meta": {"id": case.id, "intent": case.intent},
            }
        )
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the SW fine-tune seed set.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "swahili_agri.jsonl"),
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    examples = build_examples
    with out.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} seed examples -> {out}")
    print(
        "NOTE: replace the scaffold responses with real, source-checked "
        "answers before a serious fine-tune run."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main)
