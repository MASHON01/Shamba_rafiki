#!/usr/bin/env python3
"""
Probe the vector index: print the RAW similarity scores for a query.

Diagnostic for tuning SIMILARITY_THRESHOLD against the real embedder and
corpus. It loads the pre-built index and retrieves with the threshold
DISABLED (score_threshold=0.0), so you see the actual cosine scores of
the top hits - including ones the live gate would currently discard.

    python scripts/probe_retrieval.py "My maize has brown spots on the leaves and is wilting."
    python scripts/probe_retrieval.py --top-k 10 "beans aphid control"
    python scripts/probe_retrieval.py --lang sw "mahindi yana madoa ya kahawia"

Read the scores it prints, then set SIMILARITY_THRESHOLD in .env to just
BELOW the cluster of genuinely-relevant hits (paraphrase-multilingual
MiniLM typically scores relevant passages ~0.35-0.55, so 0.60 is usually
too strict). Run from the repository root, in the venv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the backend package importable without requiring `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config.logging import configure_logging  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.retrieval.search.retriever import Retriever  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print raw similarity scores for a query (threshold disabled)."
    )
    parser.add_argument("query", type=str, help="The query text to probe.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many hits to show (default: 10).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "sw"],
        help="Query language label (for your reference only).",
    )
    parser.add_argument(
        "--chars",
        type=int,
        default=140,
        help="Characters of each chunk to preview (default: 140).",
    )
    args = parser.parse_args(argv)

    configure_logging()

    print("Loading index + embedder (first run downloads/loads the model)...")
    retriever = Retriever.from_disk()
    print(f"Index: {len(retriever)} chunks.  Live SIMILARITY_THRESHOLD = {settings.SIMILARITY_THRESHOLD}\n")

    # Threshold disabled: show everything the ranker returns, scores and all.
    hits = retriever.retrieve(args.query, top_k=args.top_k, score_threshold=0.0)

    if not hits:
        print("No hits at all - the index may be empty or the query embedded to nothing.")
        return 1

    print(f'Top {len(hits)} hits for: "{args.query}"')
    print("=" * 78)
    kept = 0
    for rank, hit in enumerate(hits, start=1):
        score = hit.similarity_score
        passes = score >= settings.SIMILARITY_THRESHOLD
        if passes:
            kept += 1
        flag = "KEEP" if passes else "drop"
        source = hit.chunk.metadata.get("source", "?")
        preview = " ".join(hit.chunk.text.split())[: args.chars]
        print(f"[{rank:>2}] score={score:.3f}  {flag}  source={source}")
        print(f"     {preview}...")
    print("=" * 78)
    print(
        f"At the current threshold ({settings.SIMILARITY_THRESHOLD}), "
        f"{kept}/{len(hits)} of these would reach the LLM."
    )
    if kept == 0:
        best = hits[0].similarity_score
        print(
            f"\n>> Nothing passes. The best hit scored {best:.3f}. If it looks relevant, "
            f"set SIMILARITY_THRESHOLD in .env to ~{max(best - 0.05, 0.0):.2f} and re-test."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
