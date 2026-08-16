#!/usr/bin/env python3
"""
Build the vector index from the processed corpus.

Run at build time (Day 8), after ingestion has produced a corpus
manifest and before the kiosk is expected to answer queries. This is
a thin CLI over `app.retrieval.Indexer` - all the real work lives
there; this file only parses arguments, invokes the build, and
prints a readable summary.

Usage:
    python -m scripts.build_index
    python -m scripts.build_index --backend faiss
    python -m scripts.build_index --corpus-dir data/processed_documents \\
                                  --store-dir data/vector_store
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the backend package importable without requiring `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config.constants import VECTOR_STORE_BACKEND  # noqa: E402
from app.retrieval import Indexer  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Shamba Rafiki vector index from the "
        "processed corpus."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Directory containing manifest.json (defaults to the "
        "configured processed-documents dir).",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=None,
        help="Where to write the vector index (defaults to the "
        "configured vector-store dir).",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "faiss"],
        default=VECTOR_STORE_BACKEND,
        help=f"Vector store backend (default: {VECTOR_STORE_BACKEND}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print(f"Building vector index (backend={args.backend})...")

    indexer = Indexer(backend=args.backend)
    response = indexer.build(
        corpus_dir=args.corpus_dir,
        store_dir=args.store_dir,
    )

    if not response["success"]:
        message = response.get("error", {}).get("message", "Unknown error")
        print(f"\n  FAILED: {message}", file=sys.stderr)
        return 1

    data = response["data"]
    print("\n  Done.")
    print(f"    Documents indexed : {data['documents_indexed']}")
    print(f"    Chunks indexed    : {data['indexed_chunks']}")
    print(f"    Store directory   : {data['store_dir']}")
    if data.get("documents_failed"):
        print(f"    Documents skipped : {data['documents_failed']}")
        for failure in data.get("failures", []):
            print(f"      - {failure['document']}: {failure['error']}")
    if data.get("embedding_cache_stats"):
        stats = data["embedding_cache_stats"]
        print(
            f"    Embedding cache   : {stats['hits']} hits, "
            f"{stats['misses']} misses"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())