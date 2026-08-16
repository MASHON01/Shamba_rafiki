#!/usr/bin/env python3
"""
Ingest raw corpus documents into the processed corpus.

Turns the PDF / DOCX / TXT files under data/raw_documents/ into the
processed corpus (data/processed_documents/manifest.json) that
build_index.py then embeds into the vector index. This is the FIRST of
the two corpus build steps:

    python scripts/ingest.py        # raw_documents  -> processed_documents
    python scripts/build_index.py   # processed_documents -> vector_store

A single bad document does not abort the run: its failure is reported and
the rest of the batch proceeds, so a folder of KALRO/AFA PDFs doesn't need
to be perfect to make progress.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --source KALRO --language en
    python scripts/ingest.py --raw-dir data/raw_documents --source KAMIS

`--source` is a provenance label stored on every chunk (e.g. KALRO, AFA,
KAMIS); ingest each provider's documents in its own run so the label is
accurate. Run from the repository root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the backend package importable without requiring `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.config.logging import configure_logging  # noqa: E402
from app.config.settings import settings  # noqa: E402
from app.ingestion.pipeline import create_default_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest raw documents into the processed corpus."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory of raw documents (default: the configured "
        "raw_documents dir).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="KALRO",
        help="Provenance label stored on every chunk (e.g. KALRO, AFA, KAMIS).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "sw"],
        help="Language of the documents in this batch.",
    )
    args = parser.parse_args(argv)

    configure_logging()

    raw_dir = args.raw_dir or settings.raw_documents_path
    print(
        f"Ingesting from {raw_dir} "
        f"(source={args.source}, language={args.language})..."
    )

    pipeline = create_default_pipeline()
    response = pipeline.run(raw_dir, source=args.source, language=args.language)

    if not response["success"]:
        message = response.get("error", {}).get("message", "Unknown error")
        print(f"\n  FAILED: {message}", file=sys.stderr)
        return 1

    data = response["data"]
    print("\n  Done.")
    print(f"    Documents processed : {data['documents_processed']}")
    print(f"    Documents failed    : {data['documents_failed']}")
    print(f"    Chunks written      : {data['chunks_written']}")
    print(
        f"    Corpus totals       : {data['total_documents_in_corpus']} docs, "
        f"{data['total_chunks_in_corpus']} chunks"
    )
    print(f"    Manifest            : {data['manifest_path']}")
    for failure in data.get("failures", []):
        print(f"      - {failure['path']}: {failure['error']}")

    print("\n  Next: python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
