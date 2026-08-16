"""
Duplicate detection.

Tracks which documents (by content checksum) are already in the
corpus, so re-running ingestion doesn't index the same KALRO/AFA PDF
twice.

Crucially, this hydrates its checksum set from the existing corpus
manifest on construction. That closes the cross-restart gap noted
when `builder.py` was built: because `builder.py` merges manifest
entries by `document_id` and every freshly-constructed `Document`
gets a new random `document_id`, the *builder* alone can't recognize
a re-submitted file across process restarts. Checksum hydration here
is what makes duplicate detection survive a restart.

Satisfies the `DuplicateDetector` protocol in
`app.ingestion.pipeline` (`is_duplicate()` / `register()`).
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

from app.config.paths import PROCESSED_DOCUMENTS_DIR
from app.utils.logger import get_logger

logger = get_logger("DuplicateDetector")

MANIFEST_FILENAME = "manifest.json"


class DuplicateDetector:
    """
    In-memory checksum set, seeded from the on-disk corpus manifest.
    """

    def __init__(self, corpus_dir: Path | None = None) -> None:
        self._corpus_dir = corpus_dir or PROCESSED_DOCUMENTS_DIR
        self._seen_checksums: set[str] = set()
        self._hydrate_from_manifest()

    def is_duplicate(self, checksum: str) -> bool:
        """Whether a document with this checksum is already known."""
        return checksum in self._seen_checksums

    def register(self, checksum: str, document_id: UUID) -> None:
        """
        Record a checksum as ingested. `document_id` is accepted to
        satisfy the pipeline contract and aid debugging/logging; dedup
        itself keys on checksum (content), not the random document_id.
        """
        self._seen_checksums.add(checksum)
        logger.debug(
            "duplicate_detector.registered",
            checksum=checksum,
            document_id=str(document_id),
        )

    @property
    def known_count(self) -> int:
        """Number of distinct checksums currently tracked."""
        return len(self._seen_checksums)

    def _hydrate_from_manifest(self) -> None:
        """
        Seed the checksum set from the existing manifest, if any.
        A missing or unreadable manifest is not an error - it just
        means an empty corpus / fresh start.
        """
        manifest_path = self._corpus_dir / MANIFEST_FILENAME

        if not manifest_path.exists():
            logger.debug(
                "duplicate_detector.no_manifest",
                manifest_path=str(manifest_path),
            )
            return

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "duplicate_detector.manifest_unreadable",
                manifest_path=str(manifest_path),
                error=str(exc),
            )
            return

        for entry in manifest.get("documents", []):
            checksum = entry.get("checksum")
            if checksum:
                self._seen_checksums.add(checksum)

        logger.info(
            "duplicate_detector.hydrated",
            known_documents=len(self._seen_checksums),
        )