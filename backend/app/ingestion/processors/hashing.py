"""
Document hashing.

Computes a stable SHA-256 checksum of a file's *content* (not its
name or path), so the same document under two different filenames is
still recognized as one document by the duplicate detector.

Satisfies the `Hasher` protocol in `app.ingestion.pipeline`
(`compute(path) -> str`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.core.exceptions import DocumentLoadError

#: Read the file in chunks so a large PDF never has to sit fully in
#: memory just to be hashed - matters on the 8GB target machine.
_READ_BLOCK_SIZE = 1 << 20  # 1 MiB


class Hasher:
    """
    Computes content checksums for documents.
    """

    def __init__(self, algorithm: str = "sha256") -> None:
        self._algorithm = algorithm

    def compute(self, path: Path) -> str:
        """
        Return the hex digest of `path`'s contents.

        Raises
        ------
        DocumentLoadError
            If the file can't be read.
        """
        digest = hashlib.new(self._algorithm)

        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(_READ_BLOCK_SIZE), b""):
                    digest.update(block)
        except OSError as exc:
            raise DocumentLoadError(
                f"Could not read '{path.name}' to compute checksum: {exc}"
            ) from exc

        return digest.hexdigest()