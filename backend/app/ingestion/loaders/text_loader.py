"""
Plain text loader.

Handles .txt, .md, and .csv. Useful during development (dropping in
quick test docs) and for source data that's already plain text -
e.g. a manually cleaned KAMIS price export or a hand-written county
crop-calendar note.
"""

from __future__ import annotations

from pathlib import Path

from app.config.constants import DEFAULT_ENCODING
from app.core.exceptions import DocumentLoadError
from app.ingestion.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    """
    Loads content from plain text-like files.
    """

    supported_extensions = (".txt", ".md", ".csv")

    def load(self, path: Path) -> str:
        try:
            return path.read_text(encoding=DEFAULT_ENCODING)
        except UnicodeDecodeError:
            # Some field-collected .txt/.csv files won't be clean
            # UTF-8. Fall back to replacing bad bytes rather than
            # failing the whole document over a handful of characters.
            try:
                return path.read_text(
                    encoding=DEFAULT_ENCODING, errors="replace"
                )
            except OSError as exc:
                raise DocumentLoadError(
                    f"Could not read text file '{path.name}': {exc}"
                ) from exc
        except OSError as exc:
            raise DocumentLoadError(
                f"Could not read text file '{path.name}': {exc}"
            ) from exc