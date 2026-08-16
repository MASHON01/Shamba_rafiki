"""
Abstract loader interface.

A loader has exactly one job: given a file path, return its raw
text. No cleaning, no chunking, no OCR fallback - that belongs to
`app.ingestion.extractors` (format/strategy selection) and
`app.ingestion.processors` (cleaning, chunking) respectively.

Every format-specific loader (PDF, DOCX, plain text, and any future
format) inherits from `BaseLoader`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseLoader(ABC):
    """
    Base class for all document loaders.

    Subclasses set `supported_extensions` and implement `load()`.
    `supports()` is provided for free from `supported_extensions`.
    """

    #: Lowercase extensions this loader handles, including the dot
    #: (e.g. (".pdf",)). Must be overridden by subclasses.
    supported_extensions: tuple[str, ...] = ()

    def supports(self, path: Path) -> bool:
        """
        Whether this loader can handle `path`, based on its extension.
        """
        return path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def load(self, path: Path) -> str:
        """
        Read `path` and return its raw text content.

        Raises
        ------
        app.core.exceptions.DocumentLoadError
            If the file cannot be opened, is corrupted, or cannot be
            parsed. Implementations should catch library-specific
            exceptions and re-raise as `DocumentLoadError` so callers
            only ever need to handle one exception type here.
        """
        raise NotImplementedError