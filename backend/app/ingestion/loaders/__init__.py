"""
Format-specific document loaders.

Each loader's only job is "read this file, return its raw text."
Format selection (which loader handles a given path) and any OCR
fallback belong to `app.ingestion.extractors`, not here.
"""

from __future__ import annotations

from app.ingestion.loaders.base import BaseLoader
from app.ingestion.loaders.docx_loader import DocxLoader
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.text_loader import TextLoader

__all__ = [
    "BaseLoader",
    "PDFLoader",
    "DocxLoader",
    "TextLoader",
    "DEFAULT_LOADERS",
]

#: Loaders tried in order for a given path - order only matters in
#: that the first one whose `.supports(path)` is True wins. Kept
#: here (rather than duplicated in extractors/text_extractor.py) so
#: adding a future format means updating one place.
DEFAULT_LOADERS: tuple[BaseLoader, ...] = (
    PDFLoader(),
    DocxLoader(),
    TextLoader(),
)