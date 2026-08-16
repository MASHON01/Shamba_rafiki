"""
Extraction layer: format selection + OCR routing.

`TextExtractor` turns a path into raw text by picking the right
loader and, for scanned PDFs, optionally routing to `OCRInterface`.
The OCR engine itself is a Phase 3 concern - `OCRInterface` is a
seam that raises until an engine is wired in.
"""

from __future__ import annotations

from app.ingestion.extractors.ocr_interface import (
    OCREngine,
    OCRInterface,
    OCRNotAvailableError,
)
from app.ingestion.extractors.text_extractor import TextExtractor

__all__ = [
    "TextExtractor",
    "OCRInterface",
    "OCREngine",
    "OCRNotAvailableError",
]