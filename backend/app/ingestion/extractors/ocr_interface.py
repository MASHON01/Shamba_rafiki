"""
OCR interface (seam only - no engine wired in yet).

This is a deliberate extension point, not a working OCR
implementation. The Phase 1 corpus (KALRO/AFA manuals, county
context) is overwhelmingly digital-text PDFs, so OCR isn't needed
yet - but scanned/image-only pages do exist in the wild, and we
want the pipeline shaped so a real engine drops in during Phase 3
without touching anything upstream.

Phase 3 will implement `OCRInterface.extract_with_ocr()` on top of
Tesseract, EasyOCR, or PaddleOCR. Until then it raises
`OCRNotAvailableError`, and `TextExtractor` only calls it when
`OCR_FALLBACK_ENABLED` is True (default False) - so nothing here
can break Phase 1 ingestion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.core.exceptions import IngestionError
from app.utils.logger import get_logger

logger = get_logger("OCRInterface")


class OCRNotAvailableError(IngestionError):
    """Raised when OCR is requested but no OCR engine is wired in."""


@runtime_checkable
class OCREngine(Protocol):
    """
    Structural contract a Phase 3 OCR backend must satisfy to plug
    in here. A backend is any object exposing `image_to_text(path)`.
    """

    def image_to_text(self, path: Path) -> str: ...


class OCRInterface:
    """
    Stable entry point the extractor calls for scanned documents.

    In Phase 1 no engine is set, so `extract_with_ocr()` raises.
    Phase 3 constructs this with a real engine
    (`OCRInterface(engine=TesseractEngine())`) and everything
    upstream keeps calling `extract_with_ocr()` unchanged.
    """

    def __init__(self, engine: OCREngine | None = None) -> None:
        self._engine = engine

    @property
    def available(self) -> bool:
        """Whether a real OCR engine is currently wired in."""
        return self._engine is not None

    def extract_with_ocr(self, path: Path) -> str:
        """
        Extract text from an image-only / scanned document via OCR.

        Raises
        ------
        OCRNotAvailableError
            In Phase 1 (no engine). Callers must be prepared for this
            and should only reach it when OCR fallback is explicitly
            enabled.
        """
        if self._engine is None:
            raise OCRNotAvailableError(
                f"OCR was requested for '{path.name}' but no OCR engine "
                f"is available. OCR is a Phase 3 capability; enable it "
                f"only once an engine is wired into OCRInterface."
            )

        logger.info("ocr.extract.start", path=str(path))
        text = self._engine.image_to_text(path)
        logger.info(
            "ocr.extract.completed", path=str(path), chars=len(text)
        )
        return text