"""
Text extractor - format selection and OCR routing.

Sits between the pipeline and the loaders. The pipeline hands it a
path; it picks the right loader (via the loaders' own `supports()`),
runs it, and returns raw text. For PDFs that come back empty or
near-empty (an image-only / scanned PDF), it optionally routes to
the OCR interface.

This "which loader? OCR or not?" decision is the one piece of logic
that doesn't belong inside any single loader - which is exactly why
it lives here rather than in `loaders/`.

Satisfies the `Extractor` protocol in `app.ingestion.pipeline`
(`extract(path) -> str`), and is the concrete class
`create_default_pipeline()` wires in.
"""

from __future__ import annotations

from pathlib import Path

from app.config.constants import (
    OCR_FALLBACK_ENABLED,
    OCR_MIN_TEXT_LENGTH,
)
from app.core.exceptions import DocumentLoadError, UnsupportedFormatError
from app.ingestion.extractors.ocr_interface import (
    OCRInterface,
    OCRNotAvailableError,
)
from app.ingestion.loaders import DEFAULT_LOADERS
from app.ingestion.loaders.base import BaseLoader
from app.utils.logger import get_logger

logger = get_logger("TextExtractor")

#: Extensions eligible for OCR fallback (only PDFs can be image-only
#: in a way OCR can recover; a .txt that's empty is just empty).
_OCR_ELIGIBLE_EXTENSIONS = (".pdf",)


class TextExtractor:
    """
    Selects a loader for each path and returns its raw text, with an
    optional OCR fallback for scanned PDFs.
    """

    def __init__(
        self,
        *,
        loaders: tuple[BaseLoader, ...] | None = None,
        ocr: OCRInterface | None = None,
        ocr_fallback_enabled: bool | None = None,
    ) -> None:
        self._loaders = loaders if loaders is not None else DEFAULT_LOADERS
        self._ocr = ocr or OCRInterface()
        self._ocr_fallback_enabled = (
            OCR_FALLBACK_ENABLED
            if ocr_fallback_enabled is None
            else ocr_fallback_enabled
        )

    def extract(self, path: Path) -> str:
        """
        Extract raw text from `path`.

        Raises
        ------
        UnsupportedFormatError
            If no loader handles this file's extension.
        DocumentLoadError
            If the selected loader fails to read the file.
        """
        loader = self._select_loader(path)

        if loader is None:
            raise UnsupportedFormatError(
                f"No loader available for '{path.name}' "
                f"(extension '{path.suffix.lower()}')."
            )

        text = loader.load(path)

        if self._should_try_ocr(path, text):
            return self._extract_with_ocr_fallback(path, text)

        return text

    def _select_loader(self, path: Path) -> BaseLoader | None:
        """First loader whose `supports()` accepts the path, else None."""
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        return None

    def _should_try_ocr(self, path: Path, text: str) -> bool:
        """
        OCR is worth attempting only for an OCR-eligible extension
        whose extracted text is empty/near-empty AND when fallback is
        enabled with an engine actually available.
        """
        if not self._ocr_fallback_enabled:
            return False
        if path.suffix.lower() not in _OCR_ELIGIBLE_EXTENSIONS:
            return False
        return len(text.strip()) < OCR_MIN_TEXT_LENGTH

    def _extract_with_ocr_fallback(self, path: Path, original_text: str) -> str:
        """
        Try OCR; if it isn't available, fall back to whatever the
        loader produced (even if empty) and let the downstream
        validator make the final empty-document call - so a missing
        OCR engine degrades gracefully instead of hard-failing.
        """
        logger.info(
            "extractor.ocr.attempt",
            path=str(path),
            extracted_chars=len(original_text.strip()),
        )
        try:
            ocr_text = self._ocr.extract_with_ocr(path)
        except OCRNotAvailableError as exc:
            logger.warning(
                "extractor.ocr.unavailable", path=str(path), error=str(exc)
            )
            return original_text
        except DocumentLoadError as exc:
            logger.warning(
                "extractor.ocr.failed", path=str(path), error=str(exc)
            )
            return original_text

        # Prefer the richer of the two results.
        return ocr_text if len(ocr_text.strip()) > len(original_text.strip()) else original_text