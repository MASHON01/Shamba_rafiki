"""
PDF loader.

Uses PyMuPDF (fitz) rather than pdfplumber - faster on the KALRO/
AFA manual PDFs, and it's the same library the OCR fallback path
(app.ingestion.extractors.ocr_interface, Phase 3) will rasterize
pages with, so there's one PDF dependency in the stack, not two.
"""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import DocumentLoadError
from app.ingestion.loaders.base import BaseLoader
from app.utils.logger import get_logger

logger = get_logger("PDFLoader")


class PDFLoader(BaseLoader):
    """
    Loads text content from PDF files using PyMuPDF.
    """

    supported_extensions = (".pdf",)

    def load(self, path: Path) -> str:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise DocumentLoadError(
                "PyMuPDF is required to load PDF files. "
                "Install it with: pip install pymupdf"
            ) from exc

        try:
            document = fitz.open(path)
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not open PDF '{path.name}': {exc}"
            ) from exc

        try:
            with document:
                if document.is_encrypted:
                    raise DocumentLoadError(
                        f"PDF '{path.name}' is password-protected; "
                        f"cannot extract text without a password."
                    )

                if document.page_count == 0:
                    raise DocumentLoadError(
                        f"PDF '{path.name}' has no pages."
                    )

                pages_text = []
                for page_number in range(document.page_count):
                    try:
                        page = document.load_page(page_number)
                        pages_text.append(page.get_text())
                    except Exception as exc:
                        # One bad page shouldn't sink the whole document -
                        # log it and keep going with the pages we can read.
                        logger.warning(
                            "pdf_loader.page.failed",
                            path=str(path),
                            page_number=page_number,
                            error=str(exc),
                        )

                return "\n\n".join(pages_text)

        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(
                f"Could not extract text from PDF '{path.name}': {exc}"
            ) from exc