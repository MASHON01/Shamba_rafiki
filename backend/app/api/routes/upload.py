"""
Upload route - add a document to the corpus (save-only).

    POST /upload   (multipart file)  ->  saved path + next step

By design, indexing happens OFFLINE at build time, never on the
kiosk - running the embedder on the kiosk is exactly what the RAM
budget forbids. So this endpoint only *saves* an uploaded document
into the raw-documents directory and tells the operator to run the
build script to ingest and index it. It deliberately does not touch
the live index.

Validation before saving:
- extension must be a supported document type,
- size must be within MAX_UPLOAD_BYTES (guards disk, not RAM),
- filename is made filesystem-safe (reusing the ingestion util).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.config.constants import SUPPORTED_DOCUMENT_TYPES
from app.config.paths import RAW_DOCUMENTS_DIR
from app.config.settings import settings
from app.core.exceptions import UnsupportedFormatError, ValidationError
from app.core.responses import success_response
from app.ingestion.utils import ensure_directory, safe_filename
from app.utils.logger import get_logger

logger = get_logger("UploadRoute")

router = APIRouter(tags=["Upload"])


@router.post("/upload")
def upload(file: UploadFile = File(...)) -> dict:
    """
    Save an uploaded corpus document for offline indexing.
    """
    original_name = file.filename or "upload"
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_DOCUMENT_TYPES:
        raise UnsupportedFormatError(
            f"Unsupported file type '{extension}'. Supported: "
            f"{', '.join(SUPPORTED_DOCUMENT_TYPES)}."
        )

    # Read the file, enforcing the size cap as we go rather than
    # trusting a client-supplied Content-Length.
    contents = file.file.read()
    if len(contents) > settings.MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"File exceeds the maximum upload size of "
            f"{settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )
    if not contents:
        raise ValidationError("Uploaded file is empty.")

    ensure_directory(RAW_DOCUMENTS_DIR)
    safe_name = safe_filename(original_name)
    dest = RAW_DOCUMENTS_DIR / safe_name

    # Avoid silently overwriting an existing document with the same name.
    dest = _dedupe_path(dest)

    dest.write_bytes(contents)

    logger.info(
        "upload.saved",
        filename=safe_name,
        bytes=len(contents),
        path=str(dest),
    )

    return success_response(
        data={
            "saved_as": dest.name,
            "size_bytes": len(contents),
            "indexed": False,
        },
        message=(
            "File saved. Run the build_index script to ingest and index "
            "it into the corpus."
        ),
    )


def _dedupe_path(path: Path) -> Path:
    """If `path` exists, append -1, -2, ... to the stem until it doesn't."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1