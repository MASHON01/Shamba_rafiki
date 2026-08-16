"""
Classify route: the leaf-photo endpoint.

    POST /classify   (multipart)  file[, query, language, session_id]
        -> classification + grounded, verified advisory answer

The image counterpart to /chat. A farmer uploads a leaf photo
(optionally with a text question); the orchestrator classifies it,
folds the predicted crop + disease into the RAG retrieval, and returns
the same response shape as /chat plus a `classification` block. The
uploaded bytes are staged in a short-lived temp file and deleted.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_orchestrator
from app.config.constants import SUPPORTED_IMAGE_TYPES
from app.config.settings import settings
from app.core.exceptions import UnsupportedFormatError, ValidationError
from app.models.request import ImageRequest
from app.orchestration.orchestrator import Orchestrator
from app.utils.logger import get_logger

logger = get_logger("ClassifyRoute")

router = APIRouter(tags=["Classify"])


@router.post("/classify")
def classify(
    file: UploadFile = File(..., description="A leaf photo (.jpg/.jpeg/.png)."),
    query: str | None = Form(None, description="Optional text question about the plant."),
    language: str = Form("en", description="ISO language hint ('en' or 'sw')."),
    session_id: str | None = Form(None, description="Optional session id for follow-ups."),
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> dict:
    """Classify a leaf photo and return grounded advice."""
    original_name = file.filename or "upload"
    extension = Path(original_name).suffix.lower()

    if extension not in SUPPORTED_IMAGE_TYPES:
        raise UnsupportedFormatError(
            f"Unsupported image type '{extension}'. Supported: "
            f"{', '.join(SUPPORTED_IMAGE_TYPES)}."
        )

    contents = file.file.read()
    if not contents:
        raise ValidationError("Uploaded image is empty.")
    if len(contents) > settings.MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"Image exceeds the maximum upload size of "
            f"{settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    fd, tmp_path = tempfile.mkstemp(suffix=extension)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(contents)
        request = ImageRequest(
            image_path=tmp_path,
            query=query,
            language=language,
            session_id=session_id,
        )
        logger.info(
            "classify.received",
            filename=original_name,
            bytes=len(contents),
            has_text=bool(query),
            language=language,
        )
        return orchestrator.handle_image_query(request)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
