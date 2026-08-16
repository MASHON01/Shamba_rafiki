"""
Model metadata for /health and the benchmark reports.

One JSON-serializable snapshot of "what model is this system configured
to run, and is it actually present and resident?" - the thing the kiosk
operator sees on /health and the thing reports record
alongside the RAM/latency numbers.

It is deliberately tolerant: it never raises. A missing registry entry,
a not-yet-downloaded file, or an unknown readiness state all resolve to
honest fields (``configured: false``, ``present: false``, ``resident:
None``) rather than an exception, because /health must always answer.
"""

from __future__ import annotations

from pathlib import Path

from app.config.settings import settings
from app.model_runtime.registry import (
    ModelSpec,
    default_model,
    resolved_model_path,
)
from app.utils.logger import get_logger

logger = get_logger("ModelInfo")


def build_model_info(
    spec: ModelSpec | None = None,
    models_dir: Path | None = None,
    resident: bool | None = None,
) -> dict:
    """
    Describe the active (or given) model as a plain dict.

    Parameters
    ----------
    spec:
        The model to describe. Defaults to the configured active model
        (``settings.MODEL_ID``); if that id is not registered, returns a
        minimal ``configured: false`` payload instead of raising.
    models_dir:
        Where the GGUF is expected on disk. Defaults to ``models/``.
    resident:
        Known readiness (e.g. the LLM health flag /health already has).
        ``None`` means "unknown / not checked".

    Returns
    -------
    dict
        Registry metadata plus on-disk ``present`` and ``resident`` state.
        Always JSON-serializable, always returned.
    """
    if spec is None:
        try:
            spec = default_model
        except Exception as exc:  # noqa: BLE001 - /health must not break.
            logger.warning("model_info.unconfigured", reason=str(exc))
            return {
                "configured": False,
                "id": settings.MODEL_ID,
                "context_size": settings.MODEL_CONTEXT_SIZE,
                "resident": resident,
            }

    path = resolved_model_path(spec, models_dir)
    present = path.exists()

    return {
        "configured": True,
        "id": spec.id,
        "name": spec.display_name,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "filename": spec.filename,
        "quantization": spec.quantization,
        "params_b": spec.params_b,
        "context_size": settings.MODEL_CONTEXT_SIZE,
        "file_size_mb": spec.file_size_mb,
        "est_ram_mb": spec.est_ram_mb,
        "path": str(path),
        "present": present,
        "resident": resident,
    }
