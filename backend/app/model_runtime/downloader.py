"""
Pinned model downloader.

Fetches a registry model's GGUF from its exact Hugging Face repo +
revision + filename, then verifies it before it is ever served. "Verify,
don't trust": a download is only useful if we can prove it is the right,
complete file.

Verification is two-layered:
  1. Structural - the GGUF validator confirms magic bytes, quantization,
     and size (catches truncated or wrong files).
  2. Cryptographic - if the registry pins a SHA256, the file's digest
     must match exactly. If the registry has no SHA256 yet, we compute
     and log it, so it can be pinned for future reproducible builds.

`huggingface_hub` is imported lazily, exactly like the LlamaClient's
`requests`: importing this module never requires it, so the app stays
importable on a machine that has not set up model downloads. It rides in
transitively via sentence-transformers, so it is normally already there.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.config import paths
from app.core.exceptions import ShambaRafikiError
from app.model_runtime.gguf_validator import validate_gguf
from app.model_runtime.registry import ModelSpec, resolved_model_path
from app.utils.logger import get_logger

logger = get_logger("ModelDownloader")

_SHA_CHUNK = 1024 * 1024  # 1 MiB read blocks for hashing.


class ModelDownloadError(ShambaRafikiError):
    """Raised when a model cannot be downloaded or fails verification."""


def download_model(
    spec: ModelSpec,
    models_dir: Path | None = None,
    force: bool = False,
    verify: bool = True,
) -> Path:
    """
    Download ``spec``'s GGUF to ``models/<filename>`` and verify it.

    Parameters
    ----------
    spec:
        The registry entry to fetch (repo, revision, filename are pinned).
    models_dir:
        Destination directory. Defaults to the repo-root ``models/``.
    force:
        Re-download even if a verified file already exists on disk.
    verify:
        Run structural + checksum verification after the download. Leave
        on except in tests.

    Returns
    -------
    Path
        Absolute path to the verified GGUF.

    Raises
    ------
    ModelDownloadError
        If huggingface_hub is unavailable, the download fails, or
        verification does not pass.
    """
    dest_dir = models_dir if models_dir is not None else paths.MODELS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = resolved_model_path(spec, dest_dir)

    if target.exists() and not force:
        logger.info("model.download.cached", path=str(target))
        if verify:
            _verify(target, spec)
        return target

    hf_hub_download = _load_hf_download

    logger.info(
        "model.download.start",
        model=spec.id,
        repo=spec.repo_id,
        revision=spec.revision,
        filename=spec.filename,
    )

    try:
        downloaded = hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=str(dest_dir),
        )
    except Exception as exc:  # noqa: BLE001 - normalize any HF error.
        raise ModelDownloadError(
            f"Failed to download {spec.filename} from {spec.repo_id} " f"@ {spec.revision}: {exc}"
        ) from exc

    downloaded_path = Path(downloaded)
    # hf_hub_download may nest under the repo layout; normalize to the
    # canonical models/<filename> location the rest of the code expects.
    if downloaded_path.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        downloaded_path.replace(target)

    logger.info(
        "model.download.done",
        path=str(target),
        size_mb=round(target.stat.st_size / (1024 * 1024), 1),
    )

    if verify:
        _verify(target, spec)

    return target


def sha256_of(path: str | Path) -> str:
    """Streaming SHA256 of a file, in 1 MiB blocks (memory-bounded)."""
    digest = hashlib.sha256
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(_SHA_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()

    # ---------------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------------


def _verify(path: Path, spec: ModelSpec) -> None:
    """Structural GGUF check, then checksum pin (or record if unpinned)."""
    validate_gguf(path, expected=spec)

    digest = sha256_of(path)
    if spec.sha256:
        if digest.lower() != spec.sha256.lower():
            raise ModelDownloadError(
                f"Checksum mismatch for {path.name}: got {digest}, "
                f"expected {spec.sha256}. The file is corrupt or was "
                f"tampered with - do not serve it."
            )
        logger.info("model.verify.checksum_ok", model=spec.id)
    else:
        # No pin yet: record the digest so it can be added to the
        # registry for reproducible, verified future downloads.
        logger.info(
            "model.verify.checksum_unpinned",
            model=spec.id,
            sha256=digest,
            hint="Add this sha256 to the registry entry to pin it.",
        )


def _load_hf_download():
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download
    except ImportError as exc:
        raise ModelDownloadError(
            "huggingface_hub is required to download models. Install it "
            "with: pip install huggingface_hub (it normally ships with "
            "sentence-transformers)."
        ) from exc
