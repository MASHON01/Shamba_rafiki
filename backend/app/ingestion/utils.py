"""
Shared helpers for the ingestion pipeline.

Small, dependency-free utilities used across loaders, extractors, and
processors. Intentionally narrow: anything app-wide (logging, cache)
lives in `app.utils`, and directory bootstrapping for configured
paths lives in `app.config.paths.create_directories`. This module is
for the incidental file/text operations the ingestion stages repeat.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from app.config.constants import DEFAULT_ENCODING
from app.core.exceptions import DocumentLoadError

_WHITESPACE_RE = re.compile(r"\s+")
_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_{2,}")


def read_file(path: Path, *, encoding: str = DEFAULT_ENCODING) -> str:
    """
    Read a text file, tolerating imperfect encodings.

    Field-collected .txt/.csv files aren't always clean UTF-8, so
    undecodable bytes are replaced rather than raising - mirrors
    TextLoader's behavior so anything using this helper degrades the
    same way.

    Raises
    ------
    DocumentLoadError
        If the file can't be read from disk at all.
    """
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding=encoding, errors="replace")
        except OSError as exc:
            raise DocumentLoadError(
                f"Could not read '{path.name}': {exc}"
            ) from exc
    except OSError as exc:
        raise DocumentLoadError(f"Could not read '{path.name}': {exc}") from exc


def read_bytes(path: Path) -> bytes:
    """
    Read a file's raw bytes.

    Raises
    ------
    DocumentLoadError
        If the file can't be read.
    """
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Could not read '{path.name}': {exc}") from exc


def ensure_directory(path: Path) -> Path:
    """
    Create `path` (and parents) if absent; return it for chaining.

    Raises
    ------
    DocumentLoadError
        If the directory can't be created.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DocumentLoadError(
            f"Could not create directory '{path}': {exc}"
        ) from exc
    return path


def normalize_whitespace(text: str) -> str:
    """
    Collapse any run of whitespace (spaces, tabs, newlines) to a
    single space and strip the ends. For inline normalization; the
    Cleaner does the heavier, structure-aware cleaning.
    """
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def safe_filename(name: str, *, max_length: int = 200) -> str:
    """
    Turn an arbitrary string into a filesystem-safe filename.

    Normalizes unicode to ASCII, replaces unsafe characters with
    underscores, collapses repeats, and trims length while preserving
    the extension. Useful when deriving output filenames from document
    titles or source names.
    """
    if not name or not name.strip():
        return "unnamed"

    # Normalize accents/diacritics down to ASCII (e.g. "Café" -> "Cafe").
    normalized = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    stem = Path(normalized).stem
    suffix = Path(normalized).suffix

    stem = _UNSAFE_FILENAME_RE.sub("_", stem)
    stem = _MULTI_UNDERSCORE_RE.sub("_", stem).strip("_.")
    suffix = _UNSAFE_FILENAME_RE.sub("", suffix)

    if not stem:
        stem = "unnamed"

    # Trim the stem so stem+suffix fits within max_length.
    allowed_stem = max(1, max_length - len(suffix))
    stem = stem[:allowed_stem]

    return f"{stem}{suffix}"


def file_extension(path: Path) -> str:
    """Lowercased file extension including the dot (e.g. '.pdf')."""
    return path.suffix.lower()


def token_count(text: str) -> int:
    """
    Whitespace-word token approximation, shared by the chunker and
    pipeline so chunk sizes and reported token counts stay consistent
    until the real embedding tokenizer arrives in Output 4.
    """
    return len(text.split())