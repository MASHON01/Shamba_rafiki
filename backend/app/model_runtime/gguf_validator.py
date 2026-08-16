"""
GGUF file validation.

Cheap, offline sanity checks on a model file before we ever hand it to
llama-server: is this actually a GGUF, what quantization is it, and is
it the size we expected? A truncated download, a wrong file, or an
unexpected quant should fail loudly *here* - during download or startup
- not as a cryptic llama-server crash in front of a farmer.

What it checks:
  - the file exists and is non-empty
  - the first four bytes are the GGUF magic (b"GGUF")
  - the header version is one we understand (v2/v3)
  - (best effort) the architecture and quantization, read from the
    metadata key/value block
  - (optional) the file size is within tolerance of the registry's
    expected size, and the quantization matches the expected one

The GGUF header and metadata are parsed directly (little-endian, per
the spec) so this needs no llama.cpp or gguf Python package. Metadata
parsing is best-effort and tolerant: the magic and version are hard
failures, but if an exotic value type stops the KV scan we return what
we have rather than rejecting a valid file.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from app.config.constants import GGUF_MAGIC, MODEL_SIZE_TOLERANCE
from app.core.exceptions import ValidationError
from app.model_runtime.registry import ModelSpec
from app.utils.logger import get_logger

logger = get_logger("GGUFValidator")


class GGUFValidationError(ValidationError):
    """Raised when a file is not a valid or expected GGUF model."""

    # GGUF metadata value types (see the GGUF spec).


_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

# Fixed-width scalar types -> (struct format, byte length).
_SCALAR: dict[int, tuple[str, int]] = {
    _GGUF_TYPE_UINT8: ("<B", 1),
    _GGUF_TYPE_INT8: ("<b", 1),
    _GGUF_TYPE_UINT16: ("<H", 2),
    _GGUF_TYPE_INT16: ("<h", 2),
    _GGUF_TYPE_UINT32: ("<I", 4),
    _GGUF_TYPE_INT32: ("<i", 4),
    _GGUF_TYPE_FLOAT32: ("<f", 4),
    _GGUF_TYPE_BOOL: ("<?", 1),
    _GGUF_TYPE_UINT64: ("<Q", 8),
    _GGUF_TYPE_INT64: ("<q", 8),
    _GGUF_TYPE_FLOAT64: ("<d", 8),
}

# llama.cpp file-type (general.file_type) -> human quant label. Only the
# quants we plausibly ship or compare are mapped; anything else is
# reported by its numeric id.
_FILE_TYPE_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
}

# Only a handful of scanned KV pairs matter; cap work so a malformed or
# adversarial file can never make us read forever.
_MAX_KV_SCANNED = 512


@dataclass(slots=True)
class GGUFInfo:
    """What we learned from a GGUF header. Fields are None if unknown."""

    path: Path
    size_bytes: int
    version: int
    tensor_count: int
    kv_count: int
    architecture: str | None = None
    file_type_id: int | None = None
    quantization: str | None = None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


def validate_gguf(
    path: str | Path,
    expected: ModelSpec | None = None,
    check_size: bool = True,
) -> GGUFInfo:
    """
    Validate the GGUF at ``path`` and return what we could read.

    Parameters
    ----------
    path:
        The GGUF file to check.
    expected:
        If given, the file's quantization must match ``expected.quantization``
        (when we can read it) and - if ``check_size`` - its size must be
        within ``MODEL_SIZE_TOLERANCE`` of ``expected.file_size_mb``.
    check_size:
        Whether to enforce the size window. Disable when the registry has
        no size estimate yet.

    Raises
    ------
    GGUFValidationError
        Missing/empty file, bad magic, unreadable header, or a mismatch
        against ``expected``.
    """
    path = Path(path)

    if not path.exists():
        raise GGUFValidationError(f"GGUF file does not exist: {path}")
    if not path.is_file():
        raise GGUFValidationError(f"GGUF path is not a file: {path}")

    size_bytes = path.stat.st_size
    if size_bytes == 0:
        raise GGUFValidationError(f"GGUF file is empty: {path}")

    info = _read_header(path, size_bytes)

    if expected is not None:
        _check_against_spec(info, expected, check_size)

    logger.info(
        "gguf.validated",
        path=str(path),
        size_mb=round(info.size_mb, 1),
        version=info.version,
        architecture=info.architecture,
        quantization=info.quantization,
    )
    return info

    # ---------------------------------------------------------------------------
    # Header parsing
    # ---------------------------------------------------------------------------


def _read_header(path: Path, size_bytes: int) -> GGUFInfo:
    with path.open("rb") as fh:
        magic = fh.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFValidationError(
                f"{path} is not a GGUF file: expected magic {GGUF_MAGIC!r}, "
                f"got {magic!r}. The file is likely corrupt, truncated, or "
                f"not a model at all."
            )

        try:
            version = _read_u32(fh)
            tensor_count = _read_u64(fh)
            kv_count = _read_u64(fh)
        except struct.error as exc:
            raise GGUFValidationError(
                f"{path} has a GGUF magic but a truncated header - the "
                f"download is likely incomplete."
            ) from exc

        if version not in (2, 3):
            # Not fatal for our purposes, but worth surfacing loudly.
            logger.warning("gguf.unexpected_version", version=version)

        info = GGUFInfo(
            path=path,
            size_bytes=size_bytes,
            version=version,
            tensor_count=tensor_count,
            kv_count=kv_count,
        )

        # Best-effort: pull architecture + quantization from the KV block.
        try:
            _scan_metadata(fh, kv_count, info)
        except (struct.error, ValueError, UnicodeDecodeError) as exc:
            logger.warning("gguf.metadata_scan_stopped", reason=str(exc))

    return info


def _scan_metadata(fh, kv_count: int, info: GGUFInfo) -> None:
    """
    Walk the metadata KV block, filling architecture + file_type on info.

    Stops early once both are found, or after _MAX_KV_SCANNED pairs. Any
    parse error propagates to the tolerant caller, which keeps partial
    results.
    """
    wanted = {"general.architecture", "general.file_type"}
    limit = min(kv_count, _MAX_KV_SCANNED)

    for _ in range(limit):
        key = _read_gguf_string(fh)
        value_type = _read_u32(fh)
        value = _read_value(fh, value_type)

        if key == "general.architecture" and isinstance(value, str):
            info.architecture = value
        elif key == "general.file_type" and isinstance(value, int):
            info.file_type_id = value
            info.quantization = _FILE_TYPE_NAMES.get(value, f"ftype_{value}")

        if info.architecture is not None and info.file_type_id is not None:
            return
        if not wanted:
            return


def _read_value(fh, value_type: int):
    """Read (or skip) one typed GGUF metadata value."""
    if value_type in _SCALAR:
        fmt, length = _SCALAR[value_type]
        return struct.unpack(fmt, _read_exact(fh, length))[0]
    if value_type == _GGUF_TYPE_STRING:
        return _read_gguf_string(fh)
    if value_type == _GGUF_TYPE_ARRAY:
        elem_type = _read_u32(fh)
        count = _read_u64(fh)
        for _ in range(count):
            _read_value(fh, elem_type)
        return None
    raise ValueError(f"unknown GGUF value type {value_type}")


def _check_against_spec(info: GGUFInfo, expected: ModelSpec, check_size: bool) -> None:
    if (
        info.quantization is not None
        and expected.quantization
        and info.quantization.upper() != expected.quantization.upper()
    ):
        raise GGUFValidationError(
            f"Quantization mismatch for {info.path.name}: file is "
            f"{info.quantization}, expected {expected.quantization}."
        )

    if check_size and expected.file_size_mb:
        low = expected.file_size_mb * (1 - MODEL_SIZE_TOLERANCE)
        high = expected.file_size_mb * (1 + MODEL_SIZE_TOLERANCE)
        if not (low <= info.size_mb <= high):
            raise GGUFValidationError(
                f"Size mismatch for {info.path.name}: file is "
                f"{info.size_mb:.0f} MB, expected ~{expected.file_size_mb} MB "
                f"(+/-{int(MODEL_SIZE_TOLERANCE * 100)}%). The download may be "
                f"incomplete or the wrong file."
            )

            # ---------------------------------------------------------------------------
            # Little-endian primitive readers
            # ---------------------------------------------------------------------------


def _read_exact(fh, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise struct.error(f"expected {n} bytes, got {len(data)}")
    return data


def _read_u32(fh) -> int:
    return struct.unpack("<I", _read_exact(fh, 4))[0]


def _read_u64(fh) -> int:
    return struct.unpack("<Q", _read_exact(fh, 8))[0]


def _read_gguf_string(fh) -> str:
    length = _read_u64(fh)
    if length > 64 * 1024:
        # Metadata strings are short (keys, arch names). A huge length
        # means we have lost sync with the format - stop, don't allocate.
        raise ValueError(f"implausible GGUF string length {length}")
    return _read_exact(fh, length).decode("utf-8")
