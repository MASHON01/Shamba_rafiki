"""Unit tests for the ingestion pipeline: loaders, cleaner, chunker,
metadata generator, hashing, duplicate detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import DocumentLoadError, EmptyDocumentError
from app.ingestion.loaders import DEFAULT_LOADERS, TextLoader
from app.ingestion.processors.chunker import Chunker
from app.ingestion.processors.cleaner import Cleaner
from app.ingestion.processors.hashing import Hasher
from app.ingestion.processors.metadata_generator import MetadataGenerator
from app.models.document import Document

pytestmark = pytest.mark.unit


# --- loaders --------------------------------------------------------------

def test_text_loader_reads_utf8(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("Nakuru maize notes.", encoding="utf-8")
    assert "Nakuru" in TextLoader().load(p)


def test_text_loader_recovers_from_bad_encoding(tmp_path: Path):
    p = tmp_path / "m.txt"
    p.write_bytes("Café mahindi".encode("cp1252"))
    # Must not raise; recovers via replacement.
    assert "mahindi" in TextLoader().load(p)


def test_default_loaders_route_by_extension(tmp_path: Path):
    txt = tmp_path / "x.txt"
    txt.write_text("hi", encoding="utf-8")
    matched = [ldr for ldr in DEFAULT_LOADERS if ldr.supports(txt)]
    assert len(matched) == 1 and isinstance(matched[0], TextLoader)


def test_loader_missing_file_raises(tmp_path: Path):
    with pytest.raises(DocumentLoadError):
        TextLoader().load(tmp_path / "nope.txt")


# --- cleaner --------------------------------------------------------------

def test_cleaner_normalizes_whitespace():
    out = Cleaner().clean("Maize   blight\n\n\n\nis   bad.")
    assert "   " not in out
    assert "\n\n\n" not in out


def test_cleaner_strips_page_numbers():
    text = "Real content here.\n\n12\n\nMore real content follows."
    out = Cleaner().clean(text)
    assert "Real content" in out and "More real content" in out


def test_cleaner_empty_input():
    assert Cleaner().clean("") == ""
    assert Cleaner().clean("   ") == ""


# --- chunker --------------------------------------------------------------

def test_chunker_respects_size():
    chunker = Chunker(chunk_size=50, chunk_overlap=10)
    words = " ".join(f"w{i}" for i in range(300))
    chunks = chunker.chunk(words)
    assert all(len(c.split()) <= 50 for c in chunks)
    assert len(chunks) > 1


def test_chunker_hard_splits_unpunctuated_text():
    # No sentence punctuation -> must still split under the cap.
    chunker = Chunker(chunk_size=20, chunk_overlap=5)
    words = " ".join(f"tok{i}" for i in range(200))
    chunks = chunker.chunk(words)
    assert all(len(c.split()) <= 20 for c in chunks)


def test_chunker_empty_and_invalid_config():
    assert Chunker().chunk("") == []
    with pytest.raises(ValueError):
        Chunker(chunk_size=50, chunk_overlap=50)


# --- hashing --------------------------------------------------------------

def test_hasher_is_content_based(tmp_path: Path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("identical", encoding="utf-8")
    b.write_text("identical", encoding="utf-8")
    h = Hasher()
    # Same content, different name -> same checksum.
    assert h.compute(a) == h.compute(b)


def test_hasher_differs_on_different_content(tmp_path: Path):
    a = tmp_path / "a.txt"; a.write_text("one", encoding="utf-8")
    b = tmp_path / "b.txt"; b.write_text("two", encoding="utf-8")
    h = Hasher()
    assert h.compute(a) != h.compute(b)


# --- metadata generator ---------------------------------------------------

def _doc(tmp_path: Path, name="kalro_maize_guide.pdf", language="en"):
    p = tmp_path / name
    p.write_text("x", encoding="utf-8")
    return Document(
        filename=name, path=p, file_type=Path(name).suffix,
        checksum="abc123", language=language, source="KALRO",
    )


def test_metadata_detects_crop_and_county(tmp_path: Path):
    doc = _doc(tmp_path)
    meta = MetadataGenerator().generate(
        doc, 0, "Maize leaf blight is common in Nakuru county."
    )
    assert meta["crop"] == "maize"
    assert meta["county"] == "Nakuru"
    assert meta["language"] == "en"
    # required keys always present
    for key in ("crop", "county", "document_type", "language"):
        assert key in meta


def test_metadata_values_are_all_strings(tmp_path: Path):
    doc = _doc(tmp_path)
    meta = MetadataGenerator().generate(doc, 3, "tomato nyanya")
    assert all(isinstance(v, str) for v in meta.values())
    assert meta["chunk_index"] == "3"