"""
Processing layer: the transforms between raw text and corpus-ready
chunks.

    Cleaner            raw text  -> clean prose
    Chunker            clean prose -> ~200-400 token chunks
    MetadataGenerator  chunk     -> crop/county/type/language tags
    Hasher             file      -> content checksum
    DuplicateDetector  checksum  -> already-ingested? (manifest-backed)

Each satisfies the matching Protocol in `app.ingestion.pipeline` and
is wired in by `create_default_pipeline()`.
"""

from __future__ import annotations

from app.ingestion.processors.chunker import Chunker
from app.ingestion.processors.cleaner import Cleaner
from app.ingestion.processors.duplicate_detector import DuplicateDetector
from app.ingestion.processors.hashing import Hasher
from app.ingestion.processors.metadata_generator import MetadataGenerator

__all__ = [
    "Cleaner",
    "Chunker",
    "MetadataGenerator",
    "Hasher",
    "DuplicateDetector",
]