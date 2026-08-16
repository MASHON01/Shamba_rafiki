"""
Domain terminology: the canonical vocabulary the language layer maps to.

    CropNormalizer    surface form -> canonical crop  (mahindi -> maize)
    AgriTerminology   surface form -> canonical term  (ukungu -> blight)

Both are reverse indexes built from the shared KNOWN_CROPS / AGRI_TERMS
constants, so a word means the same thing here as it does to ingestion
metadata and retrieval filtering.
"""

from __future__ import annotations

from app.language.terminology.agri_terms import AgriTerminology
from app.language.terminology.crops import CropNormalizer

__all__ = [
    "CropNormalizer",
    "AgriTerminology",
]