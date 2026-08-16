"""
Crop name normalization.

Maps any surface form a farmer might use - English, Swahili, plural,
or common synonym - to the canonical crop label the rest of the
system keys on:

    "mahindi" / "corn" / "maize"  -> "maize"
    "nyanya" / "tomatoes"         -> "tomato"

This is the single source of truth for "what crop is this word",
built directly from the KNOWN_CROPS vocabulary already used by
ingestion metadata and entity extraction - so a crop known to one
part of the system is known identically everywhere.

Why its own module rather than inline dict lookups: crop vocabulary
grows (new synonyms, dialect forms, misspellings), and the reverse
surface-form -> canonical index is worth building once and sharing.
The entity extractor, translator fallback, and retrieval filtering
all normalize crop words the same way through here.
"""

from __future__ import annotations

from app.config.constants import KNOWN_CROPS


class CropNormalizer:
    """
    Surface form -> canonical crop label.
    """

    def __init__(self) -> None:
        # Reverse index: every lowercase surface form -> canonical.
        self._surface_to_canonical: dict[str, str] = {}
        for canonical, surface_forms in KNOWN_CROPS.items():
            self._surface_to_canonical[canonical.lower()] = canonical
            for form in surface_forms:
                self._surface_to_canonical[form.lower()] = canonical

    def normalize(self, word: str) -> str | None:
        """
        Return the canonical crop for `word`, or None if it isn't a
        known crop. Matching is case-insensitive and whitespace-tolerant.
        """
        if not word:
            return None
        return self._surface_to_canonical.get(word.strip().lower())

    def is_crop(self, word: str) -> bool:
        return self.normalize(word) is not None

    @property
    def canonical_crops(self) -> list[str]:
        """All canonical crop labels, for callers that need the set."""
        return list(KNOWN_CROPS.keys())