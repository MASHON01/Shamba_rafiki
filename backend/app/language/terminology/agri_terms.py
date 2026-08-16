"""
Agricultural terminology mapping.

The bilingual glossary for pests, diseases, symptoms, and inputs -
canonical English term plus every Swahili/English surface form:

    "ukungu" / "baa"  -> "blight"
    "kuoza" / "uozo"  -> "rot"
    "dawa"            -> "pesticide"

Built from the AGRI_TERMS vocabulary. Two consumers rely on it:
the entity extractor (to tag agri-term entities) and the translator's
dictionary fallback (to gloss Swahili domain words into English so an
English-only retrieval corpus still gets hit by a Swahili query).

Kept separate from crops.py because agricultural terminology grows on
its own axis (new diseases, new inputs) independent of the crop list,
and because the translator needs a term-level Swahili->English gloss
that the crop normalizer doesn't provide.
"""

from __future__ import annotations

from app.config.constants import AGRI_TERMS


class AgriTerminology:
    """
    Surface form <-> canonical agricultural term mapping.
    """

    def __init__(self) -> None:
        # Reverse index: lowercase surface form -> canonical English term.
        self._surface_to_canonical: dict[str, str] = {}
        for canonical, surface_forms in AGRI_TERMS.items():
            self._surface_to_canonical[canonical.lower()] = canonical
            for form in surface_forms:
                self._surface_to_canonical[form.lower()] = canonical

    def normalize(self, phrase: str) -> str | None:
        """
        Return the canonical English term for `phrase`, or None if it
        isn't a known agricultural term.
        """
        if not phrase:
            return None
        return self._surface_to_canonical.get(phrase.strip().lower())

    def is_term(self, phrase: str) -> bool:
        return self.normalize(phrase) is not None

    def gloss_token(self, token: str) -> str | None:
        """
        Translator helper: return the canonical English term for a
        single token if it's a known agri-term, else None. Used by the
        dictionary-fallback translator to replace Swahili domain words
        with their English equivalents.
        """
        return self.normalize(token)

    @property
    def surface_forms(self) -> dict[str, str]:
        """The full surface-form -> canonical index (read-only copy)."""
        return dict(self._surface_to_canonical)