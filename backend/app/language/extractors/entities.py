"""
Entity extractor.

Pulls the agricultural "nouns" out of a query - which crop, which
county, which pest/disease/input the farmer is asking about - as a
list of `AgriculturalEntity` objects. This is what lets the
orchestrator filter retrieval ("only maize docs") and lets the
prompt name the specific crop/disease.

Matching is vocabulary-driven and bilingual: the same KNOWN_CROPS /
KNOWN_COUNTIES tables the ingestion metadata generator uses, plus
AGRI_TERMS for pests/diseases/inputs. Because those tables carry
Swahili surface forms, "nyanya" and "tomato" both resolve to the
canonical crop `tomato` - the key to cross-language retrieval.

Each entity records:
    entity            the surface form actually found in the text
    entity_type       crop | county | agri_term
    normalized_value  the canonical label (what retrieval filters on)
    confidence        1.0 for an exact vocabulary hit

Duplicate detections of the same (type, normalized_value) collapse
to one entity, keeping the first surface form seen.
"""

from __future__ import annotations

from app.config.constants import (
    AGRI_TERMS,
    ENTITY_TYPE_AGRI_TERM,
    ENTITY_TYPE_COUNTY,
    ENTITY_TYPE_CROP,
    KNOWN_COUNTIES,
    KNOWN_CROPS,
)
from app.language.normalizer import InputNormalizer
from app.models.language import AgriculturalEntity


class EntityExtractor:
    """
    Vocabulary-driven bilingual agricultural entity extractor.
    """

    def __init__(self, normalizer: InputNormalizer | None = None) -> None:
        self._normalizer = normalizer or InputNormalizer()

    def extract(self, text: str) -> list[AgriculturalEntity]:
        if not text or not text.strip():
            return []

        normalized = self._normalizer.normalize(text)

        entities: list[AgriculturalEntity] = []
        seen: set[tuple[str, str]] = set()

        # Order: crops, counties, agri-terms. County names are
        # lowercased for matching (KNOWN_COUNTIES keys are display-case).
        self._match_table(
            normalized, KNOWN_CROPS, ENTITY_TYPE_CROP, entities, seen
        )
        self._match_table(
            normalized, KNOWN_COUNTIES, ENTITY_TYPE_COUNTY, entities, seen
        )
        self._match_table(
            normalized, AGRI_TERMS, ENTITY_TYPE_AGRI_TERM, entities, seen
        )

        return entities

    def _match_table(
        self,
        haystack: str,
        vocabulary: dict[str, list[str]],
        entity_type: str,
        entities: list[AgriculturalEntity],
        seen: set[tuple[str, str]],
    ) -> None:
        """
        Add an entity for each canonical label whose surface forms
        appear in `haystack`. Longer surface forms are tested first so
        a multi-word form ("dawa ya ukungu") wins over a substring
        ("dawa") when both would match.
        """
        for canonical, surface_forms in vocabulary.items():
            key = (entity_type, canonical.lower())
            if key in seen:
                continue

            for form in sorted(surface_forms, key=len, reverse=True):
                if self._contains(haystack, form.lower()):
                    entities.append(
                        AgriculturalEntity(
                            entity=form,
                            entity_type=entity_type,
                            normalized_value=canonical,
                            confidence=1.0,
                        )
                    )
                    seen.add(key)
                    break

    @staticmethod
    def _contains(haystack: str, needle: str) -> bool:
        """
        Whole-token/phrase containment. Guards against matching inside
        a larger word (so "corn" doesn't fire on "acorn", "meru"
        doesn't fire on "numerous"). Multi-word needles match as a
        substring bounded by non-word characters.
        """
        if not needle:
            return False
        idx = 0
        n = len(needle)
        while True:
            found = haystack.find(needle, idx)
            if found == -1:
                return False
            before = haystack[found - 1] if found > 0 else " "
            after_pos = found + n
            after = haystack[after_pos] if after_pos < len(haystack) else " "
            if not before.isalnum() and not after.isalnum():
                return True
            idx = found + 1