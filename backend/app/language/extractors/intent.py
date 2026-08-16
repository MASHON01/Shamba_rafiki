"""
Intent extractor.

Classifies what the farmer is trying to do, so the orchestrator can
pick the right system prompt and response shape:

    diagnosis   something is wrong with a crop (spots, pests, wilting)
    price       market / cost / whether it's worth selling
    how_to      how/when to plant, apply, grow, treat
    general     anything else (fallback)

Keyword-driven and bilingual (INTENT_KEYWORDS carries Swahili and
English triggers). This is intentionally a coarse classifier: the
LLM does the real reasoning downstream. Its only job is routing, and
a wrong guess degrades to a slightly less tailored prompt, never a
failure - so simple keyword matching is the right weight here.

Returns a plain intent string (one of the INTENT_KEYWORDS keys or
DEFAULT_INTENT). Kept as a string rather than an enum because intents
will grow as the product does, and callers only ever compare against
the constants.
"""

from __future__ import annotations

from app.config.constants import (
    DEFAULT_INTENT,
    INTENT_KEYWORDS,
)
from app.language.normalizer import InputNormalizer


class IntentExtractor:
    """
    Coarse bilingual intent classifier.
    """

    def __init__(self, normalizer: InputNormalizer | None = None) -> None:
        self._normalizer = normalizer or InputNormalizer()

    def extract(self, text: str) -> str:
        """
        Return the first intent whose keywords appear in the query,
        in INTENT_KEYWORDS order (most specific first), else
        DEFAULT_INTENT.
        """
        if not text or not text.strip():
            return DEFAULT_INTENT

        normalized = self._normalizer.normalize(text)

        for intent, keywords in INTENT_KEYWORDS.items():
            if any(self._contains(normalized, kw) for kw in keywords):
                return intent

        return DEFAULT_INTENT

    @staticmethod
    def _contains(haystack: str, needle: str) -> bool:
        """
        Substring match with word boundaries, so multi-word triggers
        ("how much", "what is wrong") work and single words don't fire
        inside larger words.
        """
        needle = needle.lower()
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