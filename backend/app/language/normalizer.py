"""
Input normalizer.

Cleans a raw farmer query into a consistent form before detection,
entity extraction, and embedding. Kiosk input is messy - stray caps,
double spaces, trailing punctuation, smart quotes pasted from a
phone - and every downstream stage benefits from seeing the same
normalized text.

Two distinct outputs, deliberately:

    normalize(text)        -> display/embedding form: lowercased,
                              whitespace-collapsed, punctuation tidied,
                              but words intact. This is what goes to
                              the embedder and is stored as
                              LanguageAnalysis.normalized_query.

    tokenize(text)         -> a list of bare word tokens (punctuation
                              stripped) for the detector and keyword
                              matchers to count over.

Keeping normalization non-destructive (it never drops words) matters:
the embedder needs the full query, and over-cleaning silently
degrades retrieval.
"""

from __future__ import annotations

import re
import unicodedata

# Smart quotes / dashes a phone keyboard inserts, mapped to ASCII.
_PUNCT_TRANSLATION = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
}
_PUNCT_TABLE = {ord(k): v for k, v in _PUNCT_TRANSLATION.items()}

_WHITESPACE_RE = re.compile(r"\s+")
# Repeated terminal punctuation ("???", "!!!") collapsed to one.
_REPEAT_PUNCT_RE = re.compile(r"([?!.,])\1{1,}")
# Token = run of letters/digits/apostrophe/hyphen (keeps "uasin-gishu").
_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


class InputNormalizer:
    """
    Normalizes and tokenizes farmer query text.
    """

    def normalize(self, text: str) -> str:
        """
        Return a cleaned, lowercased, whitespace-collapsed form with
        words preserved. Empty/whitespace input returns "".
        """
        if not text or not text.strip():
            return ""

        # Canonical unicode form, then map smart punctuation to ASCII.
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(_PUNCT_TABLE)

        text = text.lower()
        text = _REPEAT_PUNCT_RE.sub(r"\1", text)
        text = _WHITESPACE_RE.sub(" ", text)

        return text.strip()

    def tokenize(self, text: str) -> list[str]:
        """
        Return bare word tokens from `text`, punctuation stripped and
        lowercased. Runs its own normalization so callers can tokenize
        raw or pre-normalized text interchangeably.
        """
        if not text or not text.strip():
            return []
        normalized = self.normalize(text)
        return _TOKEN_RE.findall(normalized)