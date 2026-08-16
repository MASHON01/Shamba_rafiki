"""
Language detector.

Decides whether a query is English or Swahili (or neither, clearly)
using marker-word overlap - no model, no weights, no network. For
short farmer queries this is both fast and reliable: English and
Swahili function words (the/a/is vs na/ya/wa) rarely overlap, so
counting how many of each appear is a strong signal.

Returns the existing `LanguageDetectionResult` (language +
confidence). Confidence reflects how lopsided the marker counts are,
so the middleware/analyzer can treat a low-confidence guess
differently (e.g. honor an explicit request-level language hint)
without this module needing to know about hints.

A model-based detector could later replace this behind the same
`detect()` signature; nothing downstream would change.
"""

from __future__ import annotations

from app.config.constants import ENGLISH_MARKERS, SWAHILI_MARKERS
from app.language.normalizer import InputNormalizer
from app.models.language import LanguageCode, LanguageDetectionResult

# Below this many total marker hits, we don't have enough signal to
# be confident - report the leaning language but at low confidence.
_MIN_MARKERS_FOR_CONFIDENCE = 2


class LanguageDetector:
    """
    Marker-overlap language detector (en / sw / unknown).
    """

    def __init__(self, normalizer: InputNormalizer | None = None) -> None:
        self._normalizer = normalizer or InputNormalizer()

    def detect(self, text: str) -> LanguageDetectionResult:
        tokens = set(self._normalizer.tokenize(text))

        if not tokens:
            return LanguageDetectionResult(
                language=LanguageCode.UNKNOWN, confidence=0.0
            )

        sw_hits = len(tokens & SWAHILI_MARKERS)
        en_hits = len(tokens & ENGLISH_MARKERS)
        total = sw_hits + en_hits

        # No markers at all: can't tell. A bare "nyanya" or "maize"
        # (content word, no function words) lands here - genuinely
        # ambiguous without more context.
        if total == 0:
            return LanguageDetectionResult(
                language=LanguageCode.UNKNOWN, confidence=0.0
            )

        if sw_hits > en_hits:
            language = LanguageCode.SWAHILI
            dominant = sw_hits
        elif en_hits > sw_hits:
            language = LanguageCode.ENGLISH
            dominant = en_hits
        else:
            # Tie (e.g. code-switched query): unknown, low confidence.
            return LanguageDetectionResult(
                language=LanguageCode.UNKNOWN,
                confidence=round(0.5, 3),
            )

        # Confidence: share of markers that point at the winner,
        # damped when there's very little signal to go on.
        share = dominant / total
        if total < _MIN_MARKERS_FOR_CONFIDENCE:
            share *= 0.6

        return LanguageDetectionResult(
            language=language,
            confidence=round(min(share, 1.0), 3),
        )