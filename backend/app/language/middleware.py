"""
Language middleware.

The internal coordinator that runs the language stages in the right
order and resolves the interactions between them. Separated from
`analyzer.py` so the *ordering and decision logic* is testable on its
own, independent of the public `LanguageAnalysis` shape the analyzer
returns.

Pipeline it runs:

    raw text
      -> normalize                       (InputNormalizer)
      -> detect language                 (LanguageDetector)
      -> resolve language vs. hint       (this module)
      -> if Swahili: gloss to English    (Translator fallback)
      -> extract entities                (EntityExtractor)
      -> extract intent                  (IntentExtractor)

Two decisions live here, deliberately, rather than in any single
stage:

1. Language resolution. The detector can return `unknown` (a bare
   content word, a tie). When it does, an explicit request-level
   language hint - the farmer's Swahili/English toggle on the kiosk -
   is trusted instead. A confident detection overrides a missing or
   absent hint. This is exactly the kind of cross-stage judgment that
   shouldn't be baked into the detector itself.

2. Whether to translate. Entities are extracted bilingually anyway,
   but glossing a Swahili query's domain words to English first gives
   retrieval its English anchors. So translation runs for Swahili (or
   unknown-but-hinted-Swahili) queries only.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.language.detector import LanguageDetector
from app.language.extractors.entities import EntityExtractor
from app.language.extractors.intent import IntentExtractor
from app.language.normalizer import InputNormalizer
from app.language.translator import Translator
from app.models.language import (
    AgriculturalEntity,
    LanguageCode,
    LanguageDetectionResult,
)
from app.utils.logger import get_logger

logger = get_logger("LanguageMiddleware")


@dataclass(slots=True)
class MiddlewareResult:
    """
    Everything the middleware produces, before the analyzer packages
    it into the public LanguageAnalysis model.
    """

    detection: LanguageDetectionResult
    normalized_query: str
    retrieval_query: str  # normalized, and glossed to English if Swahili
    entities: list[AgriculturalEntity]
    intent: str
    translated: bool


class LanguageMiddleware:
    """
    Orders the language stages and resolves cross-stage decisions.
    """

    def __init__(
        self,
        normalizer: InputNormalizer | None = None,
        detector: LanguageDetector | None = None,
        entity_extractor: EntityExtractor | None = None,
        intent_extractor: IntentExtractor | None = None,
        translator: Translator | None = None,
    ) -> None:
        self._normalizer = normalizer or InputNormalizer()
        self._detector = detector or LanguageDetector(self._normalizer)
        self._entities = entity_extractor or EntityExtractor(self._normalizer)
        self._intent = intent_extractor or IntentExtractor(self._normalizer)
        self._translator = translator or Translator()

    def process(
        self,
        text: str,
        language_hint: str | None = None,
    ) -> MiddlewareResult:
        normalized = self._normalizer.normalize(text)

        detection = self._detector.detect(text)
        effective_language = self._resolve_language(detection, language_hint)

        # Gloss Swahili domain words to English for retrieval anchoring.
        translated = False
        retrieval_query = normalized
        if effective_language == LanguageCode.SWAHILI and normalized:
            result = self._translator.translate(
                normalized,
                source=LanguageCode.SWAHILI,
                target=LanguageCode.ENGLISH,
            )
            retrieval_query = result.translated_text
            translated = retrieval_query.lower() != normalized.lower()

        # Entities/intent are extracted from the original normalized
        # text (bilingual matchers handle Swahili directly), so a crop
        # is caught whether or not glossing changed anything.
        entities = self._entities.extract(normalized)
        intent = self._intent.extract(normalized)

        logger.info(
            "language.processed",
            detected=detection.language.value,
            effective=effective_language.value,
            hint=language_hint,
            entities=len(entities),
            intent=intent,
            translated=translated,
        )

        return MiddlewareResult(
            detection=detection,
            normalized_query=normalized,
            retrieval_query=retrieval_query,
            entities=entities,
            intent=intent,
            translated=translated,
        )

    def _resolve_language(
        self,
        detection: LanguageDetectionResult,
        language_hint: str | None,
    ) -> LanguageCode:
        """
        Reconcile the detector's result with an explicit request hint.

        - Confident detection (not unknown) is trusted.
        - On `unknown`, fall back to the hint if it's a valid code.
        - Failing both, default to English (the primary evaluation
          language per the product notes).
        """
        if detection.language != LanguageCode.UNKNOWN:
            return detection.language

        if language_hint:
            try:
                return LanguageCode(language_hint.strip().lower())
            except ValueError:
                logger.warning(
                    "language.invalid_hint", hint=language_hint
                )

        return LanguageCode.ENGLISH