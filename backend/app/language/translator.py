"""
Translator interface + offline dictionary fallback.

Two layers, matching the seam pattern used for OCR but with a working
default:

1. A `TranslatorEngine` seam - the contract a real translation engine
   (a full MT model, an API, a fine-tuned seq2seq) would implement
   later. Until one is wired in, requesting engine translation raises
   `TranslationNotAvailableError`.

2. A `DictionaryFallbackTranslator` that works offline right now. It
   does NOT do fluent sentence translation - it glosses the
   agriculturally-meaningful Swahili words in a query into their
   canonical English terms using AgriTerminology and CropNormalizer.

Why the fallback is enough for Phase 1: the point of translation here
isn't to produce readable English prose, it's to make a Swahili query
retrieve against an English corpus. Replacing "nyanya" with "tomato"
and "ukungu" with "blight" gives the embedder and keyword retrieval
the English anchors they need, which is where almost all of the
retrieval benefit comes from. Full fluent MT is a later, optional
upgrade - and when it arrives it slots in behind the same interface
without changing any caller.

Both paths return the existing `TranslationResult` model.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.core.exceptions import ShambaRafikiError
from app.language.normalizer import InputNormalizer
from app.language.terminology import AgriTerminology, CropNormalizer
from app.models.language import LanguageCode, TranslationResult
from app.utils.logger import get_logger

logger = get_logger("Translator")


class TranslationNotAvailableError(ShambaRafikiError):
    """Raised when engine translation is requested but no engine exists."""


@runtime_checkable
class TranslatorEngine(Protocol):
    """
    Contract a full translation backend must satisfy to plug in.
    A backend is any object exposing `translate(text, source, target)`.
    """

    def translate(
        self, text: str, source: LanguageCode, target: LanguageCode
    ) -> str: ...


class Translator:
    """
    Translation entry point.

    With no engine, `translate()` uses the offline dictionary
    fallback (Swahili domain-word glossing). With an engine wired in,
    it delegates to the engine. Either way callers get a
    `TranslationResult` and never need to know which path ran.
    """

    def __init__(
        self,
        engine: TranslatorEngine | None = None,
        fallback: "DictionaryFallbackTranslator | None" = None,
    ) -> None:
        self._engine = engine
        self._fallback = fallback or DictionaryFallbackTranslator()

    @property
    def has_engine(self) -> bool:
        return self._engine is not None

    def translate(
        self,
        text: str,
        source: LanguageCode = LanguageCode.SWAHILI,
        target: LanguageCode = LanguageCode.ENGLISH,
    ) -> TranslationResult:
        """
        Translate `text` from `source` to `target`. Uses the engine
        if one is wired in, otherwise the dictionary fallback.
        """
        if self._engine is not None:
            translated = self._engine.translate(text, source, target)
            return TranslationResult(
                source_language=source,
                target_language=target,
                original_text=text,
                translated_text=translated,
            )
        return self._fallback.translate(text, source, target)

    def translate_with_engine(
        self,
        text: str,
        source: LanguageCode = LanguageCode.SWAHILI,
        target: LanguageCode = LanguageCode.ENGLISH,
    ) -> TranslationResult:
        """
        Force engine translation, raising if no engine is available.
        For callers that specifically need fluent MT and would rather
        fail than silently get a word-level gloss.
        """
        if self._engine is None:
            raise TranslationNotAvailableError(
                "Full translation was requested but no translation engine "
                "is wired in. A dictionary fallback is available via "
                "translate(); fluent MT is a later, optional capability."
            )
        translated = self._engine.translate(text, source, target)
        return TranslationResult(
            source_language=source,
            target_language=target,
            original_text=text,
            translated_text=translated,
        )


class DictionaryFallbackTranslator:
    """
    Offline word-level Swahili -> English glossing for domain terms.

    Not a general translator: it replaces known crop and
    agri-terminology tokens with their canonical English forms and
    leaves everything else untouched. The result is a query with
    English anchors for retrieval, not fluent prose.
    """

    def __init__(
        self,
        normalizer: InputNormalizer | None = None,
        crops: CropNormalizer | None = None,
        terms: AgriTerminology | None = None,
    ) -> None:
        self._normalizer = normalizer or InputNormalizer()
        self._crops = crops or CropNormalizer()
        self._terms = terms or AgriTerminology()

    def translate(
        self,
        text: str,
        source: LanguageCode = LanguageCode.SWAHILI,
        target: LanguageCode = LanguageCode.ENGLISH,
    ) -> TranslationResult:
        tokens = self._normalizer.tokenize(text)

        glossed: list[str] = []
        replaced = 0
        for token in tokens:
            canonical = self._crops.normalize(token) or self._terms.gloss_token(
                token
            )
            if canonical is not None and canonical.lower() != token.lower():
                glossed.append(canonical)
                replaced += 1
            else:
                glossed.append(token)

        translated_text = " ".join(glossed)

        logger.debug(
            "translator.fallback.glossed",
            tokens=len(tokens),
            replaced=replaced,
        )

        return TranslationResult(
            source_language=source,
            target_language=target,
            original_text=text,
            translated_text=translated_text,
        )