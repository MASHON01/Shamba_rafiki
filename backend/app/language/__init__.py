"""
Language intelligence layer.

Turns a raw farmer query - Swahili or English, however phrased -
into a structured understanding the rest of the system can act on:

    Voice/Text -> Language -> Normalized Query (+ intent + entities)

This layer sits in front of retrieval. Detecting the language,
normalizing the text, and pulling out the crop/county/disease a
farmer is asking about is what lets a Swahili question retrieve
against English source documents and lets the orchestrator pick the
right system prompt.

Components:
    detector.py           en / sw / unknown
    normalizer.py         whitespace, case, punctuation, orthography
    extractors/intent.py  diagnosis / price / how-to / general
    extractors/entities.py  crop, county, pest/disease
    terminology/          canonical crop + agri-term mapping
    translator.py         translation seam + dictionary fallback
    middleware.py         orders the stages
    analyzer.py           text -> LanguageAnalysis
"""

from __future__ import annotations

from app.language.analyzer import LanguageAnalyzer
from app.language.detector import LanguageDetector
from app.language.middleware import LanguageMiddleware, MiddlewareResult
from app.language.normalizer import InputNormalizer
from app.language.translator import Translator, TranslationNotAvailableError

__all__ = [
    "LanguageAnalyzer",
    "LanguageMiddleware",
    "MiddlewareResult",
    "LanguageDetector",
    "InputNormalizer",
    "Translator",
    "TranslationNotAvailableError",
]