"""
Language analyzer - the public entry point for Output 5.

The single object the orchestrator (Output 6) calls to turn a raw
farmer query into structured understanding. It runs the middleware
and packages the result into the standard `LanguageAnalysis` model:

    analyze(text, language_hint) -> LanguageAnalysis

`LanguageAnalysis` carries detected language, the normalized query,
the extracted entities, and whether translation happened. Intent and
the retrieval-ready (English-glossed) query aren't fields on that
model, so `analyze_full()` is also provided for callers that need
them - the orchestrator will want both to filter retrieval and pick a
prompt. `analyze()` remains the clean, standard-contract path.

This mirrors how `Retriever` fronts Output 4: one obvious public
method, all the moving parts hidden behind it.
"""

from __future__ import annotations

from app.language.middleware import LanguageMiddleware, MiddlewareResult
from app.models.language import LanguageAnalysis


class LanguageAnalyzer:
    """
    Public façade over the language intelligence layer.
    """

    def __init__(self, middleware: LanguageMiddleware | None = None) -> None:
        self._middleware = middleware or LanguageMiddleware()

    def analyze(
        self,
        text: str,
        language_hint: str | None = None,
    ) -> LanguageAnalysis:
        """
        Analyze `text` and return the standard LanguageAnalysis model.
        `language_hint` is the kiosk's Swahili/English toggle, trusted
        only when detection is inconclusive.
        """
        result = self._middleware.process(text, language_hint=language_hint)
        return LanguageAnalysis(
            detected_language=result.detection,
            normalized_query=result.normalized_query,
            entities=result.entities,
            translated=result.translated,
        )

    def analyze_full(
        self,
        text: str,
        language_hint: str | None = None,
    ) -> MiddlewareResult:
        """
        Analyze `text` and return the full middleware result, which
        additionally exposes `intent` and `retrieval_query` (the
        English-anchored query to send to the retriever). Preferred by
        the orchestrator; `analyze()` is the standard-model path.
        """
        return self._middleware.process(text, language_hint=language_hint)