"""
Module dispatcher.

Executes the stages a RoutePlan selects, in order, and returns the raw
pieces the orchestrator assembles into a response:

    analyze language -> retrieve context -> build prompt -> generate

The retriever is optional at construction, so the dispatcher is
testable before an index exists and the app can run in a degraded
no-corpus mode rather than failing.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.language.analyzer import LanguageAnalyzer
from app.language.middleware import MiddlewareResult
from app.models.document import RetrievalResult
from app.models.language import LanguageCode
from app.orchestration.llm.base import BaseLLMClient, GenerationResult
from app.orchestration.llm.generation_config import GenerationConfig
from app.orchestration.llm.presets import for_intent
from app.orchestration.prompts.builder import BuiltPrompt, ConversationTurn, PromptBuilder
from app.orchestration.prompts.context_builder import BuiltContext, ContextBuilder
from app.orchestration.router import RoutePlan
from app.utils.logger import get_logger

logger = get_logger("Dispatcher")


@dataclass(slots=True)
class PreparedRequest:
    """Everything assembled before generation."""

    analysis: MiddlewareResult
    context: BuiltContext
    prompt: BuiltPrompt
    config: GenerationConfig
    language: LanguageCode


@dataclass(slots=True)
class DispatchResult:
    """Everything produced running a request through the pipeline stages."""

    analysis: MiddlewareResult
    context: BuiltContext
    prompt: BuiltPrompt
    generation: GenerationResult
    sources: list[RetrievalResult]


class Dispatcher:
    """Runs the selected pipeline stages for a request."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        analyzer: LanguageAnalyzer | None = None,
        retriever=None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._analyzer = analyzer or LanguageAnalyzer()
        self._retriever = retriever
        self._context_builder = context_builder or ContextBuilder()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._llm = llm_client

    def prepare(
        self,
        query: str,
        plan: RoutePlan,
        language_hint: str | None = None,
        history: list[ConversationTurn] | None = None,
        top_k: int | None = None,
        generation_config: GenerationConfig | None = None,
        retrieval_hint: str | None = None,
        question_note: str | None = None,
    ) -> PreparedRequest:
        """
        Run every stage up to generation and return a PreparedRequest.

        `retrieval_hint` and `question_note` are the image hooks: when a
        photo has been classified, the crop+disease hint is prepended to
        the retrieval query and a short note is appended to the prompt
        question. Both default to None, leaving the text path unchanged.
        """
        analysis = self._analyzer.analyze_full(query, language_hint=language_hint)

        retrieval_query = analysis.retrieval_query
        if retrieval_hint:
            retrieval_query = f"{retrieval_hint} {retrieval_query}".strip()

        sources: list[RetrievalResult] = []
        if plan.use_retrieval and self._retriever is not None:
            sources = self._retriever.retrieve(retrieval_query, top_k=top_k)

        context = self._context_builder.build(sources)

        language = self._effective_language(analysis)
        question = f"{query}\n\n{question_note}" if question_note else query
        prompt = self._prompt_builder.build(
            question=question,
            context=context,
            language=language,
            intent=analysis.intent,
            history=history,
        )

        config = generation_config or for_intent(analysis.intent)

        return PreparedRequest(
            analysis=analysis,
            context=context,
            prompt=prompt,
            config=config,
            language=language,
        )

    def dispatch(
        self,
        query: str,
        plan: RoutePlan,
        language_hint: str | None = None,
        history: list[ConversationTurn] | None = None,
        top_k: int | None = None,
        generation_config: GenerationConfig | None = None,
        retrieval_hint: str | None = None,
        question_note: str | None = None,
    ) -> DispatchResult:
        prepared = self.prepare(
            query=query,
            plan=plan,
            language_hint=language_hint,
            history=history,
            top_k=top_k,
            generation_config=generation_config,
            retrieval_hint=retrieval_hint,
            question_note=question_note,
        )

        generation = self._llm.generate(prepared.prompt)

        logger.info(
            "dispatch.completed",
            route=plan.route.value,
            language=prepared.language.value,
            intent=prepared.analysis.intent,
            sources=len(prepared.context.sources),
            has_context=prepared.context.has_context,
        )

        return DispatchResult(
            analysis=prepared.analysis,
            context=prepared.context,
            prompt=prepared.prompt,
            generation=generation,
            sources=prepared.context.sources,
        )

    @staticmethod
    def _effective_language(analysis: MiddlewareResult) -> LanguageCode:
        lang = analysis.detection.language
        return lang if lang != LanguageCode.UNKNOWN else LanguageCode.ENGLISH
