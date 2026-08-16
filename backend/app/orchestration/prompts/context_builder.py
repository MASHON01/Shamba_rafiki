"""
Context builder.

Turns the retriever's `RetrievalResult`s into the numbered reference
block that goes into the prompt. Its jobs, in order:

1. Drop weak matches. Chunks below MIN_CONTEXT_SIMILARITY are more
   likely to mislead the model than help it, so they're filtered even
   though retrieval returned them.
2. Tag each source. Every chunk is labelled with a number and its
   crop/county metadata, so the model can ground answers in specific
   sources and (in Output 7) citations can be checked against them.
3. Stay within budget. The block is truncated to MAX_CONTEXT_TOKENS
   using a pessimistic char/token estimate, so it never crowds out
   the system prompt, history, question, or the model's answer inside
   the context window.

Separated from the prompt builder because "how a source is formatted"
evolves independently of "how the whole prompt is assembled" - and
because the numbered-source format is what the verifier will later
parse.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config.constants import (
    CHARS_PER_TOKEN_ESTIMATE,
    MAX_CONTEXT_TOKENS,
    MIN_CONTEXT_SIMILARITY,
    NO_CONTEXT_PLACEHOLDER,
)
from app.models.document import RetrievalResult
from app.utils.logger import get_logger

logger = get_logger("ContextBuilder")


@dataclass(slots=True)
class BuiltContext:
    """
    The assembled context block plus the sources that actually made it
    in (after filtering and truncation). The orchestrator carries
    `sources` through to the response so the answer can show what it
    was grounded on.
    """

    text: str
    sources: list[RetrievalResult]
    has_context: bool


class ContextBuilder:
    """
    Formats retrieved chunks into a token-budgeted context block.
    """

    def __init__(
        self,
        max_tokens: int = MAX_CONTEXT_TOKENS,
        min_similarity: float = MIN_CONTEXT_SIMILARITY,
    ) -> None:
        self._max_tokens = max_tokens
        self._min_similarity = min_similarity

    def build(self, results: list[RetrievalResult]) -> BuiltContext:
        usable = [
            r for r in results if r.similarity_score >= self._min_similarity
        ]

        if not usable:
            logger.debug(
                "context.none_usable",
                retrieved=len(results),
                min_similarity=self._min_similarity,
            )
            return BuiltContext(
                text=NO_CONTEXT_PLACEHOLDER, sources=[], has_context=False
            )

        char_budget = self._max_tokens * CHARS_PER_TOKEN_ESTIMATE
        blocks: list[str] = []
        included: list[RetrievalResult] = []
        used_chars = 0

        for i, result in enumerate(usable, start=1):
            block = self._format_source(i, result)
            # Always include at least the first source, even if a single
            # long chunk exceeds the budget (better a truncated answer
            # than none); stop adding once the budget is spent.
            if used_chars + len(block) > char_budget and included:
                break
            blocks.append(block)
            included.append(result)
            used_chars += len(block)

        logger.debug(
            "context.built",
            retrieved=len(results),
            usable=len(usable),
            included=len(included),
            approx_tokens=used_chars // CHARS_PER_TOKEN_ESTIMATE,
        )

        return BuiltContext(
            text="\n\n".join(blocks),
            sources=included,
            has_context=True,
        )

    def _format_source(self, index: int, result: RetrievalResult) -> str:
        """
        One numbered source with its crop/county tag and text. The
        "[Source N]" shape is what the citation checker will look for.
        """
        meta = result.chunk.metadata
        tags = []
        crop = meta.get("crop")
        county = meta.get("county")
        if crop and crop != "unknown":
            tags.append(crop)
        if county and county != "unknown":
            tags.append(county)
        tag_str = f" ({', '.join(tags)})" if tags else ""

        return f"[Source {index}{tag_str}]\n{result.chunk.text.strip()}"