"""
Semantic validator.

Measures how much of the answer's substance actually appears in the
retrieved context. Where the evidence scorer asks "did we have good
sources?", this asks "does the answer stay close to them, or does it
wander off into content the sources never mentioned?"

Dependency-free by design: it compares the set of meaningful content
words in the answer against those in the retrieved context (using the
same tokenizer the language layer uses, for consistency). A low
overlap means the answer introduced a lot of material with no basis
in the sources - a drift signal, not proof of error, which is why it
feeds confidence rather than hard-rejecting on its own.

Returns a `CheckResult` whose score is the supported fraction of the
answer's content words.
"""

from __future__ import annotations

from app.config.constants import (
    MIN_CONTENT_WORD_LENGTH,
    SEMANTIC_OVERLAP_MIN,
)
from app.language.normalizer import InputNormalizer
from app.models.document import RetrievalResult
from app.verification import CheckResult


class SemanticValidator:
    """
    Content-word overlap between an answer and its sources.
    """

    def __init__(self, normalizer: InputNormalizer | None = None) -> None:
        self._normalizer = normalizer or InputNormalizer()

    def validate(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> CheckResult:
        answer_words = self._content_words(answer)

        if not answer_words:
            # Nothing substantive to validate (e.g. a one-word reply).
            return CheckResult(
                name="semantic",
                score=1.0,
                passed=True,
                detail="Answer has no substantive content words to check.",
            )

        if not sources:
            return CheckResult(
                name="semantic",
                score=0.0,
                passed=False,
                detail="No context to validate the answer against.",
                flags=["no_context"],
            )

        context_words = set()
        for result in sources:
            context_words |= self._content_words(result.chunk.text)

        supported = answer_words & context_words
        overlap = len(supported) / len(answer_words)
        overlap = round(overlap, 3)

        passed = overlap >= SEMANTIC_OVERLAP_MIN
        flags = [] if passed else ["low_overlap"]

        return CheckResult(
            name="semantic",
            score=overlap,
            passed=passed,
            detail=(
                f"{len(supported)}/{len(answer_words)} answer content words "
                f"appear in the sources."
            ),
            flags=flags,
        )

    def _content_words(self, text: str) -> set[str]:
        """
        Meaningful tokens: normalizer tokens at/above the minimum
        length. Short function words are dropped without needing a
        full stopword list.
        """
        return {
            token
            for token in self._normalizer.tokenize(text)
            if len(token) >= MIN_CONTENT_WORD_LENGTH
        }