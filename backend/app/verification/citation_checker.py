"""
Citation checker.

Checks that the answer's claims are traceable to the retrieved
sources. The context builder (Output 6) formatted sources as numbered
"[Source N]" blocks specifically so grounding could be checked here.

Two cases it handles:

1. Explicit citations. If the answer references "[Source N]" or
   "Source N", the checker verifies that source number actually
   exists in what was retrieved - catching a model that cites a
   source that isn't there.

2. Implicit grounding. Models usually don't cite; they just answer.
   So the checker also works sentence-by-sentence, asking whether each
   substantive sentence's content words appear in some source. A
   sentence grounded in no source is flagged as an uncited claim.

The score is the fraction of substantive sentences that are grounded.
Like the other checks it feeds confidence rather than hard-rejecting,
because a fluent restatement can be correct while using different
words - the policy layer decides what to do with a low score.

Returns a `CheckResult`.
"""

from __future__ import annotations

import re

from app.config.constants import MIN_CONTENT_WORD_LENGTH
from app.language.normalizer import InputNormalizer
from app.models.document import RetrievalResult
from app.verification import CheckResult

# Sentence splitter (shared shape with the chunker's approach).
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Explicit citation references the model might emit.
_CITATION_REF_RE = re.compile(r"\[?\s*source\s*(\d+)\s*\]?", re.IGNORECASE)
# Fraction of a sentence's content words that must appear across the
# retrieved sources for that sentence to count as grounded. Checked
# against the union of all sources, so a correct paraphrase that draws
# on more than one retrieved chunk is not wrongly marked uncited.
_SENTENCE_GROUNDING_RATIO = 0.4


class CitationChecker:
    """
    Verifies answer claims trace back to retrieved sources.
    """

    def __init__(self, normalizer: InputNormalizer | None = None) -> None:
        self._normalizer = normalizer or InputNormalizer()

    def check(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> CheckResult:
        flags: list[str] = []

        # 1. Any explicit [Source N] references must be in range.
        explicit_refs = [
            int(n) for n in _CITATION_REF_RE.findall(answer)
        ]
        for ref in explicit_refs:
            if ref < 1 or ref > len(sources):
                flags.append(f"invalid_citation_source_{ref}")

        # 2. Implicit grounding, sentence by sentence.
        sentences = [
            s.strip() for s in _SENTENCE_RE.split(answer) if s.strip()
        ]
        substantive = [
            s for s in sentences if len(self._content_words(s)) >= 2
        ]

        if not substantive:
            return CheckResult(
                name="citation",
                score=1.0,
                passed=not flags,
                detail="No substantive claims requiring citation.",
                flags=flags,
            )

        if not sources:
            return CheckResult(
                name="citation",
                score=0.0,
                passed=False,
                detail="Answer makes claims but no sources were retrieved.",
                flags=flags + ["uncited_claims"],
            )

        source_words: set[str] = set()
        for r in sources:
            source_words |= self._content_words(r.chunk.text)

        grounded = 0
        for sentence in substantive:
            if self._is_grounded(sentence, source_words):
                grounded += 1

        ungrounded = len(substantive) - grounded
        if ungrounded:
            flags.append("uncited_claims")

        score = round(grounded / len(substantive), 3)

        return CheckResult(
            name="citation",
            score=score,
            passed=score >= 0.5 and not any(
                f.startswith("invalid_citation") for f in flags
            ),
            detail=(
                f"{grounded}/{len(substantive)} substantive sentences "
                f"grounded in a source."
            ),
            flags=flags,
        )

    def _is_grounded(
        self,
        sentence: str,
        source_words: set[str],
    ) -> bool:
        words = self._content_words(sentence)
        if not words:
            return True
        overlap = len(words & source_words) / len(words)
        return overlap >= _SENTENCE_GROUNDING_RATIO

    def _content_words(self, text: str) -> set[str]:
        return {
            token
            for token in self._normalizer.tokenize(text)
            if len(token) >= MIN_CONTENT_WORD_LENGTH
        }