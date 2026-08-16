"""
Grounding check: did the answer actually use the retrieved sources?

A confident answer that ignores the reference material is exactly the
failure mode this whole system is built to avoid. This check measures, on
two axes, how much an answer is anchored to the sources it was given:

  - citation: does it reference the numbered [Source N] tags at all?
  - overlap: how much of the answer's own vocabulary appears in the
    source texts (a cheap, model-free proxy for "is it talking about
    what the sources say")?

Kept dependency-light on purpose so the grader can call it without a
model. When there are no sources (retrieval found nothing), grounding is
"not applicable" - a correct answer there is an honest "I'm not sure",
which the hallucination probes score instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config.constants import MIN_CONTENT_WORD_LENGTH
from app.models.document import RetrievalResult

_CITATION_RE = re.compile(r"\[source\s*\d+", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-zA-ZÀ-ſ]+")


@dataclass(slots=True)
class GroundingResult:
    """How well an answer is anchored to its sources."""

    applicable: bool  # were there any sources to ground in?
    cited: bool  # does the answer reference [Source N]?
    overlap: float  # share of answer words found in the sources (0-1)
    score: float  # blended grounding score (0-1)
    used: bool  # did the answer meaningfully use the sources?


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= MIN_CONTENT_WORD_LENGTH}


def grounding_score(
    answer: str,
    sources: list[RetrievalResult],
    overlap_threshold: float = 0.15,
) -> GroundingResult:
    """
    Score how much ``answer`` is grounded in ``sources``.

    ``score`` blends citation presence and vocabulary overlap. ``used``
    is True if the answer either cites a source or shares a meaningful
    amount of vocabulary with them.
    """
    if not sources:
        return GroundingResult(applicable=False, cited=False, overlap=0.0, score=0.0, used=False)

    cited = bool(_CITATION_RE.search(answer))

    answer_words = _content_words(answer)
    source_words: set[str] = set
    for result in sources:
        source_words |= _content_words(result.chunk.text)

    overlap = len(answer_words & source_words) / len(answer_words) if answer_words else 0.0

    # Citation and overlap each contribute half; citation is a strong
    # signal of intent to ground, overlap of actual grounding.
    score = 0.5 * float(cited) + 0.5 * min(1.0, overlap / max(overlap_threshold, 1e-9))
    score = min(1.0, score)
    used = cited or overlap >= overlap_threshold

    return GroundingResult(
        applicable=True,
        cited=cited,
        overlap=overlap,
        score=score,
        used=used,
    )
