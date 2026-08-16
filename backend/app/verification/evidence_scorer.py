"""
Evidence scorer.

Scores how strongly the retrieved sources back an answer, independent
of the answer's wording. It blends two signals:

- strength: how similar the best-matching sources were to the query
  (the retrieval similarity scores already computed in Output 4), and
- coverage: how many sources corroborate, not just one lucky hit.

A high evidence score means "we had good material to answer from";
a low score means the answer rests on thin or no retrieval, which
should pull confidence down regardless of how fluent the answer
reads. This is the check that most directly reflects "is this
grounded at all".

Returns a `CheckResult` with a 0-1 score.
"""

from __future__ import annotations

from app.config.constants import (
    EVIDENCE_COVERAGE_WEIGHT,
    EVIDENCE_SIMILARITY_WEIGHT,
)
from app.models.document import RetrievalResult
from app.verification import CheckResult

# Number of sources at which coverage is considered "full". Beyond a
# few corroborating chunks, extra sources add little confidence.
_COVERAGE_SATURATION = 3


class EvidenceScorer:
    """
    Scores retrieval-evidence strength behind an answer.
    """

    def score(self, sources: list[RetrievalResult]) -> CheckResult:
        if not sources:
            return CheckResult(
                name="evidence",
                score=0.0,
                passed=False,
                detail="No supporting sources were retrieved.",
                flags=["no_evidence"],
            )

        # Strength: the best source's similarity (clamped to [0, 1]).
        top_similarity = max(0.0, min(1.0, sources[0].similarity_score))

        # Coverage: how many sources corroborate, saturating at a few.
        coverage = min(len(sources), _COVERAGE_SATURATION) / _COVERAGE_SATURATION

        score = (
            EVIDENCE_SIMILARITY_WEIGHT * top_similarity
            + EVIDENCE_COVERAGE_WEIGHT * coverage
        )
        score = round(max(0.0, min(1.0, score)), 3)

        return CheckResult(
            name="evidence",
            score=score,
            passed=score >= 0.5,
            detail=(
                f"Top similarity {top_similarity:.2f} across "
                f"{len(sources)} source(s)."
            ),
        )