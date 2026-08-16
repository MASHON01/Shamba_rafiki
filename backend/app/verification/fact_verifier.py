"""
Fact verifier.

Runs the four individual checks (evidence, semantic, citation,
hallucination) over an answer + its sources and collects their
`CheckResult`s into a list of `CheckSignal`s. It coordinates *what
was checked*; the confidence engine decides *how much to trust it*
and the policy decides *what to do about it* - three separate
concerns, three separate files, so each is independently tunable.

Kept deliberately thin: it owns no thresholds and makes no
judgement, it just fans out to the checks and gathers their signals
(plus the union of any flags they raised). That makes the set of
checks easy to extend - add a checker, add one line here.
"""

from __future__ import annotations

from app.models.document import RetrievalResult
from app.models.verification import CheckSignal
from app.verification import CheckResult
from app.verification.citation_checker import CitationChecker
from app.verification.evidence_scorer import EvidenceScorer
from app.verification.hallucination_detector import HallucinationDetector
from app.verification.semantic_validator import SemanticValidator


class FactVerifier:
    """
    Runs all verification checks and collects their signals.
    """

    def __init__(
        self,
        evidence_scorer: EvidenceScorer | None = None,
        semantic_validator: SemanticValidator | None = None,
        citation_checker: CitationChecker | None = None,
        hallucination_detector: HallucinationDetector | None = None,
    ) -> None:
        self._evidence = evidence_scorer or EvidenceScorer()
        self._semantic = semantic_validator or SemanticValidator()
        self._citation = citation_checker or CitationChecker()
        self._hallucination = (
            hallucination_detector or HallucinationDetector()
        )

    def verify(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> list[CheckSignal]:
        """
        Run every check and return their signals. Order is stable
        (evidence, semantic, citation, hallucination) so downstream
        weighting and logging are deterministic.
        """
        results: list[CheckResult] = [
            self._evidence.score(sources),
            self._semantic.validate(answer, sources),
            self._citation.check(answer, sources),
            self._hallucination.detect(answer, sources),
        ]
        return [self._to_signal(r) for r in results]

    @staticmethod
    def _to_signal(result: CheckResult) -> CheckSignal:
        return CheckSignal(
            name=result.name,
            score=result.score,
            passed=result.passed,
            detail=result.detail,
            flags=result.flags,
        )