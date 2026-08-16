"""
Verifier - the public entry point for Output 7.

The single object the orchestrator calls after generation:

    verify(answer, sources, language) -> VerifiedAnswer

It wires the three stages - fact verifier (run the checks),
confidence engine (score them), decision policy (act on them) - into
one call, and honours the ENABLE_VERIFICATION setting so verification
can be switched off wholesale (e.g. for A/B measuring its impact on
the accuracy score) without the orchestrator changing.

With verification disabled, it returns the answer untouched wrapped
in an APPROVED VerifiedAnswer at neutral confidence, so the
orchestrator's downstream handling is identical either way.
"""

from __future__ import annotations

from app.config.settings import settings
from app.models.classifier import ConfidenceLevel
from app.models.document import RetrievalResult
from app.models.language import LanguageCode
from app.models.verification import (
    VerificationAction,
    VerificationReport,
    VerifiedAnswer,
)
from app.utils.logger import get_logger
from app.verification.confidence_engine import ConfidenceEngine
from app.verification.fact_verifier import FactVerifier
from app.verification.policies import DecisionPolicy

logger = get_logger("Verifier")


class Verifier:
    """
    Coordinates fact-checking, confidence scoring, and the decision
    policy into a single verify() call.
    """

    def __init__(
        self,
        fact_verifier: FactVerifier | None = None,
        confidence_engine: ConfidenceEngine | None = None,
        policy: DecisionPolicy | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._fact_verifier = fact_verifier or FactVerifier()
        self._confidence = confidence_engine or ConfidenceEngine()
        self._policy = policy or DecisionPolicy()
        self._enabled = (
            settings.ENABLE_VERIFICATION if enabled is None else enabled
        )

    def verify(
        self,
        answer: str,
        sources: list[RetrievalResult],
        language: LanguageCode = LanguageCode.ENGLISH,
    ) -> VerifiedAnswer:
        if not self._enabled:
            return self._passthrough(answer)

        signals = self._fact_verifier.verify(answer, sources)
        report = self._confidence.score(signals)
        verified = self._policy.decide(answer, report, language=language)

        logger.info(
            "verifier.completed",
            action=verified.action.value,
            confidence=verified.confidence_level.value,
            score=verified.confidence_score,
            flags=report.flags,
        )
        return verified

    @staticmethod
    def _passthrough(answer: str) -> VerifiedAnswer:
        """Verification disabled: return the answer as-is."""
        empty_report = VerificationReport(
            checks=[],
            confidence_score=1.0,
            confidence_level=ConfidenceLevel.HIGH,
            flags=[],
        )
        return VerifiedAnswer(
            text=answer,
            original_text=answer,
            action=VerificationAction.APPROVED,
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=1.0,
            report=empty_report,
        )