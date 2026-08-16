"""
Decision policies.

Given a verification report, decides what actually happens to the
answer. This is the conservative "policy A": caveat wherever
reasonable rather than reject, so the farmer still gets the model's
best answer clearly flagged when uncertain - and only replace the
answer outright in the narrow, genuinely dangerous case.

The rules, in order:

1. REPLACE (safe fallback) only when the answer both makes a specific
   fabricated claim (an unsupported dose/price - the
   'unsupported_specific' flag) AND rests on essentially no evidence
   (evidence score <= POLICY_REJECT_MAX_EVIDENCE). This is the one
   case where handing the farmer the answer could cause real harm, so
   it's withheld.

2. CAVEAT when confidence is not HIGH, or any flag was raised. The
   answer is returned with an honest "I'm not fully certain" note in
   the farmer's language.

3. APPROVE as-is when confidence is HIGH and nothing was flagged.

All thresholds and messages live in constants, so this policy can be
tuned during hidden-prompt stress testing without touching detection
or scoring logic.
"""

from __future__ import annotations

from app.config.constants import (
    CAVEAT_EN,
    CAVEAT_SW,
    POLICY_REJECT_MAX_EVIDENCE,
    SAFE_FALLBACK_EN,
    SAFE_FALLBACK_SW,
)
from app.models.classifier import ConfidenceLevel
from app.models.language import LanguageCode
from app.models.verification import (
    VerificationAction,
    VerificationReport,
    VerifiedAnswer,
)


class DecisionPolicy:
    """
    Conservative approve / caveat / replace policy.
    """

    def decide(
        self,
        answer: str,
        report: VerificationReport,
        language: LanguageCode = LanguageCode.ENGLISH,
    ) -> VerifiedAnswer:
        if self._should_replace(report):
            action = VerificationAction.REPLACED
            text = self._fallback_for(language)
        elif self._should_caveat(report):
            action = VerificationAction.CAVEATED
            text = answer + self._caveat_for(language)
        else:
            action = VerificationAction.APPROVED
            text = answer

        return VerifiedAnswer(
            text=text,
            original_text=answer,
            action=action,
            confidence_level=report.confidence_level,
            confidence_score=report.confidence_score,
            report=report,
        )

    def _should_replace(self, report: VerificationReport) -> bool:
        """
        Replace only when a fabricated specific coincides with
        essentially no supporting evidence - the one genuinely
        harmful case.
        """
        has_fabricated_specific = any(
            flag.startswith("unsupported_specific") for flag in report.flags
        )
        if not has_fabricated_specific:
            return False

        evidence = next(
            (c.score for c in report.checks if c.name == "evidence"), 0.0
        )
        return evidence <= POLICY_REJECT_MAX_EVIDENCE

    @staticmethod
    def _should_caveat(report: VerificationReport) -> bool:
        return (
            report.confidence_level != ConfidenceLevel.HIGH
            or bool(report.flags)
        )

    @staticmethod
    def _caveat_for(language: LanguageCode) -> str:
        return CAVEAT_SW if language == LanguageCode.SWAHILI else CAVEAT_EN

    @staticmethod
    def _fallback_for(language: LanguageCode) -> str:
        return (
            SAFE_FALLBACK_SW
            if language == LanguageCode.SWAHILI
            else SAFE_FALLBACK_EN
        )