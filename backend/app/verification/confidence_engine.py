"""
Confidence engine.

Turns the fact verifier's per-check signals into a single confidence
score and level. The score is a weighted blend of the four check
scores (CONFIDENCE_WEIGHTS), with hallucination and citation weighted
highest because an unsupported specific or an uncited claim is the
most direct hallucination signal.

The blended score maps to the existing ConfidenceLevel enum
(LOW/MEDIUM/HIGH) via two thresholds. The engine also carries every
check's flags up into the report, so the policy layer can see not
just "how confident" but "which specific problems" - the difference
between a slightly-thin answer and one with a fabricated dose.

Produces the complete `VerificationReport`.
"""

from __future__ import annotations

from app.config.constants import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    CONFIDENCE_WEIGHTS,
)
from app.models.classifier import ConfidenceLevel
from app.models.verification import CheckSignal, VerificationReport


class ConfidenceEngine:
    """
    Blends check signals into a confidence score and level.
    """

    def score(self, signals: list[CheckSignal]) -> VerificationReport:
        by_name = {s.name: s for s in signals}

        # Weighted blend over whichever weighted checks are present.
        # If a check is missing, its weight is redistributed implicitly
        # by normalizing over the weights actually used.
        total_weight = 0.0
        weighted_sum = 0.0
        for name, weight in CONFIDENCE_WEIGHTS.items():
            signal = by_name.get(name)
            if signal is None:
                continue
            weighted_sum += weight * signal.score
            total_weight += weight

        confidence_score = (
            round(weighted_sum / total_weight, 3) if total_weight else 0.0
        )

        level = self._level_for(confidence_score)

        flags = sorted({flag for s in signals for flag in s.flags})

        return VerificationReport(
            checks=signals,
            confidence_score=confidence_score,
            confidence_level=level,
            flags=flags,
        )

    @staticmethod
    def _level_for(score: float) -> ConfidenceLevel:
        if score >= CONFIDENCE_HIGH_THRESHOLD:
            return ConfidenceLevel.HIGH
        if score >= CONFIDENCE_MEDIUM_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW