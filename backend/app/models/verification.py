"""
Verification-related models.

The public schemas the verification engine produces:

    CheckSignal          one check's result, as a Pydantic model
    VerificationReport   all four checks + the blended confidence
    VerifiedAnswer       the final answer text + confidence + action

These wrap the internal `CheckResult` dataclasses (used inside the
verification package) into serializable models the orchestrator and
API layer can return. Reuses the existing ConfidenceLevel enum from
the classifier models rather than defining a second one.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.classifier import ConfidenceLevel


class VerificationAction(str, Enum):
    """
    What the decision policy did with the answer.

        APPROVED   returned as-is (high confidence)
        CAVEATED   returned with an uncertainty note appended
        REPLACED   answer withheld, safe fallback returned instead
    """

    APPROVED = "approved"
    CAVEATED = "caveated"
    REPLACED = "replaced"


class CheckSignal(BaseModel):
    """One verification check's result."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    detail: str = ""
    flags: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """
    The full outcome of verifying an answer: every check signal plus
    the blended confidence score and level.
    """

    checks: list[CheckSignal]
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    flags: list[str] = Field(default_factory=list)


class VerifiedAnswer(BaseModel):
    """
    The final, policy-processed answer returned to the orchestrator.

    `text` is what the farmer should see (possibly caveated or
    replaced); `original_text` preserves the model's raw answer for
    logging/debugging; `report` explains the decision.
    """

    text: str
    original_text: str
    action: VerificationAction
    confidence_level: ConfidenceLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    report: VerificationReport