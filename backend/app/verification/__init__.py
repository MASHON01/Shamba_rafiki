"""
Verification & confidence engine (Phase 1, Output 7).

The safety layer between the LLM's raw answer and the farmer:

    LLM Response -> Verification -> Confidence -> Approved Response

It is the runtime half of hallucination control (grounding-first
system prompts were the generation-side half). Each check turns the
answer + its retrieved sources into one signal; the fact verifier
aggregates them, the confidence engine scores them, and the decision
policy decides whether to approve, caveat, or replace the answer.

This is what lets Shamba Rafiki honestly say "I'm not certain"
instead of confidently fabricating - the behaviour the hidden test
prompts are designed to probe.

Checks (this batch):
    EvidenceScorer          how strongly sources back the answer
    CitationChecker         claims traceable to numbered [Source N]
    SemanticValidator       answer content overlaps retrieved context
    HallucinationDetector   specific claims (numbers/doses) unsupported

Coordination / decision (later batches):
    fact_verifier, confidence_engine, policies, verifier
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CheckResult:
    """
    The output of a single verification check.

    A common, minimal shape so the fact verifier can aggregate every
    check uniformly, before the full `models/verification.py` schemas
    (next batch) wrap these into the public VerificationReport.

    Fields:
        name     which check produced this
        score    0.0-1.0, higher = more trustworthy on this axis
        passed   whether the check's own threshold was met
        detail   short human-readable explanation
        flags    specific problems found (e.g. unsupported numbers)
    """

    name: str
    score: float
    passed: bool
    detail: str = ""
    flags: list[str] = field(default_factory=list)


__all__ = ["CheckResult"]


def __getattr__(name: str):
    """
    Lazily expose the check classes so importing the package for just
    `CheckResult` (as the check modules themselves do) doesn't create
    a circular import at module load.
    """
    _lazy = {
        "EvidenceScorer": "app.verification.evidence_scorer",
        "SemanticValidator": "app.verification.semantic_validator",
        "CitationChecker": "app.verification.citation_checker",
        "HallucinationDetector": "app.verification.hallucination_detector",
    }
    if name in _lazy:
        import importlib

        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")