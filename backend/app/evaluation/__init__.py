"""
Answer-quality evaluation harness (, ).

A repeatable accuracy score you can watch move as you tune prompts,
model, and corpus - plus a self-run stand-in for the judges' hidden
prompts, so you don't overfit to your own submission questions.

    golden_set curated questions (EN primary, SW secondary)
    grader keyword/point/grounding/language scoring
    grounding_check did the answer use the retrieved sources?
    hallucination_probes out-of-corpus questions that must say 'unsure'
    report config-tagged, comparable scored report

The grader reuses the Verifier ( runtime) for the grounding and
hallucination judgement, so the harness and the live pipeline agree on
what "grounded" means.
"""

from __future__ import annotations

from app.evaluation.golden_set import GOLDEN_SET, GoldenCase, golden_cases
from app.evaluation.grader import AnswerGrader, GradeResult
from app.evaluation.grounding_check import GroundingResult, grounding_score
from app.evaluation.hallucination_probes import (
    HALLUCINATION_PROBES,
    HallucinationProbe,
    ProbeResult,
    grade_probe,
    said_unsure,
)
from app.evaluation.report import build_report, format_report
from app.evaluation.runner import evaluate

__all__ = [
    "evaluate",
    "GOLDEN_SET",
    "GoldenCase",
    "golden_cases",
    "AnswerGrader",
    "GradeResult",
    "GroundingResult",
    "grounding_score",
    "HALLUCINATION_PROBES",
    "HallucinationProbe",
    "ProbeResult",
    "grade_probe",
    "said_unsure",
    "build_report",
    "format_report",
]
