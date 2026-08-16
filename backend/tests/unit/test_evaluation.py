"""
Unit tests for the answer-quality harness.

Grounding check, grader sub-scores + overall, hallucination-probe
grading, and report aggregation - all offline and deterministic
(VerifiedAnswer objects are constructed directly to control action).
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.evaluation import (
    AnswerGrader,
    build_report,
    format_report,
    grade_probe,
    grounding_score,
    said_unsure,
)
from app.evaluation.golden_set import GOLDEN_SET, GoldenCase, golden_cases
from app.evaluation.hallucination_probes import HALLUCINATION_PROBES, HallucinationProbe
from app.models.classifier import ConfidenceLevel
from app.models.document import DocumentChunk, RetrievalResult
from app.models.verification import (
    VerificationAction,
    VerificationReport,
    VerifiedAnswer,
)

pytestmark = pytest.mark.unit


def _result(text, score=0.9, **meta):
    chunk = DocumentChunk(
        document_id=uuid4,
        chunk_index=0,
        text=text,
        token_count=len(text.split()),
        metadata={k: str(v) for k, v in meta.items()},
    )
    return RetrievalResult(chunk=chunk, similarity_score=score)


def _verified(text, action=VerificationAction.APPROVED, score=0.9, flags=None):
    report = VerificationReport(
        checks=[],
        confidence_score=score,
        confidence_level=ConfidenceLevel.HIGH,
        flags=flags or [],
    )
    return VerifiedAnswer(
        text=text,
        original_text=text,
        action=action,
        confidence_level=ConfidenceLevel.HIGH,
        confidence_score=score,
        report=report,
    )

    # ---------------------------------------------------------------------------
    # Golden set + probes integrity
    # ---------------------------------------------------------------------------


def test_golden_set_has_en_sw_and_hidden():
    langs = {c.language for c in GOLDEN_SET}
    cats = {c.category for c in GOLDEN_SET}
    assert {"en", "sw"} <= langs
    assert "hidden" in cats
    assert golden_cases(language="en")  # filter works
    assert all(c.category == "hidden" for c in golden_cases(category="hidden"))


def test_probe_ids_unique():
    ids = [p.id for p in HALLUCINATION_PROBES]
    assert len(ids) == len(set(ids)) and len(ids) >= 3

    # ---------------------------------------------------------------------------
    # Grounding check
    # ---------------------------------------------------------------------------


def test_grounding_not_applicable_without_sources():
    g = grounding_score("anything", [])
    assert g.applicable is False and g.used is False


def test_grounding_detects_citation_and_overlap():
    sources = [_result("Maize blight is treated with mancozeb fungicide.")]
    g = grounding_score("Apply mancozeb fungicide to the maize blight [Source 1].", sources)
    assert g.cited is True
    assert g.overlap > 0.0
    assert g.used is True
    assert 0.0 < g.score <= 1.0


def test_grounding_ungrounded_answer_scores_low():
    sources = [_result("Maize blight is treated with mancozeb fungicide.")]
    g = grounding_score("The weather is nice and markets are open today.", sources)
    assert g.cited is False
    assert g.used is False

    # ---------------------------------------------------------------------------
    # Grader
    # ---------------------------------------------------------------------------


def test_grader_scores_good_grounded_answer_high():
    case = GoldenCase(
        id="t1",
        question="q",
        intent="diagnosis",
        expected_terms=("blight", "maize", "fungicide"),
    )
    sources = [_result("Maize blight: apply mancozeb fungicide and rotate.")]
    answer = "Your maize has blight. Apply mancozeb fungicide [Source 1] and rotate."
    grade = AnswerGrader.grade(case, answer, sources, _verified(answer))
    assert grade.term_coverage == 1.0
    assert grade.grounding.used is True
    assert grade.passed is True


def test_grader_penalizes_missing_concepts():
    case = GoldenCase(
        id="t2",
        question="q",
        intent="diagnosis",
        expected_terms=("blight", "maize", "fungicide"),
    )
    sources = [_result("Maize blight: apply mancozeb fungicide.")]
    answer = "The plant looks unwell. Try something."
    grade = AnswerGrader.grade(case, answer, sources, _verified(answer))
    assert grade.term_coverage < 0.5
    assert grade.passed is False


def test_grader_hidden_case_ignores_grounding_weight():
    # must_ground False -> grounding weight redistributed to content.
    case = GoldenCase(
        id="t3",
        question="q",
        intent="how_to",
        expected_terms=("maize", "seed"),
        must_ground=False,
        category="hidden",
    )
    answer = "For maize, use certified seed at the right spacing."
    grade = AnswerGrader.grade(case, answer, [], _verified(answer))
    assert grade.term_coverage == 1.0
    assert grade.passed is True  # not dragged down by absent citations

    # ---------------------------------------------------------------------------
    # Hallucination probes
    # ---------------------------------------------------------------------------


def test_said_unsure_detects_markers():
    assert said_unsure("I'm not sure, please consult an extension officer.")
    assert said_unsure("Sina uhakika na jibu hili.")
    assert not said_unsure("Use exactly 40 kg per acre.")


def test_probe_passes_when_uncertain():
    probe = HallucinationProbe(id="p", question="q")
    v = _verified("I'm not certain; consult a local agricultural officer.")
    assert grade_probe(probe, v).passed is True


def test_probe_passes_when_verifier_declines():
    probe = HallucinationProbe(id="p", question="q")
    v = _verified("Use 40 kg DAP per acre.", action=VerificationAction.REPLACED)
    assert grade_probe(probe, v).passed is True


def test_probe_fails_on_confident_unsupported_answer():
    probe = HallucinationProbe(id="p", question="q")
    v = _verified(
        "Use exactly 40 kg DAP per acre for best yield.", action=VerificationAction.APPROVED
    )
    r = grade_probe(probe, v)
    assert r.passed is False and r.said_unsure is False

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------


def test_build_and_format_report():
    case = GoldenCase(id="c", question="q", intent="diagnosis", expected_terms=("maize",))
    grade = AnswerGrader.grade(
        case, "maize blight [Source 1]", [_result("maize blight")], _verified("x")
    )
    probe = HallucinationProbe(id="p", question="q")
    probe_res = grade_probe(probe, _verified("I'm not sure."))

    config = {
        "prompt_version": "v1",
        "model_id": "m",
        "swahili_enrich": True,
        "verification_enabled": True,
    }
    report = build_report(config, [grade], [probe_res])

    assert report["summary"]["cases"] == 1
    assert report["summary"]["hallucination_pass_rate"] == 1.0
    assert "by_intent" in report["summary"]
    text = format_report(report)
    assert "prompt_version: v1" in text
    assert "hallucination" in text
