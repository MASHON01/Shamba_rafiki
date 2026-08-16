"""Unit tests for verification: the four checks, confidence engine, policy,
and the top-level verifier."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.models.classifier import ConfidenceLevel
from app.models.document import DocumentChunk, RetrievalResult
from app.models.language import LanguageCode
from app.models.verification import VerificationAction
from app.verification.citation_checker import CitationChecker
from app.verification.confidence_engine import ConfidenceEngine
from app.verification.evidence_scorer import EvidenceScorer
from app.verification.fact_verifier import FactVerifier
from app.verification.hallucination_detector import HallucinationDetector
from app.verification.policies import DecisionPolicy
from app.verification.semantic_validator import SemanticValidator
from app.verification.verifier import Verifier

pytestmark = pytest.mark.unit


def _src(text, score, **meta):
    chunk = DocumentChunk(
        document_id=uuid4(), chunk_index=0, text=text,
        token_count=len(text.split()),
        metadata={k: str(v) for k, v in meta.items()})
    return RetrievalResult(chunk=chunk, similarity_score=score)


SOURCES = [
    _src("Maize leaf blight is a fungal disease. Apply mancozeb fungicide at "
         "40 grams per 20 litres of water. Spray every 7 days.", 0.88, crop="maize"),
    _src("Crop rotation with beans reduces blight in following seasons.", 0.72, crop="maize"),
]


# --- individual checks ----------------------------------------------------

def test_evidence_scorer():
    assert EvidenceScorer().score(SOURCES).score > 0.5
    r = EvidenceScorer().score([])
    assert r.score == 0.0 and "no_evidence" in r.flags


def test_semantic_validator():
    good = SemanticValidator().validate("Apply mancozeb fungicide, rotate beans.", SOURCES)
    assert good.passed
    drift = SemanticValidator().validate("Install solar panels and wifi routers.", SOURCES)
    assert not drift.passed and "low_overlap" in drift.flags


def test_citation_checker_invalid_reference():
    r = CitationChecker().check("According to [Source 5], spray daily.", SOURCES)
    assert any("invalid_citation" in f for f in r.flags)


def test_hallucination_detector_supported_vs_fabricated():
    ok = HallucinationDetector().detect(
        "Apply 40 grams per 20 litres every 7 days.", SOURCES)
    assert ok.passed and ok.score == 1.0
    bad = HallucinationDetector().detect(
        "Apply 99 grams per 3 litres every 2 days.", SOURCES)
    assert not bad.passed
    assert any("unsupported_specific" in f for f in bad.flags)


# --- confidence engine ----------------------------------------------------

def test_confidence_high_for_grounded():
    signals = FactVerifier().verify(
        "Apply mancozeb fungicide at 40 grams per 20 litres every 7 days. Rotate beans.",
        SOURCES)
    report = ConfidenceEngine().score(signals)
    assert report.confidence_level == ConfidenceLevel.HIGH


def test_confidence_low_for_fabricated_ungrounded():
    signals = FactVerifier().verify(
        "Apply 99 grams per 3 litres every 2 days and expect 500 kg more.", [])
    report = ConfidenceEngine().score(signals)
    assert report.confidence_level == ConfidenceLevel.LOW
    assert any("unsupported_specific" in f for f in report.flags)


def test_check_order_is_stable():
    signals = FactVerifier().verify("anything", SOURCES)
    assert [s.name for s in signals] == \
        ["evidence", "semantic", "citation", "hallucination"]


# --- policy ---------------------------------------------------------------

def test_policy_approves_clean_high():
    good = "Apply mancozeb fungicide at 40 grams per 20 litres every 7 days. Rotate beans."
    report = ConfidenceEngine().score(FactVerifier().verify(good, SOURCES))
    v = DecisionPolicy().decide(good, report, language=LanguageCode.ENGLISH)
    assert v.action == VerificationAction.APPROVED and v.text == good


def test_policy_caveats_flagged():
    partial = "Spray every 7 days and expect 500 kg extra yield."
    report = ConfidenceEngine().score(FactVerifier().verify(partial, SOURCES))
    v = DecisionPolicy().decide(partial, report, language=LanguageCode.ENGLISH)
    assert v.action == VerificationAction.CAVEATED
    assert v.text.startswith(partial) and v.original_text == partial


def test_policy_replaces_fabricated_ungrounded():
    danger = "Apply 99 grams per 3 litres every 2 days."
    report = ConfidenceEngine().score(FactVerifier().verify(danger, []))
    v = DecisionPolicy().decide(danger, report, language=LanguageCode.ENGLISH)
    assert v.action == VerificationAction.REPLACED
    assert "99 grams" not in v.text


def test_policy_swahili_caveat():
    partial = "Spray every 7 days and expect 500 kg extra yield."
    report = ConfidenceEngine().score(FactVerifier().verify(partial, SOURCES))
    v = DecisionPolicy().decide(partial, report, language=LanguageCode.SWAHILI)
    assert "Kumbuka" in v.text


# --- verifier toggle ------------------------------------------------------

def test_verifier_disabled_passthrough():
    v = Verifier(enabled=False).verify("anything 99 grams", [])
    assert v.action == VerificationAction.APPROVED and v.text == "anything 99 grams"