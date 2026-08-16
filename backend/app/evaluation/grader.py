"""
Automated answer grader.

Turns one (question, answer, sources, verifier-result) into a set of
comparable sub-scores and a single overall number, so tuning a prompt,
model, or corpus produces a metric that visibly moves. It measures what
can be measured cheaply and honestly, offline:

  - concept coverage: did the answer engage the expected domain concepts
    (the English canonical term OR its Swahili form both count)?
  - point coverage: did it hit the expected key points/phrases?
  - grounding: is it anchored to the retrieved sources (grounding_check)?
  - language: did it answer in the language asked?
  - confidence/action: what did the Verifier make of it?

It is not a human judge and doesn't pretend to be; it's a fast, stable
proxy that catches regressions and ranks configs. The Verifier supplies
the grounding/hallucination judgement so this module doesn't re-derive
it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config.constants import AGRI_TERMS, KNOWN_CROPS
from app.evaluation.golden_set import GoldenCase
from app.evaluation.grounding_check import GroundingResult, grounding_score
from app.models.document import RetrievalResult
from app.models.language import LanguageCode
from app.models.verification import VerifiedAnswer

# canonical term -> every surface form (English + Swahili), for coverage.
_SURFACE = {**KNOWN_CROPS, **AGRI_TERMS}
_WORD_RE = re.compile(r"[a-zA-ZÀ-ſ]+")


@dataclass(slots=True)
class GradeResult:
    """A graded answer, sub-scores plus one overall."""

    case_id: str
    category: str
    language: str
    intent: str
    term_coverage: float
    point_coverage: float
    grounding: GroundingResult
    language_match: bool
    confidence_score: float
    action: str
    overall: float
    passed: bool
    flags: list[str] = field(default_factory=list)


class AnswerGrader:
    """Scores an answer against a GoldenCase."""

    def __init__(self, pass_threshold: float = 0.6, lang_detector=None) -> None:
        self._pass = pass_threshold
        self._detector = lang_detector  # optional; lazily built if needed

    def grade(
        self,
        case: GoldenCase,
        answer: str,
        sources: list[RetrievalResult],
        verified: VerifiedAnswer,
    ) -> GradeResult:
        term_cov = _term_coverage(answer, case.expected_terms)
        point_cov = _point_coverage(answer, case.expected_points)
        grounding = grounding_score(answer, sources)
        lang_ok = self._language_ok(answer, case.language)

        overall = self._overall(case, term_cov, point_cov, grounding, lang_ok)

        return GradeResult(
            case_id=case.id,
            category=case.category,
            language=case.language,
            intent=case.intent,
            term_coverage=term_cov,
            point_coverage=point_cov,
            grounding=grounding,
            language_match=lang_ok,
            confidence_score=verified.confidence_score,
            action=verified.action.value,
            overall=overall,
            passed=overall >= self._pass,
            flags=list(verified.report.flags),
        )

        # ------------------------------------------------------------------

    def _overall(self, case, term_cov, point_cov, grounding, lang_ok) -> float:
        # Concept coverage is the backbone; points refine it.
        content = term_cov if not case.expected_points else (0.7 * term_cov + 0.3 * point_cov)
        lang = 1.0 if lang_ok else 0.0

        # Grounding only counts when there were sources to ground in AND
        # the case is expected to be grounded; otherwise redistribute its
        # weight to content (a hidden/no-corpus case is judged on whether
        # it engaged the concept and language, not on citations).
        if grounding.applicable and case.must_ground:
            return round(0.5 * content + 0.3 * grounding.score + 0.2 * lang, 4)
        return round(0.75 * content + 0.25 * lang, 4)

    def _language_ok(self, answer: str, expected: str) -> bool:
        if not answer.strip():
            return False
        detector = self._detector or self._make_detector
        detected = detector.detect(answer).language
        want = LanguageCode.SWAHILI if expected == "sw" else LanguageCode.ENGLISH
        # UNKNOWN (very short answers) is not counted as a mismatch.
        return detected in (want, LanguageCode.UNKNOWN)

    def _make_detector(self):
        from app.language.detector import LanguageDetector

        self._detector = LanguageDetector
        return self._detector

        # ---------------------------------------------------------------------------
        # Coverage helpers
        # ---------------------------------------------------------------------------


def _term_coverage(answer: str, expected_terms) -> float:
    if not expected_terms:
        return 1.0
    low = answer.lower()
    hits = 0
    for canonical in expected_terms:
        forms = _SURFACE.get(canonical, [canonical])
        if canonical.lower() in low or any(f.lower() in low for f in forms):
            hits += 1
    return hits / len(expected_terms)


def _point_coverage(answer: str, expected_points) -> float:
    if not expected_points:
        return 1.0
    low = answer.lower()
    hits = sum(1 for point in expected_points if point.lower() in low)
    return hits / len(expected_points)
