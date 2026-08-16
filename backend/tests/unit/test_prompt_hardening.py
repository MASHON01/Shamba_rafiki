"""
Unit tests for prompt versioning + budgeting.

The prompt tests (test_prompts.py) still pin the v1 baseline;
these cover the new versioned registry, the hardened v2 content, the
context-window budgeter, and the builder wiring - all offline.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from app.config.constants import DEFAULT_PROMPT_VERSION
from app.config.settings import settings
from app.models.document import DocumentChunk, RetrievalResult
from app.models.language import LanguageCode
from app.orchestration.prompts import (
    ContextBuilder,
    ConversationTurn,
    PromptBuilder,
    estimate_tokens,
    fit_history,
    get_system_prompt,
    list_versions,
)
from app.orchestration.prompts.budgeter import _render_turn

pytestmark = pytest.mark.unit


def _result(text, score, **meta):
    chunk = DocumentChunk(
        document_id=uuid4,
        chunk_index=0,
        text=text,
        token_count=len(text.split()),
        metadata={k: str(v) for k, v in meta.items()},
    )
    return RetrievalResult(chunk=chunk, similarity_score=score)

    # ---------------------------------------------------------------------------
    # Versioned registry
    # ---------------------------------------------------------------------------


def test_versions_registered():
    assert "v1" in list_versions and "v2" in list_versions


def test_default_matches_settings_version():
    # Default (no version arg) resolves to the configured baseline.
    assert get_system_prompt(LanguageCode.ENGLISH, "diagnosis") == get_system_prompt(
        LanguageCode.ENGLISH, "diagnosis", settings.PROMPT_VERSION
    )
    assert settings.PROMPT_VERSION == DEFAULT_PROMPT_VERSION == "v1"


def test_v1_and_v2_differ():
    v1 = get_system_prompt(LanguageCode.ENGLISH, "diagnosis", "v1")
    v2 = get_system_prompt(LanguageCode.ENGLISH, "diagnosis", "v2")
    assert v1 != v2


def test_unknown_version_falls_back_to_default():
    assert get_system_prompt(LanguageCode.ENGLISH, "diagnosis", "v999") == get_system_prompt(
        LanguageCode.ENGLISH, "diagnosis", "v1"
    )


def test_v2_hardening_content():
    v2 = get_system_prompt(LanguageCode.ENGLISH, "diagnosis", "v2")
    # Grounding-first, citation, and uncertainty language are explicit.
    assert "[Source 1]" in v2
    assert "unsure" in v2.lower()
    assert "Ground every specific claim" in v2


def test_v2_swahili_stays_swahili():
    v2 = get_system_prompt(LanguageCode.SWAHILI, "diagnosis", "v2")
    assert "Kiswahili" in v2
    assert "[Source 1]" in v2  # citation shape shared across languages


def test_language_and_intent_fallbacks_hold_per_version():
    # unknown intent -> general; unknown language -> english, within v2.
    assert get_system_prompt(LanguageCode.ENGLISH, "xyz", "v2") == get_system_prompt(
        LanguageCode.ENGLISH, "general", "v2"
    )
    assert get_system_prompt(LanguageCode.UNKNOWN, "diagnosis", "v2") == get_system_prompt(
        LanguageCode.ENGLISH, "diagnosis", "v2"
    )

    # ---------------------------------------------------------------------------
    # Budgeter
    # ---------------------------------------------------------------------------


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcde") == 2  # ceil(5/4)


def _turns(n):
    # Each rendered turn is a fixed, known size for deterministic budgeting.
    return [ConversationTurn(question="AAAA", answer="BBBB") for _ in range(n)]


def test_fit_history_drops_oldest_first():
    turns = _turns(5)
    per_turn = estimate_tokens(_render_turn(turns[0]))
    # Budget for exactly 2 turns (fixed/reserve/margin zeroed out).
    window = per_turn * 2
    kept, report = fit_history(
        turns,
        fixed_text="",
        window_tokens=window,
        answer_reserve_tokens=0,
        safety_margin_tokens=0,
    )
    assert report.kept_turns == 2
    assert report.dropped_turns == 3
    # The two KEPT turns are the most recent, in original order.
    assert kept == turns[-2:]


def test_fit_history_keeps_all_when_roomy():
    turns = _turns(3)
    kept, report = fit_history(turns, fixed_text="", window_tokens=100_000)
    assert kept == turns and report.dropped_turns == 0


def test_fit_history_overflow_drops_everything_and_flags():
    turns = _turns(3)
    kept, report = fit_history(
        turns,
        fixed_text="x" * 10_000,
        window_tokens=50,
        answer_reserve_tokens=0,
        safety_margin_tokens=0,
    )
    assert kept == []
    assert report.fits is False

    # ---------------------------------------------------------------------------
    # Builder wiring
    # ---------------------------------------------------------------------------


def test_builder_uses_requested_version():
    ctx = ContextBuilder.build([_result("Maize blight: use fungicide.", 0.9, crop="maize")])
    bp = PromptBuilder.build(
        question="How do I treat maize blight?",
        context=ctx,
        language=LanguageCode.ENGLISH,
        intent="diagnosis",
        version="v2",
    )
    assert "[Source 1]" in bp.system_prompt  # v2 selected
    assert bp.full_prompt.rstrip().endswith("Answer:")  # shape preserved


def test_builder_trims_history_to_window(monkeypatch):
    # Force a tiny window so the builder's budgeter has to drop turns.
    monkeypatch.setattr(settings, "MODEL_CONTEXT_SIZE", 200)
    ctx = ContextBuilder.build([_result("x", 0.9, crop="maize")])
    history = [
        ConversationTurn(question=f"Q{i} " + "word " * 20, answer="A " + "word " * 20)
        for i in range(10)
    ]
    bp = PromptBuilder.build(
        question="latest question?",
        context=ctx,
        history=history,
        intent="general",
    )
    assert len(bp.history) < len(history)  # something was trimmed
    assert bp.history == history[len(history) - len(bp.history) :]  # newest kept, in order
