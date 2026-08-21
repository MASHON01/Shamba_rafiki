"""Unit tests for the prompt engine: system prompts, context builder,
prompt builder."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.config.constants import NO_CONTEXT_PLACEHOLDER
from app.models.document import DocumentChunk, RetrievalResult
from app.models.language import LanguageCode
from app.orchestration.prompts import (
    ContextBuilder, ConversationTurn, PromptBuilder, get_system_prompt,
)

pytestmark = pytest.mark.unit


def _result(text, score, **meta):
    chunk = DocumentChunk(
        document_id=uuid4(), chunk_index=0, text=text,
        token_count=len(text.split()),
        metadata={k: str(v) for k, v in meta.items()},
    )
    return RetrievalResult(chunk=chunk, similarity_score=score)


# --- system prompts -------------------------------------------------------

def test_system_prompt_english_diagnosis():
    p = get_system_prompt(LanguageCode.ENGLISH, "diagnosis")
    assert "Farm Pal" in p and "numbered steps" in p


def test_system_prompt_swahili():
    p = get_system_prompt(LanguageCode.SWAHILI, "diagnosis")
    assert "Kiswahili" in p


def test_system_prompt_fallbacks():
    # unknown intent -> general; unknown language -> english
    assert get_system_prompt(LanguageCode.ENGLISH, "xyz") == \
        get_system_prompt(LanguageCode.ENGLISH, "general")
    assert get_system_prompt(LanguageCode.UNKNOWN, "diagnosis") == \
        get_system_prompt(LanguageCode.ENGLISH, "diagnosis")


# --- context builder ------------------------------------------------------

def test_context_numbered_and_tagged():
    ctx = ContextBuilder().build([
        _result("Maize blight treatment.", 0.9, crop="maize", county="Nakuru"),
    ])
    assert ctx.has_context
    assert "[Source 1 (maize, Nakuru)]" in ctx.text


def test_context_filters_weak_matches():
    ctx = ContextBuilder().build([
        _result("strong", 0.9, crop="maize"),
        _result("weak noise", 0.05),
    ])
    assert len(ctx.sources) == 1 and "weak noise" not in ctx.text


def test_context_all_weak_placeholder():
    ctx = ContextBuilder().build([_result("noise", 0.05)])
    assert not ctx.has_context and ctx.text == NO_CONTEXT_PLACEHOLDER


def test_context_suppresses_unknown_tags():
    ctx = ContextBuilder().build([_result("text", 0.9, crop="unknown", county="unknown")])
    assert "unknown" not in ctx.text and "[Source 1]" in ctx.text


# --- prompt builder -------------------------------------------------------

def test_prompt_assembles_all_parts():
    ctx = ContextBuilder().build([_result("Maize blight: use fungicide.", 0.9, crop="maize")])
    bp = PromptBuilder().build(
        question="How do I treat maize blight?", context=ctx,
        language=LanguageCode.ENGLISH, intent="diagnosis")
    assert "Farm Pal" in bp.system_prompt
    assert "Maize blight: use fungicide." in bp.user_prompt
    assert "How do I treat maize blight?" in bp.user_prompt
    assert bp.full_prompt.rstrip().endswith("Answer:")


def test_prompt_includes_history():
    ctx = ContextBuilder().build([_result("x", 0.9, crop="maize")])
    bp = PromptBuilder().build(
        question="How treat it?", context=ctx,
        history=[ConversationTurn(question="What is blight?", answer="Fungal disease.")])
    assert "What is blight?" in bp.full_prompt
    assert "Fungal disease." in bp.full_prompt