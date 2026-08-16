"""
Unit tests for Swahili quality.

Covers glossary term selection, the few-shot exemplars, the combined
enrichment, the builder wiring (Swahili only, toggleable), and the eval
set's integrity - all offline.
"""

from __future__ import annotations

import pytest
from app.config.settings import settings
from app.language.swahili import (
    SWAHILI_EVAL_SET,
    build_glossary,
    fewshot_block,
    relevant_terms,
    swahili_prompt_enrichment,
)
from app.models.language import LanguageCode
from app.orchestration.prompts import ContextBuilder, PromptBuilder

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Glossary injection
# ---------------------------------------------------------------------------


def test_relevant_terms_picks_swahili_domain_words():
    pairs = relevant_terms("Nyanya zangu zina ukungu, nifanye nini?")
    canonicals = {canonical for _forms, canonical in pairs}
    assert "tomato" in canonicals
    assert "blight" in canonicals


def test_build_glossary_maps_sw_to_canonical():
    block = build_glossary("Muhogo wangu una batobato")
    assert "cassava" in block
    assert "mosaic" in block
    assert "Istilahi" in block  # Swahili header


def test_glossary_empty_when_no_domain_terms():
    assert build_glossary("Habari yako rafiki?") == ""


def test_glossary_respects_max_terms():
    from app.config.constants import SWAHILI_GLOSSARY_MAX_TERMS

    # A query stuffed with many domain words must not exceed the cap.
    query = "mahindi maharagwe nyanya muhogo ukungu kutu kunyauka kuoza batobato wadudu"
    assert len(relevant_terms(query)) <= SWAHILI_GLOSSARY_MAX_TERMS

    # ---------------------------------------------------------------------------
    # Few-shot exemplars
    # ---------------------------------------------------------------------------


def test_fewshot_block_per_intent_and_cited():
    block = fewshot_block("diagnosis")
    assert "Mfano" in block
    assert "[Source 1]" in block  # exemplar models citing a source


def test_fewshot_unknown_intent_falls_back():
    assert fewshot_block("nonsense")  # non-empty (general exemplar)


def test_enrichment_combines_glossary_and_exemplar():
    text = swahili_prompt_enrichment("Nyanya zangu zina ukungu, nifanye nini?", "diagnosis")
    assert "Istilahi" in text and "Mfano" in text

    # ---------------------------------------------------------------------------
    # Builder wiring
    # ---------------------------------------------------------------------------


def _prompt(language, question, intent="diagnosis", enrich=None):
    ctx = ContextBuilder.build([])
    return PromptBuilder.build(
        question=question,
        context=ctx,
        language=language,
        intent=intent,
        enrich_swahili=enrich,
    )


def test_builder_enriches_swahili_only():
    sw = _prompt(LanguageCode.SWAHILI, "Nyanya zangu zina ukungu?")
    en = _prompt(LanguageCode.ENGLISH, "My tomatoes have blight?")
    assert "Istilahi" in sw.system_prompt and "Mfano" in sw.system_prompt
    assert "Istilahi" not in en.system_prompt and "Mfano" not in en.system_prompt
    # Prompt shape preserved.
    assert sw.full_prompt.rstrip().endswith("Answer:")


def test_builder_enrichment_toggle_off():
    off = _prompt(LanguageCode.SWAHILI, "Nyanya zangu zina ukungu?", enrich=False)
    assert "Istilahi" not in off.system_prompt
    assert "Mfano" not in off.system_prompt


def test_builder_enrichment_default_follows_setting():
    # Default is on; explicitly assert the setting drives it.
    assert settings.SWAHILI_PROMPT_ENRICH is True
    on = _prompt(LanguageCode.SWAHILI, "Nyanya zangu zina ukungu?", enrich=None)
    assert "Mfano" in on.system_prompt

    # ---------------------------------------------------------------------------
    # Eval set integrity
    # ---------------------------------------------------------------------------


def test_eval_set_wellformed():
    assert len(SWAHILI_EVAL_SET) >= 6
    ids = [c.id for c in SWAHILI_EVAL_SET]
    assert len(ids) == len(set(ids))  # unique ids
    intents = {c.intent for c in SWAHILI_EVAL_SET}
    assert {"diagnosis", "how_to", "price", "general"} <= intents
    for case in SWAHILI_EVAL_SET:
        assert case.question.strip()
        assert case.expected_terms  # every case scores something
