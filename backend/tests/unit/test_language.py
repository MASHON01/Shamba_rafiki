"""Unit tests for the language layer: detector, normalizer, extractors,
terminology, translator."""

from __future__ import annotations

import pytest

from app.language import InputNormalizer, LanguageAnalyzer, LanguageDetector
from app.language.extractors import EntityExtractor, IntentExtractor
from app.language.terminology import AgriTerminology, CropNormalizer
from app.language.translator import Translator, TranslationNotAvailableError
from app.models.language import LanguageCode

pytestmark = pytest.mark.unit


# --- normalizer -----------------------------------------------------------

def test_normalizer_collapses_and_lowercases():
    assert InputNormalizer().normalize("  Hello   WORLD ") == "hello world"


def test_normalizer_non_destructive():
    # words preserved, only whitespace/case/punct tidied
    assert InputNormalizer().normalize("Nyanya zangu!") == "nyanya zangu!"


def test_tokenize_strips_punctuation():
    assert InputNormalizer().tokenize("maize, beans; tomato!") == \
        ["maize", "beans", "tomato"]


# --- detector -------------------------------------------------------------

def test_detect_english():
    r = LanguageDetector().detect("How do I treat the disease on my maize?")
    assert r.language == LanguageCode.ENGLISH and r.confidence > 0.5


def test_detect_swahili():
    r = LanguageDetector().detect("Mahindi yangu yana ugonjwa gani na nifanye nini?")
    assert r.language == LanguageCode.SWAHILI and r.confidence > 0.5


def test_detect_unknown_on_bare_word():
    assert LanguageDetector().detect("nyanya").language == LanguageCode.UNKNOWN


# --- entity extractor -----------------------------------------------------

def test_entities_english():
    ents = EntityExtractor().extract("My maize in Nakuru has leaf blight")
    crops = {e.normalized_value for e in ents if e.entity_type == "crop"}
    counties = {e.normalized_value for e in ents if e.entity_type == "county"}
    assert crops == {"maize"} and counties == {"Nakuru"}


def test_entities_swahili_resolve_to_canonical():
    ents = EntityExtractor().extract("Nyanya zangu zina ukungu")
    crops = {e.normalized_value for e in ents if e.entity_type == "crop"}
    agri = {e.normalized_value for e in ents if e.entity_type == "agri_term"}
    assert crops == {"tomato"} and "blight" in agri


def test_entities_whole_token_guard():
    # 'corn'/'meru' must not fire inside 'acorn'/'numerous'
    assert EntityExtractor().extract("the acorn fell on numerous stones") == []


# --- intent extractor -----------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("My maize has spots and is dying", "diagnosis"),
    ("Nyanya zangu zina madoa", "diagnosis"),
    ("What is the price of beans at the market?", "price"),
    ("How do I plant tomatoes?", "how_to"),
    ("Tell me about maize", "general"),
])
def test_intent_classification(query, expected):
    assert IntentExtractor().extract(query) == expected


# --- terminology ----------------------------------------------------------

def test_crop_normalizer():
    c = CropNormalizer()
    assert c.normalize("mahindi") == "maize"
    assert c.normalize("corn") == "maize"
    assert c.normalize("wheat") is None


def test_agri_terminology():
    t = AgriTerminology()
    assert t.normalize("ukungu") == "blight"
    assert t.normalize("dawa") == "pesticide"
    assert t.normalize("hello") is None


# --- translator (dictionary fallback) -------------------------------------

def test_translator_dictionary_fallback():
    res = Translator().translate("nyanya zangu zina ukungu")
    assert "tomato" in res.translated_text and "blight" in res.translated_text


def test_translator_engine_required_when_forced():
    with pytest.raises(TranslationNotAvailableError):
        Translator().translate_with_engine("nyanya")


# --- analyzer (end-to-end of the language layer) --------------------------

def test_analyzer_full_swahili():
    full = LanguageAnalyzer().analyze_full(
        "Mahindi yangu yana ukungu, nifanye nini?", language_hint="sw")
    assert full.detection.language == LanguageCode.SWAHILI
    assert full.intent == "diagnosis"
    assert any(e.normalized_value == "maize" for e in full.entities)
    assert "tomato" not in full.retrieval_query  # it's maize, not tomato
    assert "blight" in full.retrieval_query  # ukungu glossed