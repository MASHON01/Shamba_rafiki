"""
Swahili glossary injection.

The corpus is in English; a farmer's question is in Swahili. The
language layer already glosses the *query* to English for retrieval, but
the model still has to answer, and it grounds better when it is told, in
the prompt, which English term in the reference material corresponds to
the Swahili word the farmer used: that 'ukungu' is 'blight', 'muhogo' is
'cassava'. That small dictionary is the cheapest, safest Swahili-quality
win - no fine-tune required.

This module scans the question for known domain words (reusing the same
CropNormalizer / AgriTerminology that ingestion and retrieval use, so the
mapping never drifts) and renders a compact glossary block for the
prompt. Only terms actually present in the question are included, so the
block stays short and relevant and never bloats the token budget.
"""

from __future__ import annotations

import re

from app.config.constants import AGRI_TERMS, KNOWN_CROPS, SWAHILI_GLOSSARY_MAX_TERMS
from app.language.terminology import AgriTerminology, CropNormalizer

_GLOSSARY_HEADER = (
    "Istilahi (neno la Kiswahili = neno la Kiingereza linalotumika " "katika marejeo):"
)

# Canonical term -> all its surface forms, from both vocabularies.
_SURFACE_FORMS: dict[str, list[str]] = {**KNOWN_CROPS, **AGRI_TERMS}

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ſ]+")

# Built once and reused; the normalizers are cheap, pure lookups.
_crops = CropNormalizer
_terms = AgriTerminology


def relevant_terms(text: str) -> list[tuple[str, str]]:
    """
    Domain terms present in ``text``, as (swahili_forms, canonical) pairs.

    For each canonical concept found, the Swahili/other surface forms are
    joined with '/', e.g. ("ukungu/baa", "blight"). De-duplicated by
    canonical, capped at SWAHILI_GLOSSARY_MAX_TERMS, in first-seen order.
    """
    seen: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        canonical = _crops.normalize(token) or _terms.normalize(token)
        if canonical and canonical not in seen:
            seen.append(canonical)
        if len(seen) >= SWAHILI_GLOSSARY_MAX_TERMS:
            break

    pairs: list[tuple[str, str]] = []
    for canonical in seen:
        forms = _SURFACE_FORMS.get(canonical, [])
        # Show the non-English-canonical surface forms (the Swahili ones a
        # farmer would type), falling back to the canonical if that's all
        # there is.
        others = [f for f in forms if f.lower() != canonical.lower()]
        display = "/".join(others) if others else canonical
        pairs.append((display, canonical))
    return pairs


def build_glossary(text: str) -> str:
    """
    A prompt glossary block for the domain terms in ``text``.

    Returns "" when the question contains no known domain terms, so the
    caller can skip injecting anything.
    """
    pairs = relevant_terms(text)
    if not pairs:
        return ""
    lines = [f"- {forms} = {canonical}" for forms, canonical in pairs]
    return _GLOSSARY_HEADER + "\n" + "\n".join(lines)
