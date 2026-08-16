"""
Swahili quality (prompt-side).

The safe, always-on core of Swahili support: a query-specific glossary
and an intent-specific few-shot exemplar, injected into the prompt when a
farmer asks in Kiswahili. No model weights change - this is the
prompt-side capability the build plan calls the safe core, with the LoRA
fine-tune (training/) as optional upside on top.

    build_glossary Swahili domain terms in the query -> glossary block
    fewshot_block one grounded, cited SW exemplar for the intent
    swahili_prompt_enrichment the two combined, ready to append to a prompt
    SWAHILI_EVAL_SET questions that measure whether any of it works
"""

from __future__ import annotations

from app.language.swahili.eval_set import (
    RUBRIC,
    SWAHILI_EVAL_SET,
    SwahiliEvalCase,
    list_cases,
)
from app.language.swahili.fewshot_examples import fewshot_block
from app.language.swahili.glossary_injection import build_glossary, relevant_terms

__all__ = [
    "build_glossary",
    "relevant_terms",
    "fewshot_block",
    "swahili_prompt_enrichment",
    "SWAHILI_EVAL_SET",
    "SwahiliEvalCase",
    "list_cases",
    "RUBRIC",
]


def swahili_prompt_enrichment(question: str, intent: str) -> str:
    """
    The Swahili prompt add-on for one question: a query-specific glossary
    (only the terms present) plus one intent exemplar.

    Returns "" only if there is genuinely nothing to add (there is always
    at least a general exemplar, so in practice this returns the exemplar
    even when no glossary terms match).
    """
    blocks = []
    glossary = build_glossary(question)
    if glossary:
        blocks.append(glossary)
    exemplar = fewshot_block(intent)
    if exemplar:
        blocks.append(exemplar)
    return "\n\n".join(blocks)
