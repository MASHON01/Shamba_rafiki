"""
System prompts.

The instructions that tell the model who it is and how to answer,
selected by language (English / Swahili) and query intent
(diagnosis / price / how-to / general). Per the build plan, prompt
wording drives the accuracy score more than almost anything else, so
this file is deliberately isolated: you can iterate on phrasing here
with real test questions without touching any orchestration logic.

Design choices baked into the prompts:
- Grounding first. Every prompt instructs the model to answer from
  the provided reference material and to say plainly when the
  material doesn't cover the question, rather than inventing facts.
  This is the prompt-side half of hallucination control (the
  Output 7 verifier is the other half).
- Actionable shape. Diagnosis and how-to answers are asked for as
  concrete, ordered steps a smallholder can act on, with cost/market
  framing last so the answer reads as a decision aid.
- Language fidelity. The Swahili prompts instruct the model to
  answer in clear Kiswahili. English remains the primary evaluation
  language (per the product notes); Swahili is a real, tested path.

`get_system_prompt(language, intent)` returns the best-matching
template, falling back to the general prompt for unknown intents and
to English for unknown languages.
"""

from __future__ import annotations

from app.config.constants import DEFAULT_INTENT
from app.models.language import LanguageCode

# --- English -------------------------------------------------------------

_EN_BASE = (
    "You are Farm Pal, an offline farming advisor for smallholder "
    "farmers in Kenya. Answer using ONLY the reference material provided "
    "below. If the material does not cover the question, say so clearly "
    "and give only general, widely-accepted guidance - never invent "
    "specific figures, product names, or citations. Keep answers short, "
    "practical, and easy to act on."
)

_EN_INTENT = {
    "diagnosis": (
        " The farmer is describing a crop problem. Identify the most likely "
        "cause from the reference material, then give clear numbered steps to "
        "manage it. End with a short note on whether treatment is worth the "
        "cost versus replanting."
    ),
    "price": (
        " The farmer is asking about market prices or whether a crop is worth "
        "selling. Use the reference material's price and market information. "
        "Be explicit that prices change and are approximate."
    ),
    "how_to": (
        " The farmer wants to know how or when to do something. Give clear, "
        "ordered steps suited to a smallholder, in the correct sequence."
    ),
    "general": "",
}

# --- Swahili --------------------------------------------------------------

_SW_BASE = (
    "Wewe ni Farm Pal, mshauri wa kilimo asiyetumia mtandao kwa "
    "wakulima wadogo nchini Kenya. Jibu kwa kutumia TU maelezo ya rejea "
    "yaliyotolewa hapa chini. Kama maelezo hayana jibu, sema wazi na utoe "
    "ushauri wa jumla tu unaokubalika - usibuni kamwe takwimu, majina ya "
    "bidhaa, au marejeo. Jibu kwa Kiswahili wazi, kifupi na cha vitendo."
)

_SW_INTENT = {
    "diagnosis": (
        " Mkulima anaeleza tatizo la zao. Tambua chanzo kinachowezekana zaidi "
        "kutoka kwenye maelezo ya rejea, kisha toa hatua zilizo na namba za "
        "kudhibiti. Malizia kwa dokezo fupi kama matibabu yanafaa kulingana na "
        "gharama dhidi ya kupanda upya."
    ),
    "price": (
        " Mkulima anauliza kuhusu bei za soko au kama zao linafaa kuuzwa. "
        "Tumia taarifa za bei na soko kutoka kwenye maelezo ya rejea. Eleza "
        "wazi kwamba bei hubadilika na ni za makadirio."
    ),
    "how_to": (
        " Mkulima anataka kujua jinsi au wakati wa kufanya jambo. Toa hatua "
        "zilizopangwa vizuri zinazofaa mkulima mdogo, kwa mpangilio sahihi."
    ),
    "general": "",
}

_TEMPLATES = {
    LanguageCode.ENGLISH: (_EN_BASE, _EN_INTENT),
    LanguageCode.SWAHILI: (_SW_BASE, _SW_INTENT),
}


def get_system_prompt(
    language: LanguageCode = LanguageCode.ENGLISH,
    intent: str = DEFAULT_INTENT,
) -> str:
    """
    Return the system prompt for a language + intent.

    Unknown language falls back to English; unknown intent falls back
    to the general (base-only) prompt.
    """
    base, intents = _TEMPLATES.get(language, _TEMPLATES[LanguageCode.ENGLISH])
    suffix = intents.get(intent, intents[DEFAULT_INTENT])
    return base + suffix