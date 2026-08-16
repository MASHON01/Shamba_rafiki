"""
Per-intent generation presets.

The same model should answer a *diagnosis* differently from a *how-to*.
A diagnosis or a price question is a factual lookup - the answer should
be as consistent and grounded as possible, so temperature sits at its
lowest. A how-to explanation benefits from slightly more freedom to
phrase steps naturally, so it gets a small nudge up. Everything else
falls back to the house default.

These are deliberately conservative: even the "explanatory" preset stays
well below creative-writing temperatures. On an advisory kiosk, a wrong
but confident answer is the failure mode we are guarding against, so the
whole band is low.

Intents are the plain strings the language layer already produces
(diagnosis / price / how_to / general - DEFAULT_INTENT). Keeping the
mapping keyed on those strings means this table and the intent extractor
never drift apart.
"""

from __future__ import annotations

from app.config.constants import DEFAULT_INTENT
from app.orchestration.llm.generation_config import GenerationConfig

# Intent -> the one thing that varies between profiles today: how tightly
# the model samples. Other sampling knobs stay at the grounded house
# defaults in GenerationConfig, so there is one place to change them.
_INTENT_TEMPERATURE: dict[str, float] = {
    "diagnosis": 0.20,  # factual: a crop is sick, be consistent.
    "price": 0.20,  # factual: market/cost framing, be consistent.
    "how_to": 0.35,  # explanatory: room to phrase steps naturally.
    "general": 0.40,  # open-ended fallback, still conservative.
}


def _preset_for(temperature: float) -> GenerationConfig:
    return GenerationConfig(temperature=temperature)

    # Prebuilt, reusable profiles (immutable in practice - override returns
    # copies, so sharing an instance per intent is safe).


PRESETS: dict[str, GenerationConfig] = {
    intent: _preset_for(temp) for intent, temp in _INTENT_TEMPERATURE.items()
}


def for_intent(intent: str | None) -> GenerationConfig:
    """
    The generation profile for ``intent``.

    Falls back to the default-intent profile for an unknown or missing
    intent, so a new intent string can never blow up generation - it
    just gets the safe house default until a profile is added here.
    """
    if intent and intent in PRESETS:
        return PRESETS[intent]
    return PRESETS.get(DEFAULT_INTENT, GenerationConfig)
