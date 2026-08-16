"""
System-prompt template registry.

The single place that selects a system prompt by (version, language,
intent) and assembles it from the base persona + the intent suffix. This
is what makes prompt wording a *versioned* asset: you can add a candidate
version, A/B it against the baseline (scripts/compare_prompts.py), and
flip the default only once the eval harness shows it scores
higher - deliberately, not by guesswork.

Selection is total and forgiving, so generation never breaks on an
unexpected value:
  - unknown version -> the default version (settings.PROMPT_VERSION)
  - unknown language -> English
  - unknown intent -> the general (base-only, or general-shape) prompt
"""

from __future__ import annotations

from app.config.constants import DEFAULT_INTENT, DEFAULT_PROMPT_VERSION
from app.config.settings import settings
from app.models.language import LanguageCode
from app.orchestration.prompts.templates import (
    EN_V1_BASE,
    EN_V1_INTENT,
    EN_V2_BASE,
    EN_V2_INTENT,
    SW_V1_BASE,
    SW_V1_INTENT,
    SW_V2_BASE,
    SW_V2_INTENT,
)

# version -> language -> (base_prompt, {intent: suffix})
_REGISTRY: dict[str, dict[LanguageCode, tuple[str, dict[str, str]]]] = {
    "v1": {
        LanguageCode.ENGLISH: (EN_V1_BASE, EN_V1_INTENT),
        LanguageCode.SWAHILI: (SW_V1_BASE, SW_V1_INTENT),
    },
    "v2": {
        LanguageCode.ENGLISH: (EN_V2_BASE, EN_V2_INTENT),
        LanguageCode.SWAHILI: (SW_V2_BASE, SW_V2_INTENT),
    },
}


def list_versions() -> list[str]:
    """All registered prompt versions, e.g. ['v1', 'v2']."""
    return list(_REGISTRY)


def default_version() -> str:
    """
    The configured default version, or the constant fallback if the
    configured one is not registered (so a bad env value can't break
    prompt selection).
    """
    configured = settings.PROMPT_VERSION
    if configured in _REGISTRY:
        return configured
    return DEFAULT_PROMPT_VERSION if DEFAULT_PROMPT_VERSION in _REGISTRY else next(iter(_REGISTRY))


def get_system_prompt(
    language: LanguageCode = LanguageCode.ENGLISH,
    intent: str = DEFAULT_INTENT,
    version: str | None = None,
) -> str:
    """
    Return the system prompt for a version + language + intent.

    ``version`` defaults to the configured default. Unknown version,
    language, or intent all fall back safely (see module docstring).
    """
    version = version if version in _REGISTRY else default_version
    by_language = _REGISTRY[version]

    base, intents = by_language.get(language, by_language[LanguageCode.ENGLISH])
    suffix = intents.get(intent, intents[DEFAULT_INTENT])
    return base + suffix
