"""
Query understanding extractors.

    EntityExtractor  what the query is about  (crop / county / pest)
    IntentExtractor  what the query wants     (diagnosis/price/how-to)

Entities shape retrieval filtering and prompt specifics; intent
shapes which system prompt and response format the orchestrator
selects.
"""

from __future__ import annotations

from app.language.extractors.entities import EntityExtractor
from app.language.extractors.intent import IntentExtractor

__all__ = [
    "EntityExtractor",
    "IntentExtractor",
]