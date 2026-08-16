"""
Language-related models.

These models standardize language detection,
normalization, translation, and entity extraction.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LanguageCode(str, Enum):
    """
    Supported application languages.
    """

    ENGLISH = "en"
    SWAHILI = "sw"
    UNKNOWN = "unknown"


class LanguageDetectionResult(BaseModel):
    """
    Result returned by the language detector.
    """

    language: LanguageCode

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class TranslationResult(BaseModel):
    """
    Translation output.
    """

    source_language: LanguageCode

    target_language: LanguageCode

    original_text: str

    translated_text: str


class AgriculturalEntity(BaseModel):
    """
    Extracted agricultural entity.
    """

    entity: str

    entity_type: str

    normalized_value: str | None = None

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )


class LanguageAnalysis(BaseModel):
    """
    Complete language processing output.
    """

    detected_language: LanguageDetectionResult

    normalized_query: str

    entities: list[AgriculturalEntity] = Field(
        default_factory=list
    )

    translated: bool = False