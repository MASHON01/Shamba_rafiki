"""
Image classifier models. Shared by the vision module, orchestrator, and API.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    """Confidence categories."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImageClassificationRequest(BaseModel):
    """Request sent to the image classifier."""

    image_path: Path
    crop: str | None = None
    language: str = "en"


class Prediction(BaseModel):
    """Single classifier prediction."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)

    # Populated from the label sidecar. Optional so a model shipped
    # without a labels.json still yields valid predictions.
    crop: str | None = None
    condition: str | None = None


class ClassificationResult(BaseModel):
    """Final classifier output."""

    predictions: list[Prediction]
    best_prediction: Prediction
    confidence_level: ConfidenceLevel
    inference_time_ms: float
    model_name: str
