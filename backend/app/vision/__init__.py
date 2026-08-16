"""
Vision module: image disease classification.

A farmer photographs a diseased leaf, MobileNetV3-small (ONNX, CPU)
names the likely crop and disease, and that label is folded into the
RAG retrieval so the advice is grounded in the corpus.
"""

from __future__ import annotations

from app.vision.engine import (
    ClassifierUnavailableError,
    ImageClassifier,
    VisionResult,
)
from app.vision.labels import ClassLabel, ClassLabels, confidence_band
from app.vision.preprocess import preprocess

__all__ = [
    "ImageClassifier",
    "VisionResult",
    "ClassifierUnavailableError",
    "ClassLabel",
    "ClassLabels",
    "confidence_band",
    "preprocess",
]
