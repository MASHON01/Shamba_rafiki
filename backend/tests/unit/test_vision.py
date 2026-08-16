"""Unit tests for the vision module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("onnxruntime", reason="image path needs onnxruntime")
pytest.importorskip("PIL", reason="image path needs Pillow")

from app.config.constants import (  # noqa: E402
    CLASSIFIER_CONFIDENCE_HIGH,
    CLASSIFIER_CONFIDENCE_MEDIUM,
    CLASSIFIER_INPUT_SIZE,
)
from app.models.classifier import ConfidenceLevel  # noqa: E402
from app.vision import (  # noqa: E402
    ClassifierUnavailableError,
    ClassLabel,
    ClassLabels,
    ImageClassifier,
    confidence_band,
    preprocess,
)


def test_preprocess_shape_and_dtype(leaf_image: Path):
    tensor = preprocess(leaf_image)
    assert tensor.shape == (1, 3, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)
    assert tensor.dtype == np.float32


def test_preprocess_normalizes(leaf_image: Path):
    tensor = preprocess(leaf_image)
    assert tensor.min() > -5.0 and tensor.max() < 5.0


def test_preprocess_accepts_bytes(leaf_image: Path):
    data = leaf_image.read_bytes()
    assert preprocess(data).shape == (1, 3, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE)


def test_preprocess_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        preprocess(tmp_path / "nope.jpg")


def test_preprocess_bad_bytes_raises():
    with pytest.raises(ValueError):
        preprocess(b"this is not an image")


def test_retrieval_hint_diseased_and_healthy():
    diseased = ClassLabel(crop="maize", condition="Common Rust", label="x")
    healthy = ClassLabel(crop="tomato", condition="healthy", label="y")
    assert diseased.retrieval_hint() == "maize Common Rust"
    assert healthy.retrieval_hint() == "tomato"
    assert healthy.is_healthy and not diseased.is_healthy


def test_labels_generic_fallback_on_missing(tmp_path: Path):
    labels = ClassLabels.load_for_model(tmp_path / "absent.onnx", num_classes=3)
    assert len(labels) == 3
    assert labels.get(0).label == "class_0"


def test_labels_count_mismatch_falls_back(tiny_classifier_path: Path):
    labels = ClassLabels.load_for_model(tiny_classifier_path, num_classes=2)
    assert len(labels) == 2
    assert labels.get(0).label == "class_0"


def test_confidence_bands():
    assert confidence_band(CLASSIFIER_CONFIDENCE_HIGH) is ConfidenceLevel.HIGH
    assert confidence_band(CLASSIFIER_CONFIDENCE_MEDIUM) is ConfidenceLevel.MEDIUM
    assert confidence_band(CLASSIFIER_CONFIDENCE_MEDIUM - 0.01) is ConfidenceLevel.LOW


def test_missing_model_is_unavailable(tmp_path: Path):
    clf = ImageClassifier(model_path=tmp_path / "none.onnx")
    assert clf.is_available() is False
    assert clf.health() is False
    with pytest.raises(ClassifierUnavailableError):
        clf.predict(b"irrelevant")


def test_predict_returns_ranked_result(image_classifier: ImageClassifier, leaf_image: Path):
    assert image_classifier.is_available() is True
    out = image_classifier.predict(leaf_image)
    probs = [p.confidence for p in out.result.predictions]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert out.result.best_prediction is out.result.predictions[0]
    assert out.result.confidence_level in set(ConfidenceLevel)


def test_top_k_is_capped(image_classifier: ImageClassifier, leaf_image: Path):
    out = image_classifier.predict(leaf_image, top_k=10)
    assert len(out.result.predictions) == 4
