"""
ONNX image-classifier engine.

Loads a MobileNetV3-small model exported to ONNX and runs it on CPU
via onnxruntime. No PyTorch at serve time. The engine is lazy and
defensive: onnxruntime is imported only when a prediction is first
attempted, and if the model file is absent the engine reports
`available = False` rather than raising at startup, so a kiosk with no
classifier is degraded, not broken.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config.constants import CLASSIFIER_TOP_K
from app.config.settings import settings
from app.models.classifier import ClassificationResult, ConfidenceLevel, Prediction
from app.utils.logger import get_logger
from app.vision.labels import ClassLabel, ClassLabels, confidence_band
from app.vision.preprocess import preprocess

logger = get_logger("Classifier")


class ClassifierUnavailableError(RuntimeError):
    """Raised when a prediction is attempted but no model is loaded."""


@dataclass(slots=True)
class VisionResult:
    """Engine output: the API result plus the resolved best label."""

    result: ClassificationResult
    best_label: ClassLabel
    ranked_labels: list[tuple[ClassLabel, float]]

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return self.result.confidence_level

    @property
    def is_confident(self) -> bool:
        """Usable enough to steer retrieval (not a LOW-band guess)."""
        return self.result.confidence_level is not ConfidenceLevel.LOW


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / np.sum(exp)


class ImageClassifier:
    """Thread-safe, lazily-loaded ONNX classifier."""

    def __init__(self, model_path: Path | None = None, model_name: str = "mobilenetv3-small") -> None:
        self._model_path = Path(model_path) if model_path else settings.onnx_model_path
        self._model_name = model_name
        self._session = None
        self._input_name: str | None = None
        self._labels: ClassLabels | None = None
        self._lock = threading.Lock()

    @property
    def model_path(self) -> Path:
        return self._model_path

    def is_available(self) -> bool:
        """True if a model file exists to load. Cheap: no import, no load."""
        return self._model_path.is_file()

    def health(self) -> bool:
        """True if the model is present and can actually be loaded."""
        if not self.is_available():
            return False
        try:
            self.load()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("classifier.health_failed", error=str(exc))
            return False

    def load(self) -> None:
        """Load the ONNX session and labels once (idempotent, thread-safe)."""
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            if not self._model_path.is_file():
                raise ClassifierUnavailableError(
                    f"No classifier model at {self._model_path}. Train one with "
                    "training/train_classifier.py, or run text-only."
                )
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover
                raise ClassifierUnavailableError(
                    "onnxruntime is required for image classification (pip install onnxruntime)."
                ) from exc

            session = ort.InferenceSession(str(self._model_path), providers=["CPUExecutionProvider"])
            self._input_name = session.get_inputs()[0].name
            num_classes = int(session.get_outputs()[0].shape[-1])
            self._labels = ClassLabels.load_for_model(self._model_path, num_classes)
            self._session = session
            logger.info(
                "classifier.loaded",
                model=str(self._model_path),
                classes=len(self._labels),
                crops=self._labels.crops,
            )

    def predict(self, source: str | Path | bytes, top_k: int = CLASSIFIER_TOP_K) -> VisionResult:
        """Classify one image. Returns ranked predictions with a confidence band."""
        self.load()
        assert self._session is not None and self._labels is not None

        tensor = preprocess(source)
        start = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: tensor})
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logits = np.asarray(outputs[0]).reshape(-1)
        probs = _softmax(logits)
        k = max(1, min(top_k, probs.shape[0]))
        top_idx = np.argsort(probs)[::-1][:k]

        predictions: list[Prediction] = []
        ranked_labels: list[tuple[ClassLabel, float]] = []
        for idx in top_idx:
            label = self._labels.get(int(idx))
            prob = float(probs[int(idx)])
            predictions.append(
                Prediction(
                    label=label.label,
                    confidence=prob,
                    crop=None if label.crop == "unknown" else label.crop,
                    condition=None if label.condition == "unknown" else label.condition,
                )
            )
            ranked_labels.append((label, prob))

        best_pred = predictions[0]
        best_label = ranked_labels[0][0]
        result = ClassificationResult(
            predictions=predictions,
            best_prediction=best_pred,
            confidence_level=confidence_band(best_pred.confidence),
            inference_time_ms=round(elapsed_ms, 2),
            model_name=self._model_name,
        )
        logger.info(
            "classifier.predicted",
            best=best_label.label,
            confidence=round(best_pred.confidence, 3),
            band=result.confidence_level.value,
            latency_ms=result.inference_time_ms,
        )
        return VisionResult(result=result, best_label=best_label, ranked_labels=ranked_labels)
