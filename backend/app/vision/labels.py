"""
Class-label map for the classifier.

An ONNX model outputs a bare vector of logits. The mapping from index
to a named class lives here, in a small JSON sidecar written next to
the model by the training script (plant_classifier.labels.json), so
the labels and the weights never drift apart. Each class carries the
crop, the condition, and the raw training label; the (crop, condition)
pair is folded into retrieval and the prompt. A missing or mismatched
sidecar degrades to generic labels rather than crashing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config.constants import (
    CLASSIFIER_CONFIDENCE_HIGH,
    CLASSIFIER_CONFIDENCE_MEDIUM,
    CLASSIFIER_HEALTHY_CONDITION,
    CLASSIFIER_LABELS_SUFFIX,
    METADATA_UNKNOWN,
)
from app.models.classifier import ConfidenceLevel
from app.utils.logger import get_logger

logger = get_logger("ClassLabels")


@dataclass(frozen=True, slots=True)
class ClassLabel:
    """One output class of the classifier."""

    crop: str
    condition: str
    label: str

    @property
    def is_healthy(self) -> bool:
        return self.condition.strip().lower() == CLASSIFIER_HEALTHY_CONDITION

    def retrieval_hint(self) -> str:
        """Text folded into the retrieval query for this class."""
        if self.is_healthy or self.condition == METADATA_UNKNOWN:
            return self.crop
        return f"{self.crop} {self.condition}".strip()


class ClassLabels:
    """Ordered list of ClassLabel, indexable by model output position."""

    def __init__(self, labels: list[ClassLabel]) -> None:
        if not labels:
            raise ValueError("ClassLabels needs at least one class.")
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def get(self, index: int) -> ClassLabel:
        if 0 <= index < len(self._labels):
            return self._labels[index]
        logger.warning("labels.index_out_of_range", index=index, size=len(self._labels))
        return ClassLabel(crop=METADATA_UNKNOWN, condition=METADATA_UNKNOWN, label=f"class_{index}")

    @property
    def crops(self) -> list[str]:
        return sorted({lbl.crop for lbl in self._labels})

    @classmethod
    def sidecar_path(cls, model_path: Path) -> Path:
        return model_path.with_name(model_path.stem + CLASSIFIER_LABELS_SUFFIX)

    @classmethod
    def from_json(cls, path: Path) -> "ClassLabels":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            items = [raw[k] for k in sorted(raw, key=lambda x: int(x))]
        else:
            items = list(raw)
        labels = [
            ClassLabel(
                crop=str(item.get("crop", METADATA_UNKNOWN)),
                condition=str(item.get("condition", METADATA_UNKNOWN)),
                label=str(item.get("label", item.get("condition", "class"))),
            )
            for item in items
        ]
        return cls(labels)

    @classmethod
    def load_for_model(cls, model_path: Path, num_classes: int) -> "ClassLabels":
        sidecar = cls.sidecar_path(model_path)
        try:
            labels = cls.from_json(sidecar)
        except FileNotFoundError:
            logger.warning("labels.sidecar_missing", path=str(sidecar))
            return cls.generic(num_classes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("labels.sidecar_unreadable", path=str(sidecar), error=str(exc))
            return cls.generic(num_classes)

        if len(labels) != num_classes:
            logger.warning("labels.count_mismatch", sidecar=len(labels), model_outputs=num_classes)
            return cls.generic(num_classes)
        return labels

    @classmethod
    def generic(cls, num_classes: int) -> "ClassLabels":
        return cls(
            [
                ClassLabel(crop=METADATA_UNKNOWN, condition=METADATA_UNKNOWN, label=f"class_{i}")
                for i in range(max(1, num_classes))
            ]
        )


def confidence_band(probability: float) -> ConfidenceLevel:
    """Map a softmax probability to a HIGH/MEDIUM/LOW band."""
    if probability >= CLASSIFIER_CONFIDENCE_HIGH:
        return ConfidenceLevel.HIGH
    if probability >= CLASSIFIER_CONFIDENCE_MEDIUM:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW
