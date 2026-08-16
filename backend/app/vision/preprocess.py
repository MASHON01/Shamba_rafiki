"""
Image preprocessing for the classifier.

Turns an uploaded photo (a file path or raw bytes) into the exact
float tensor MobileNetV3-small expects: a normalized 1x3x224x224
NCHW array. Pillow + numpy only, no OpenCV on the hot path, because
this runs on the 8 GB kiosk and every megabyte of resident library
matters. The steps mirror the training transform exactly so that
inference and training preprocessing never disagree.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.config.constants import (
    CLASSIFIER_INPUT_SIZE,
    CLASSIFIER_NORM_MEAN,
    CLASSIFIER_NORM_STD,
)

_MEAN = np.array(CLASSIFIER_NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(CLASSIFIER_NORM_STD, dtype=np.float32).reshape(3, 1, 1)


def load_image(source: str | Path | bytes) -> Image.Image:
    """Open an image from a path or raw bytes as RGB."""
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path)
        else:
            img = Image.open(BytesIO(source))
        img = img.convert("RGB")
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not decode image: {exc}") from exc
    return img


def _resize_and_center_crop(img: Image.Image, size: int) -> Image.Image:
    width, height = img.size
    if width == 0 or height == 0:
        raise ValueError("Image has a zero dimension.")
    scale = size / min(width, height)
    new_w = max(size, round(width * scale))
    new_h = max(size, round(height * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess(source: str | Path | bytes, size: int = CLASSIFIER_INPUT_SIZE) -> np.ndarray:
    """Load, resize/center-crop, normalize, and return a (1,3,size,size) float32 batch."""
    img = load_image(source)
    img = _resize_and_center_crop(img, size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - _MEAN) / _STD
    return arr[np.newaxis, ...].astype(np.float32)
