"""Integration tests for the image -> classify -> retrieve -> ground path."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("onnxruntime", reason="image path needs onnxruntime")
pytest.importorskip("PIL", reason="image path needs Pillow")

from app.models.request import ImageRequest  # noqa: E402


def test_image_plus_text_grounds_on_corpus(image_orchestrator, leaf_image: Path):
    resp = image_orchestrator.handle_image_query(
        ImageRequest(image_path=str(leaf_image), query="maize blight treatment", language="en")
    )
    assert resp["success"] is True
    assert resp["data"]["grounded"] is True
    assert resp["data"]["sources"]
    assert "classification" in resp["data"]


def test_image_only_synthesizes_question(image_orchestrator, leaf_image: Path):
    resp = image_orchestrator.handle_image_query(
        ImageRequest(image_path=str(leaf_image), language="en")
    )
    assert resp["success"] is True
    assert resp["data"]["answer"]
    assert resp["data"]["classification"]["confidence_level"] in {"low", "medium", "high"}


def test_no_model_no_text_is_clean_error(retriever, fake_llm, tmp_path, leaf_image: Path):
    from app.orchestration.dispatcher import Dispatcher
    from app.orchestration.orchestrator import Orchestrator
    from app.vision import ImageClassifier

    orch = Orchestrator(
        llm_client=fake_llm,
        dispatcher=Dispatcher(llm_client=fake_llm, retriever=retriever),
        classifier=ImageClassifier(model_path=tmp_path / "absent.onnx"),
    )
    resp = orch.handle_image_query(ImageRequest(image_path=str(leaf_image), language="en"))
    assert resp["success"] is False
    assert resp["error"]["code"] == "CLASSIFIER_UNAVAILABLE"


def test_bad_image_returns_invalid_image(image_orchestrator, tmp_path):
    bad = tmp_path / "not_image.jpg"
    bad.write_text("definitely not an image")
    resp = image_orchestrator.handle_image_query(ImageRequest(image_path=str(bad), language="en"))
    assert resp["success"] is False
    assert resp["error"]["code"] == "INVALID_IMAGE"
