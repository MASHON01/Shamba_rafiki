"""API tests for /classify and the /health classifier block."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.api

pytest.importorskip("onnxruntime", reason="image path needs onnxruntime")
pytest.importorskip("PIL", reason="image path needs Pillow")


@pytest.fixture
def classify_client(image_orchestrator, retriever):
    from fastapi.testclient import TestClient

    from app.api.app_factory import create_app
    from app.api.dependencies import get_orchestrator, get_retriever

    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: image_orchestrator
    app.dependency_overrides[get_retriever] = lambda: retriever
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_classify_image_only(classify_client, leaf_image):
    with open(leaf_image, "rb") as fh:
        resp = classify_client.post(
            "/classify", files={"file": ("leaf.jpg", fh, "image/jpeg")}, data={"language": "en"}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "classification" in body["data"]
    assert {"answer", "sources", "grounded"} <= body["data"].keys()


def test_classify_image_plus_text(classify_client, leaf_image):
    with open(leaf_image, "rb") as fh:
        resp = classify_client.post(
            "/classify",
            files={"file": ("leaf.png", fh, "image/png")},
            data={"query": "maize blight treatment", "language": "en"},
        )
    assert resp.status_code == 200
    assert resp.json()["data"]["grounded"] is True


def test_classify_rejects_non_image(classify_client):
    resp = classify_client.post("/classify", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_classify_rejects_empty(classify_client):
    resp = classify_client.post("/classify", files={"file": ("leaf.jpg", b"", "image/jpeg")})
    assert resp.status_code == 400


def test_health_reports_classifier(classify_client):
    body = classify_client.get("/health").json()
    assert "classifier" in body["data"]
    assert body["data"]["classifier"]["available"] is True
