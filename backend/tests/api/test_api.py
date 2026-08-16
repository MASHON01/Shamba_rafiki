"""API tests: every endpoint through FastAPI's TestClient.

Uses the api_client fixture (real app, fake LLM + real retriever via
dependency overrides). Confirms routing, the standard response
envelope, validation errors, and endpoint behaviour over real HTTP."""

from __future__ import annotations

import io

import pytest

pytestmark = [pytest.mark.api, pytest.mark.slow]


# --- health / meta --------------------------------------------------------

def test_root(api_client):
    r = api_client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] and body["data"]["status"] == "running"


def test_version(api_client):
    r = api_client.get("/version")
    assert r.status_code == 200 and r.json()["data"]["version"]


def test_health_reports_llm(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["data"]["llm_available"] is True


# --- chat -----------------------------------------------------------------

def test_chat_grounded(api_client):
    r = api_client.post("/chat", json={
        "query": "How do I treat maize blight in Nakuru?",
        "language": "en", "session_id": "farmer1"})
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["answer"]
    assert d["grounded"] is True
    assert len(d["sources"]) >= 1
    assert "confidence" in d and "verification_action" in d


def test_chat_swahili(api_client):
    r = api_client.post("/chat", json={
        "query": "Mahindi yangu yana ukungu, nifanye nini?",
        "language": "sw"})
    assert r.status_code == 200
    assert r.json()["data"]["language"] == "sw"


def test_chat_empty_query_422(api_client):
    r = api_client.post("/chat", json={"query": ""})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"


def test_chat_missing_query_422(api_client):
    r = api_client.post("/chat", json={"language": "en"})
    assert r.status_code == 422


def test_chat_request_id_header(api_client):
    r = api_client.post("/chat", json={"query": "maize blight"},
                        headers={"X-Request-ID": "trace-abc"})
    assert r.headers.get("X-Request-ID") == "trace-abc"


# --- retrieve -------------------------------------------------------------

def test_retrieve(api_client):
    r = api_client.post("/retrieve", json={"query": "maize blight fungicide", "top_k": 3})
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["index_loaded"] is True
    assert d["count"] >= 1
    assert d["results"][0]["crop"] == "maize"


def test_retrieve_empty_query_422(api_client):
    r = api_client.post("/retrieve", json={"query": ""})
    assert r.status_code == 422


def test_retrieve_top_k_bounds(api_client):
    r = api_client.post("/retrieve", json={"query": "maize", "top_k": 999})
    assert r.status_code == 422  # exceeds le=20


# --- upload ---------------------------------------------------------------

def test_upload_saves(api_client):
    files = {"file": ("guide.txt", io.BytesIO(b"Bean rust guide."), "text/plain")}
    r = api_client.post("/upload", files=files)
    assert r.status_code == 200
    d = r.json()["data"]
    assert d["indexed"] is False
    assert d["saved_as"].endswith(".txt")


def test_upload_unsupported_type_400(api_client):
    files = {"file": ("x.exe", io.BytesIO(b"nope"), "application/octet-stream")}
    r = api_client.post("/upload", files=files)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UnsupportedFormatError"


def test_upload_empty_file_400(api_client):
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    r = api_client.post("/upload", files=files)
    assert r.status_code == 400


# --- docs -----------------------------------------------------------------

def test_openapi_lists_all_routes(api_client):
    schema = api_client.get("/openapi.json").json()
    paths = set(schema["paths"].keys())
    for expected in ("/", "/health", "/version", "/chat", "/retrieve", "/upload"):
        assert expected in paths