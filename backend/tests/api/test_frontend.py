"""API tests for the static kiosk UI."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.api


def test_ui_index_served(api_client):
    resp = api_client.get("/app/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Shamba Rafiki" in resp.text


def test_ui_is_bilingual_and_links_app_js(api_client):
    html = api_client.get("/app/").text
    assert "app.js" in html
    assert 'data-lang="sw"' in html
    assert "data-sw=" in html


def test_ui_assets_served(api_client):
    css = api_client.get("/app/styles.css")
    js = api_client.get("/app/app.js")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]


def test_root_liveness_still_json(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_api_still_works_alongside_ui(api_client):
    resp = api_client.post("/chat", json={"query": "How do I treat maize blight?", "language": "en"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True
