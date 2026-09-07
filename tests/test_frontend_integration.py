"""Tests for SatQuery AI Frontend Integration (M3 Frontend Vertical Slice).

Verifies that the frontend static UI is correctly mounted at `/ui` and served by
the FastAPI application alongside all existing `/api/v1/...` routes.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_ui_index_html():
    """Verify that GET /ui and /ui/ return HTTP 200 with HTML content."""
    res = client.get("/ui/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "SatQuery AI" in res.text
    assert "Bi-Temporal Imagery Ingestion" in res.text
    assert "t1-file-input" in res.text
    assert "t2-file-input" in res.text
    assert "analyze-btn" in res.text
    assert "split-slider-frame" in res.text
    assert "semantic-card" in res.text
    assert "SemanticInterpretation" in res.text


def test_ui_static_assets():
    """Verify that CSS and JS assets are served properly."""
    res_css = client.get("/ui/index.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers["content-type"] or "css" in res_css.headers["content-type"]

    res_js = client.get("/ui/app.js")
    assert res_js.status_code == 200
    assert "javascript" in res_js.headers["content-type"] or "application/" in res_js.headers["content-type"]
    assert "SatQueryClient" in res_js.text
    assert "renderSemanticInterpretation" in res_js.text


def test_root_endpoint_preserved():
    """Verify that GET / still returns JSON for existing API contracts."""
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/json"
    data = res.json()
    assert data["app"] == "SatQuery AI"
    assert data["status"] == "online"


def test_api_health_endpoint():
    """Verify that /api/v1/health is accessible by the UI."""
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "CHANGE_DETECT" in data["specialists_status"]
    assert "CHANGE_VQA" in data["specialists_status"]
