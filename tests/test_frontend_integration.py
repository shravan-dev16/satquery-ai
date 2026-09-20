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
    assert "complementarity-card" in res.text
    assert "optical-sar-preview-card" in res.text


def test_ui_static_assets():
    """Verify that CSS and JS assets are served properly."""
    res_css = client.get("/ui/index.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers["content-type"] or "css" in res_css.headers["content-type"]
    assert "complementarity-card" in res_css.text

    res_js = client.get("/ui/app.js")
    assert res_js.status_code == 200
    assert "javascript" in res_js.headers["content-type"] or "application/" in res_js.headers["content-type"]
    assert "SatQueryClient" in res_js.text
    assert "renderSemanticInterpretation" in res_js.text
    assert "complementarityCard" in res_js.text


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
    assert "OPTICAL_SAR_FUSION" in data["specialists_status"]


def test_m13_root_url_browser_experience():
    """Verify that GET / serves the polished M12/M13 HTML to web browsers (Phase 0)."""
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    res = client.get("/", headers=headers)
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "SatQuery AI" in res.text
    assert "Bi-Temporal Imagery Ingestion" in res.text
    assert "demo-sample-select" in res.text
    assert "analyze-btn" in res.text


def test_m13_root_static_assets():
    """Verify that /index.css and /app.js are directly resolvable from root URL."""
    res_css = client.get("/index.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers["content-type"] or "css" in res_css.headers["content-type"]

    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    assert "javascript" in res_js.headers["content-type"] or "application/" in res_js.headers["content-type"]
    assert "SatQueryClient" in res_js.text


def test_m13_docs_and_health_preserved():
    """Verify that /docs and /health remain fully functional alongside root UI."""
    res_docs = client.get("/docs")
    assert res_docs.status_code == 200
    assert "text/html" in res_docs.headers["content-type"]

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"


