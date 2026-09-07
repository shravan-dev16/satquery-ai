"""Integration tests for FastAPI endpoints."""

from starlette.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root status response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "SatQuery AI"
    assert data["status"] == "online"


def test_health_endpoint():
    """Verify health probe response."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "registered_specialists" in data


def test_models_endpoint():
    """Verify models listing response."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "registered_specialists" in data
    assert "hardware" in data
    assert data["hardware"]["device"] == "cuda"
