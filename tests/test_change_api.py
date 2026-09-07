"""Integration tests for bi-temporal change detection via FastAPI endpoints."""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_api_validate_bitemporal_pair():
    """Verify POST /api/v1/validate with two valid GeoTIFFs returns pair compatibility."""
    t1_p = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
    t2_p = Path("tests/fixtures/bitemporal/time2_post_change.tif")

    with open(t1_p, "rb") as f1, open(t2_p, "rb") as f2:
        res = client.post(
            "/api/v1/validate",
            files={
                "image_primary": ("t1.tif", f1, "image/tiff"),
                "image_secondary": ("t2.tif", f2, "image/tiff"),
            },
        )

    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True
    assert len(data["images"]) == 2
    assert data["compatibility"] is not None
    assert data["compatibility"]["spatial_overlap_percentage"] == 100.0


def test_api_analyze_bitemporal_pipeline():
    """Verify POST /api/v1/analyze executes 9-step bi-temporal workflow and returns StandardResultContract."""
    t1_p = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
    t2_p = Path("tests/fixtures/bitemporal/time2_post_change.tif")

    with open(t1_p, "rb") as f1, open(t2_p, "rb") as f2:
        res = client.post(
            "/api/v1/analyze",
            data={"query": "What changed between these two dates?"},
            files={
                "image_primary": ("time1.tif", f1, "image/tiff"),
                "image_secondary": ("time2.tif", f2, "image/tiff"),
            },
        )

    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "bitemporal_change_detection"
    assert data["status"] == "success"
    assert data["confidence"] > 0.5
    assert data["evidence"]["masks"]
    assert data["evidence"]["boxes"]

    # Verify observable 9-step execution trace
    step_names = [s["step_name"] for s in data["execution_trace"]]
    assert step_names == [
        "InputValidation",
        "TemporalValidation",
        "SpatialCompatibility",
        "Alignment",
        "SpecialistSelection",
        "ModelExecution",
        "ChangeMaskValidation",
        "Statistics",
        "EvidenceAssembly",
    ]


def test_api_missing_secondary_image_when_change_query():
    """Verify error 400 when change query is submitted with only one image."""
    t1_p = Path("tests/fixtures/bitemporal/time1_pre_change.tif")

    with open(t1_p, "rb") as f1:
        res = client.post(
            "/api/v1/analyze",
            data={"query": "What changed between these two dates?"},
            files={"image_primary": ("time1.tif", f1, "image/tiff")},
        )

    assert res.status_code == 400
    assert "requires both" in res.json()["detail"]
