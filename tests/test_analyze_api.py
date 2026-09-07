"""Integration tests for /api/v1/validate and /api/v1/analyze endpoints."""

from pathlib import Path
from starlette.testclient import TestClient

from backend.agent.registry import registry
from backend.agent.schema import (
    EvidenceBundle,
    ModelCapability,
    SpecialistInput,
    SpecialistOutput,
    TaskType,
    ModalityType,
)
from backend.main import app
from backend.models.base import BaseSpecialist
from backend.models.vqa import RemoteSensingVQASpecialist

client = TestClient(app)


class FastMockVQASpecialist(BaseSpecialist):
    """Fast mock VQA specialist to verify analyze endpoint pipeline quickly."""

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            identifier="RS_VQA",
            name="Fast Mock RS-VQA",
            version="1.0.0",
            task=TaskType.VQA,
            supported_modalities=[ModalityType.OPTICAL],
            supported_input_count=[1],
            requires_gpu=False,
            vram_budget_mb=0,
            confidence_available=True,
        )

    def load(self) -> None:
        self._is_loaded = True

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_id="RS_VQA",
            success=True,
            answer_text="The area contains mixed vegetation and urban structures.",
            confidence=0.92,
            evidence=EvidenceBundle(),
            parameters_used=inputs.parameters,
            execution_time_ms=42,
        )

    def health_check(self) -> bool:
        return True


def test_api_validate_valid_geotiff(optical_sample_path: Path):
    """Verify /api/v1/validate succeeds on valid GeoTIFF."""
    with open(optical_sample_path, "rb") as f:
        response = client.post(
            "/api/v1/validate",
            files={"image_primary": ("optical_s2_sample.tif", f, "image/tiff")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert len(data["images"]) == 1
    assert data["images"][0]["width"] == 256
    assert data["images"][0]["crs"] == "EPSG:32643"


def test_api_validate_corrupted_file(fixtures_dir: Path):
    """Verify /api/v1/validate returns HTTP 400 on corrupted file."""
    corrupt = fixtures_dir / "invalid" / "corrupted_header.tif"
    with open(corrupt, "rb") as f:
        response = client.post(
            "/api/v1/validate",
            files={"image_primary": ("corrupted_header.tif", f, "image/tiff")},
        )
    assert response.status_code == 400
    data = response.json()
    assert data["valid"] is False
    assert len(data["errors"]) >= 1


def test_api_analyze_validation_failure(fixtures_dir: Path):
    """Verify /api/v1/analyze rejects invalid files with HTTP 400 before model execution."""
    corrupt = fixtures_dir / "invalid" / "corrupted_header.tif"
    with open(corrupt, "rb") as f:
        response = client.post(
            "/api/v1/analyze",
            data={"query": "What is in this image?"},
            files={"image_primary": ("corrupted_header.tif", f, "image/tiff")},
        )
    assert response.status_code == 400
    assert "Input validation failed" in response.json()["detail"]


def test_api_analyze_pipeline_success(optical_sample_path: Path):
    """Verify /api/v1/analyze executes full pipeline and returns StandardResultContract."""
    # Temporarily register fast mock to test route execution
    original_spec = registry.get("RS_VQA")
    registry.register(FastMockVQASpecialist())

    try:
        with open(optical_sample_path, "rb") as f:
            response = client.post(
                "/api/v1/analyze",
                data={"query": "Identify the predominant land cover."},
                files={"image_primary": ("optical_s2_sample.tif", f, "image/tiff")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "single_image_vqa"
        assert data["status"] == "success"
        assert "vegetation" in data["answer"]
        assert 0.0 < data["confidence"] <= 1.0
        assert data["confidence_breakdown"]["heuristic_name"] == "evidence-weighted confidence heuristic"
        assert data["confidence_breakdown"]["is_calibrated_probability"] is False
        assert len(data["execution_trace"]) >= 4

        # Check trace steps
        step_names = [s["step_name"] for s in data["execution_trace"]]
        assert "InputValidation" in step_names
        assert "TaskRouting" in step_names
        assert "SpecialistSelection" in step_names
        assert "ModelExecution" in step_names
        assert "EvidenceAssembly" in step_names

    finally:
        # Restore original specialist
        if original_spec:
            registry.register(original_spec)


class FastMockGroundingSpecialist(BaseSpecialist):
    """Fast mock grounding specialist to verify analyze endpoint pipeline quickly."""

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            identifier="RS_GROUND",
            name="Fast Mock RS-Grounding",
            version="1.0.0",
            task=TaskType.GROUNDING,
            supported_modalities=[ModalityType.OPTICAL],
            supported_input_count=[1],
            requires_gpu=False,
            vram_budget_mb=0,
            confidence_available=True,
        )

    def load(self) -> None:
        self._is_loaded = True

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        from backend.agent.schema import BoundingBox, DetectedRegion
        return SpecialistOutput(
            specialist_id="RS_GROUND",
            success=True,
            answer_text=f"Detected 1 region(s) matching '{inputs.query}'.",
            confidence=0.88,
            evidence=EvidenceBundle(
                boxes=[
                    BoundingBox(
                        box_id="box_001",
                        label="water body",
                        confidence=0.88,
                        coordinates_normalized=(0.1, 0.1, 0.5, 0.5),
                        coordinates_pixel=(25, 25, 128, 128),
                        geojson={
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [725100.0, 3129800.0],
                                    [725500.0, 3129800.0],
                                    [725500.0, 3129200.0],
                                    [725100.0, 3129200.0],
                                    [725100.0, 3129800.0],
                                ]
                            ],
                        },
                    )
                ],
                regions=[
                    DetectedRegion(
                        region_name="water body",
                        label="water body",
                        bbox_pixel=(25.0, 25.0, 128.0, 128.0),
                        confidence=0.88,
                        polygon_pixel=None,
                        details={"bbox_pixel": [25.0, 25.0, 128.0, 128.0]},
                    )
                ],
            ),
            parameters_used=inputs.parameters,
            execution_time_ms=35,
        )

    def health_check(self) -> bool:
        return True


def test_api_analyze_grounding_pipeline_success(optical_sample_path: Path):
    """Verify /api/v1/analyze routes grounding queries to RS_GROUND and returns spatial evidence."""
    original_spec = registry.get("RS_GROUND")
    registry.register(FastMockGroundingSpecialist())

    try:
        with open(optical_sample_path, "rb") as f:
            response = client.post(
                "/api/v1/analyze",
                data={"query": "Where is the water body?"},
                files={"image_primary": ("optical_s2_sample.tif", f, "image/tiff")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["task"] == "single_image_grounding"
        assert data["status"] == "success"
        assert len(data["evidence"]["boxes"]) == 1

        box = data["evidence"]["boxes"][0]
        assert box["label"] == "water body"
        assert box["coordinates_pixel"] == [25, 25, 128, 128]
        assert box["geojson"]["type"] == "Polygon"

        # Verify regions contract
        assert len(data["evidence"]["regions"]) == 1
        region = data["evidence"]["regions"][0]
        assert region["bbox_pixel"] == [25.0, 25.0, 128.0, 128.0]
        assert region["polygon_pixel"] is None

        # Verify execution trace
        step_names = [s["step_name"] for s in data["execution_trace"]]
        assert "InputValidation" in step_names
        assert "TaskRouting" in step_names
        assert "SpecialistSelection" in step_names
        assert "ModelExecution" in step_names
        assert "EvidenceAssembly" in step_names

    finally:
        if original_spec:
            registry.register(original_spec)


import pytest


@pytest.mark.smoke
def test_api_analyze_real_sample_live(fixtures_dir: Path):
    """Live smoke test executing real VQA specialist on real satellite GeoTIFF through /api/v1/analyze."""
    real_sample = fixtures_dir / "real_rs_sample.tif"
    if not real_sample.exists():
        pytest.skip("Real remote-sensing sample not present")

    # Ensure real VQA specialist is registered
    vqa_spec = registry.get("RS_VQA")
    if not vqa_spec:
        vqa_spec = RemoteSensingVQASpecialist()
        registry.register(vqa_spec)

    with open(real_sample, "rb") as f:
        response = client.post(
            "/api/v1/analyze",
            data={"query": "Is there a water body or coastline visible?"},
            files={"image_primary": ("real_rs_sample.tif", f, "image/tiff")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "single_image_vqa"
    assert data["status"] == "success"
    assert len(data["answer"]) > 10
    assert data["confidence_breakdown"]["heuristic_name"] == "evidence-weighted confidence heuristic"
    assert len(data["execution_trace"]) >= 4
