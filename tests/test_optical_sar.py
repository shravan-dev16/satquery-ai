"""Unit and Integration Tests for Milestone M6: Optical + SAR Joint Analysis.

Tests:
A. Valid optical + SAR pair execution
B. Reversed modality ordering (SAR primary, Optical secondary)
C. Two optical images rejected
D. Two SAR images rejected
E. Missing CRS rejected
F. Incompatible spatial coverage rejected
G. Alignment behavior across grids and channels
H. Query-dependent output (query asking for water vs built-up)
I. Joint fusion genuinely depends on BOTH modalities (SAR sensitivity & Optical sensitivity)
J. Evidence is returned (complementarity report, detected regions, bounding boxes, visual composite, masks)
K. Trace is returned (all 10 observable trace steps)
L. API end-to-end execution via FastAPI TestClient
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.agent.registry import registry
from backend.agent.schema import SpecialistInput, TaskType
from backend.evaluation.optical_sar_fixtures import generate_optical_sar_fixtures
from backend.main import app
from backend.models.optical_sar import OpticalSARSpecialist
from backend.preprocessing.alignment import CrossModalAligner, CrossModalValidator


@pytest.fixture(scope="module")
def fixtures():
    return generate_optical_sar_fixtures()


@pytest.fixture(scope="module")
def specialist():
    return OpticalSARSpecialist()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test: Specialist Registry Registration (AGENTS.md Rule 5)
# ---------------------------------------------------------------------------
def test_optical_sar_registry_registration():
    spec = registry.get("OPTICAL_SAR_FUSION")
    assert spec is not None
    assert spec.capability.identifier == "OPTICAL_SAR_FUSION"
    assert spec.capability.task == TaskType.OPTICAL_SAR_ANALYSIS
    assert spec.health_check() is True


# ---------------------------------------------------------------------------
# Test A: Valid Optical + SAR Pair Execution
# ---------------------------------------------------------------------------
def test_valid_optical_sar_pair_execution(fixtures, specialist):
    c = fixtures["standard_multimodal"]
    inputs = SpecialistInput(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query=c.query,
        primary_image_path=str(c.optical_path),
        secondary_image_path=str(c.sar_path),
    )
    out = specialist.predict(inputs)
    assert out.success is True
    assert out.specialist_id == "OPTICAL_SAR_FUSION"
    assert out.confidence >= 0.60
    assert "built structure" in out.answer_text.lower() or "water body" in out.answer_text.lower()
    assert len(out.evidence.regions) > 0
    assert len(out.evidence.boxes) > 0


# ---------------------------------------------------------------------------
# Test B: Reversed Modality Ordering (SAR Primary, Optical Secondary)
# ---------------------------------------------------------------------------
def test_reversed_modality_ordering(fixtures, specialist):
    c = fixtures["standard_multimodal"]
    # Provide SAR as primary, Optical as secondary
    inputs = SpecialistInput(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query=c.query,
        primary_image_path=str(c.sar_path),
        secondary_image_path=str(c.optical_path),
    )
    out = specialist.predict(inputs)
    assert out.success is True
    assert out.parameters_used["is_reversed_input_order"] is True
    assert any("Reversed input ordering detected" in w for w in out.warnings)


# ---------------------------------------------------------------------------
# Test C & D: Rejection of Dual Optical and Dual SAR Pairs
# ---------------------------------------------------------------------------
def test_two_optical_images_rejected(fixtures):
    c = fixtures["standard_multimodal"]
    val = CrossModalValidator.validate_pair(c.optical_path, c.optical_path)
    assert val.is_valid is False
    assert "Both images are Optical/Multispectral" in (val.error or "")


def test_two_sar_images_rejected(fixtures):
    c = fixtures["standard_multimodal"]
    val = CrossModalValidator.validate_pair(c.sar_path, c.sar_path)
    assert val.is_valid is False
    assert "Both images are SAR" in (val.error or "")


# ---------------------------------------------------------------------------
# Test E & F: Missing CRS and Incompatible Spatial Coverage Rejection
# ---------------------------------------------------------------------------
def test_missing_crs_rejected(fixtures):
    c_nocrs = fixtures["missing_crs"]
    val = CrossModalValidator.validate_pair(c_nocrs.optical_path, c_nocrs.sar_path)
    assert val.is_valid is False
    assert "lacks a valid Coordinate Reference System" in (val.error or "")


def test_disjoint_spatial_coverage_rejected(fixtures):
    c_disjoint = fixtures["disjoint_pair"]
    val = CrossModalValidator.validate_pair(c_disjoint.optical_path, c_disjoint.sar_path)
    assert val.is_valid is False
    assert "zero spatial intersection" in (val.error or "")


# ---------------------------------------------------------------------------
# Test G: Alignment Behavior Preserving Independent Channels
# ---------------------------------------------------------------------------
def test_alignment_channel_preservation(fixtures):
    c = fixtures["standard_multimodal"]
    aligned = CrossModalAligner.align_pair(c.optical_path, c.sar_path)
    assert aligned.optical_array.ndim == 3
    assert aligned.optical_array.shape[0] == 3  # RGB preserved
    assert aligned.sar_array.ndim == 3
    assert aligned.sar_array.shape[0] == 2      # VV, VH preserved
    assert aligned.width == 256
    assert aligned.height == 256
    assert aligned.crs == "EPSG:32643"


# ---------------------------------------------------------------------------
# Test H: Query-Dependent Output
# ---------------------------------------------------------------------------
def test_query_dependent_output(fixtures, specialist):
    c = fixtures["standard_multimodal"]

    # Query focused on water
    out_water = specialist.predict(SpecialistInput(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Identify water-covered regions using both optical and SAR images.",
        primary_image_path=str(c.optical_path),
        secondary_image_path=str(c.sar_path),
    ))
    assert "water body" in out_water.answer_text.lower() or "water-covered" in out_water.answer_text.lower()

    # Query focused on built structures
    out_built = specialist.predict(SpecialistInput(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Identify the built-up areas using both optical and SAR information.",
        primary_image_path=str(c.optical_path),
        secondary_image_path=str(c.sar_path),
    ))
    assert "built-up" in out_built.answer_text.lower() or "built structure" in out_built.answer_text.lower()


# ---------------------------------------------------------------------------
# Test I: True Joint Fusion Depends on BOTH Modalities
# ---------------------------------------------------------------------------
def test_sar_dependency_sensitivity(fixtures, specialist):
    """Verifies that holding optical constant while changing SAR changes the classification."""
    c_built = fixtures["sar_sensitivity_built"]
    out_built = specialist.predict(SpecialistInput(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Identify land-cover classes",
        primary_image_path=str(c_built.optical_path),
        secondary_image_path=str(c_built.sar_path),
    ))

    c_smooth = fixtures["sar_sensitivity_smooth"]
    out_smooth = specialist.predict(SpecialistInput(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Identify land-cover classes",
        primary_image_path=str(c_smooth.optical_path),
        secondary_image_path=str(c_smooth.sar_path),
    ))

    # Optical image is identical across both runs!
    assert c_built.optical_path == c_smooth.optical_path

    # With high SAR double-bounce & roughness -> detects built_structure
    assert out_built.parameters_used["class_distribution"].get("built_structure", 0) > 0
    # With smooth flat SAR backscatter -> 0 built_structure pixels!
    assert out_smooth.parameters_used["class_distribution"].get("built_structure", 0) == 0


def test_optical_dependency_sensitivity(fixtures, specialist):
    """Verifies that holding SAR constant while changing optical changes the classification."""
    c_water = fixtures["optical_sensitivity_water"]
    out_water = specialist.predict(SpecialistInput(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Identify land-cover classes",
        primary_image_path=str(c_water.optical_path),
        secondary_image_path=str(c_water.sar_path),
    ))

    c_veg = fixtures["optical_sensitivity_veg"]
    out_veg = specialist.predict(SpecialistInput(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query="Identify land-cover classes",
        primary_image_path=str(c_veg.optical_path),
        secondary_image_path=str(c_veg.sar_path),
    ))

    # SAR image is identical across both runs!
    assert c_water.sar_path == c_veg.sar_path

    # Dark blue optical -> classifies as water_body
    assert out_water.parameters_used["class_distribution"].get("water_body", 0) > 0
    assert out_water.parameters_used["class_distribution"].get("vegetation_or_cropland", 0) == 0

    # Green optical -> classifies as vegetation_or_cropland
    assert out_veg.parameters_used["class_distribution"].get("vegetation_or_cropland", 0) > 0
    assert out_veg.parameters_used["class_distribution"].get("water_body", 0) == 0


# ---------------------------------------------------------------------------
# Test J: Evidence Returned
# ---------------------------------------------------------------------------
def test_evidence_bundle_completeness(fixtures, specialist):
    c = fixtures["standard_multimodal"]
    out = specialist.predict(SpecialistInput(
        task=TaskType.OPTICAL_SAR_ANALYSIS,
        query=c.query,
        primary_image_path=str(c.optical_path),
        secondary_image_path=str(c.sar_path),
    ))
    ev = out.evidence
    assert ev.complementarity_report is not None
    assert len(ev.complementarity_report.layers) == 3
    assert len(ev.images) >= 1
    assert any(img.role == "semantic_composite" for img in ev.images)
    assert len(ev.masks) >= 1
    assert len(ev.statistics) >= 5
    assert len(ev.regions) >= 1

    # Check that regions have detailed optical, SAR, and joint evidence
    first_reg = ev.regions[0]
    assert "optical_evidence" in first_reg.details
    assert "sar_evidence" in first_reg.details
    assert "joint_evidence" in first_reg.details


# ---------------------------------------------------------------------------
# Test K & L: End-to-End API Execution and 10-Step Trace
# ---------------------------------------------------------------------------
def test_api_optical_sar_10_step_trace_and_contract(fixtures, client):
    c = fixtures["standard_multimodal"]

    with open(c.optical_path, "rb") as f1, open(c.sar_path, "rb") as f2:
        resp = client.post(
            "/api/v1/analyze",
            data={"query": c.query},
            files={
                "image_primary": ("optical_01.tif", f1, "image/tiff"),
                "image_secondary": ("sar_01.tif", f2, "image/tiff"),
            },
        )

    assert resp.status_code == 200
    contract = resp.json()

    assert contract["task"] == "optical_sar_analysis"
    assert contract["status"] == "success"
    assert contract["confidence"] > 0.0
    assert contract["confidence_breakdown"] is not None

    trace = contract["execution_trace"]
    assert len(trace) == 10

    expected_steps = [
        "InputValidation",
        "ModalityValidation",
        "SpatialCompatibility",
        "Alignment",
        "SpecialistSelection",
        "OpticalFeatureExtraction",
        "SARFeatureExtraction",
        "JointFusion",
        "RegionExtraction",
        "EvidenceAssembly",
    ]
    actual_steps = [s["step_name"] for s in trace]
    assert actual_steps == expected_steps
    assert all(s["status"] in ["completed", "warning"] for s in trace)

    # Verify registered model
    assert len(contract["models"]) == 1
    assert contract["models"][0]["identifier"] == "OPTICAL_SAR_FUSION"
