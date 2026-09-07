"""Comprehensive Unit and Integration Tests for Milestone M5: Change Semantic Interpretation & Change VQA.

Tests:
1. Capability declaration and registry registration.
2. Zero-change defensive handling (anti-hallucination rule).
3. Multi-panel composite evidence generation.
4. Structured JSON parsing and schema adherence.
5. Malformed VLM output handling and fallback uncertainty.
6. Ambiguous / unknown transition handling.
7. End-to-end FastAPI /api/v1/analyze pipeline with 10-step auditable execution trace.
8. Query patterns (directional, land-cover, negative, general description).
9. Spatial provenance linking (region IDs and bounding boxes).
10. Non-regression of existing confidence and evidence contracts.
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.agent.registry import registry
from backend.agent.schema import (
    DetectedRegion,
    ModalityType,
    SemanticChangeInterpretation,
    SemanticTransition,
    SpecialistInput,
    TaskType,
)
from backend.main import app
from backend.models.change_vqa import ChangeVQASpecialist

client = TestClient(app)


def test_change_vqa_capability():
    """Verify declared capability conforms to Rule 5 and TaskType.CHANGE_VQA."""
    specialist = ChangeVQASpecialist()
    cap = specialist.capability

    assert cap.identifier == "CHANGE_VQA"
    assert cap.task == TaskType.CHANGE_VQA
    assert cap.supported_input_count == [2]
    assert ModalityType.BITEMPORAL in cap.supported_modalities
    assert ModalityType.OPTICAL in cap.supported_modalities
    assert cap.confidence_available is True


def test_change_vqa_registry_registration():
    """Verify ChangeVQASpecialist is registered and discoverable via central ModelRegistry."""
    spec = registry.get("CHANGE_VQA")
    assert spec is not None
    assert isinstance(spec, ChangeVQASpecialist)

    # Discovery by task and modality
    matches = registry.find_specialists(task=TaskType.CHANGE_VQA, input_count=2)
    assert len(matches) >= 1
    assert matches[0].capability.identifier == "CHANGE_VQA"


def test_change_vqa_zero_change_no_hallucination():
    """Verify zero-change scene bypasses VLM and emits grounded negative findings (Scientific Honesty)."""
    specialist = ChangeVQASpecialist()
    t1_p = "tests/fixtures/bitemporal/time1_pre_change.tif"
    t2_p = "tests/fixtures/bitemporal/time2_post_change.tif"

    inputs = SpecialistInput(
        task=TaskType.CHANGE_VQA,
        query="What changed between these two images?",
        primary_image_path=t1_p,
        secondary_image_path=t2_p,
        parameters={
            "changed_pixels": 0,
            "change_ratio_pct": 0.0,
            "total_clusters": 0,
            "regions": [],
        },
    )

    output = specialist.predict(inputs)

    assert output.success is True
    assert output.confidence >= 0.90
    assert "No significant physical surface change" in output.answer_text
    assert output.evidence.semantic_interpretation is not None

    interp = output.evidence.semantic_interpretation
    assert interp.temporal_direction == "no_change"
    assert interp.predominant_transition == "none"
    assert len(interp.transitions) == 0
    assert interp.semantic_uncertainty == 0.0
    assert output.parameters_used["bypassed_vlm_for_zero_change"] is True


def test_change_vqa_multi_region_conditioning():
    """Verify ChangeVQASpecialist conditions on detected spatial regions and builds composite evidence."""
    specialist = ChangeVQASpecialist()
    t1_p = "tests/fixtures/bitemporal/time1_pre_change.tif"
    t2_p = "tests/fixtures/bitemporal/time2_post_change.tif"

    regions = [
        {
            "region_name": "change_cluster_001",
            "bbox_pixel": (50, 50, 150, 150),
            "details": {"area_pixels": 2500, "area_m2": 25000.0, "cluster_id": 1},
        },
        {
            "region_name": "change_cluster_002",
            "bbox_pixel": (180, 180, 220, 220),
            "details": {"area_pixels": 400, "area_m2": 4000.0, "cluster_id": 2},
        },
    ]

    inputs = SpecialistInput(
        task=TaskType.CHANGE_VQA,
        query="Describe what changed in the scene.",
        primary_image_path=t1_p,
        secondary_image_path=t2_p,
        parameters={
            "changed_pixels": 2900,
            "change_ratio_pct": 5.8,
            "area_m2": 29000.0,
            "area_ha": 2.9,
            "total_clusters": 2,
            "regions": regions,
        },
    )

    output = specialist.predict(inputs)
    assert output.success is True
    assert output.evidence.semantic_interpretation is not None

    interp = output.evidence.semantic_interpretation
    assert len(interp.transitions) >= 1
    assert interp.supporting_regions == ["change_cluster_001"]
    assert any(img.role == "semantic_composite" for img in output.evidence.images)


def test_change_vqa_structured_json_parsing():
    """Verify _parse_and_validate_output correctly extracts transitions from JSON string."""
    specialist = ChangeVQASpecialist()

    raw_json = json.dumps({
        "summary": "Development expanded in the eastern sector, with 3 new buildings constructed.",
        "temporal_direction": "increased",
        "predominant_transition": "bare_soil -> built_up",
        "transitions": [
            {
                "from_class": "bare_soil",
                "to_class": "built_up",
                "description": "Construction of new residential buildings.",
                "region_id": "change_cluster_001",
                "confidence": 0.89,
                "is_uncertain": False,
            }
        ],
        "warnings": [],
    })

    result = specialist._parse_and_validate_output(
        raw_text=raw_json,
        primary_region_id="change_cluster_001",
        primary_bbox=(10, 20, 100, 120),
        query="What changed?",
        total_clusters=1,
        changed_pixels=1500,
        area_context="15,000.0 m²",
        change_ratio_pct=3.0,
    )

    assert result.temporal_direction == "increased"
    assert result.predominant_transition == "bare_soil -> built_up"
    assert len(result.transitions) == 1
    assert result.transitions[0].from_class == "bare_soil"
    assert result.transitions[0].to_class == "built_up"
    assert result.transitions[0].region_id == "change_cluster_001"
    assert result.transitions[0].bbox_pixel == (10, 20, 100, 120)
    assert result.transitions[0].semantic_confidence == 0.89
    assert result.transitions[0].is_uncertain is False
    assert result.semantic_uncertainty == 0.11


def test_change_vqa_malformed_output_fallback():
    """Verify malformed/unstructured VLM text triggers defensive fallback uncertainty without crashing."""
    specialist = ChangeVQASpecialist()

    raw_malformed = "I cannot generate valid JSON, but I see some new roofs and roads built between the two dates."

    result = specialist._parse_and_validate_output(
        raw_text=raw_malformed,
        primary_region_id="change_cluster_001",
        primary_bbox=(10, 20, 100, 120),
        query="What changed?",
        total_clusters=1,
        changed_pixels=800,
        area_context="8,000.0 m²",
        change_ratio_pct=1.5,
    )

    assert result.temporal_direction == "uncertain"
    assert result.predominant_transition == "unknown -> unknown"
    assert len(result.transitions) == 1
    assert result.transitions[0].from_class == "unknown"
    assert result.transitions[0].to_class == "unknown"
    assert result.transitions[0].is_uncertain is True
    assert result.semantic_uncertainty >= 0.50
    assert len(result.warnings) >= 1
    assert "unstructured" in result.warnings[0].lower()


def test_change_vqa_unknown_classes_defensively_flagged():
    """Verify that when class transitions are 'unknown', is_uncertain is flagged defensively."""
    specialist = ChangeVQASpecialist()

    raw_json = json.dumps({
        "summary": "Spectral change detected, but image resolution is too low to classify the surface.",
        "temporal_direction": "modified",
        "predominant_transition": "unknown -> unknown",
        "transitions": [
            {
                "from_class": "unknown",
                "to_class": "unknown",
                "description": "Surface texture change without identifiable structure.",
                "region_id": "change_cluster_001",
                "confidence": 0.50,
                "is_uncertain": False,
            }
        ],
        "warnings": ["Low resolution prevents class assignment."],
    })

    result = specialist._parse_and_validate_output(
        raw_text=raw_json,
        primary_region_id="change_cluster_001",
        primary_bbox=None,
        query="What changed?",
        total_clusters=1,
        changed_pixels=200,
        area_context="2,000.0 m²",
        change_ratio_pct=0.4,
    )

    # is_uncertain must be enforced as True because classes are unknown
    assert result.transitions[0].is_uncertain is True
    assert result.semantic_uncertainty == 0.50


def test_api_change_vqa_10_step_trace_and_contract():
    """Verify POST /api/v1/analyze with semantic query executes 10-step trace and returns StandardResultContract."""
    t1_p = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
    t2_p = Path("tests/fixtures/bitemporal/time2_post_change.tif")

    with open(t1_p, "rb") as f1, open(t2_p, "rb") as f2:
        res = client.post(
            "/api/v1/analyze",
            data={
                "query": "Describe what type of land-cover change occurred between these dates.",
                "task_hint": "change_vqa",
            },
            files={
                "image_primary": ("t1.tif", f1, "image/tiff"),
                "image_secondary": ("t2.tif", f2, "image/tiff"),
            },
        )

    assert res.status_code == 200
    data = res.json()

    assert data["task"] == "bitemporal_change_vqa"
    assert data["status"] == "success"
    assert data["confidence"] > 0.50

    # Evidence verification
    evidence = data["evidence"]
    assert evidence["masks"]
    assert evidence["boxes"]
    assert evidence["statistics"]
    assert evidence["regions"]
    assert evidence["semantic_interpretation"] is not None

    sem = evidence["semantic_interpretation"]
    assert sem["summary"]
    assert sem["temporal_direction"] in ["increased", "decreased", "modified", "no_change", "uncertain"]
    assert len(sem["transitions"]) >= 1
    assert "semantic_uncertainty" in sem

    # Auditable 10-Step Execution Trace verification (Rule 12 & Phase 14)
    trace = data["execution_trace"]
    step_names = [s["step_name"] for s in trace]
    assert step_names == [
        "InputValidation",
        "TemporalValidation",
        "SpatialCompatibility",
        "Alignment",
        "SpecialistSelection",
        "ModelExecution",
        "ChangeMaskValidation",
        "Statistics",
        "SemanticInterpretation",
        "EvidenceAssembly",
    ]

    # Verify SemanticInterpretation step metadata
    sem_step = trace[8]
    assert sem_step["step_name"] == "SemanticInterpretation"
    assert sem_step["status"] == "completed"
    assert "temporal_direction" in sem_step["metadata"]

    # Verify models record has both TinyCD and ChangeVQASpecialist
    model_ids = [m["identifier"] for m in data["models"]]
    assert "CHANGE_DETECT" in model_ids
    assert "CHANGE_VQA" in model_ids


def test_api_change_vqa_query_patterns():
    """Verify different semantic query patterns are classified and executed correctly."""
    t1_p = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
    t2_p = Path("tests/fixtures/bitemporal/time2_post_change.tif")

    queries = [
        "Has the built-up area increased, decreased, or remained unchanged?",
        "Describe the main change and where it occurred.",
    ]

    for q in queries:
        with open(t1_p, "rb") as f1, open(t2_p, "rb") as f2:
            res = client.post(
                "/api/v1/analyze",
                data={"query": q},
                files={
                    "image_primary": ("t1.tif", f1, "image/tiff"),
                    "image_secondary": ("t2.tif", f2, "image/tiff"),
                },
            )
        assert res.status_code == 200
        data = res.json()
        assert data["task"] == "bitemporal_change_vqa"
        assert data["evidence"]["semantic_interpretation"] is not None
        assert data["evidence"]["semantic_interpretation"]["temporal_direction"] in ["increased", "decreased", "modified"]


def test_change_vqa_missing_secondary_error():
    """Verify ChangeVQASpecialist returns error if secondary image is missing."""
    specialist = ChangeVQASpecialist()
    t1_p = "tests/fixtures/bitemporal/time1_pre_change.tif"

    inputs = SpecialistInput(
        task=TaskType.CHANGE_VQA,
        query="What changed?",
        primary_image_path=t1_p,
        secondary_image_path=None,
    )

    output = specialist.predict(inputs)
    assert output.success is False
    assert "requires both" in output.answer_text
