"""Unit tests for deterministic semantic validation fixtures and spatial vs semantic separation."""

from pathlib import Path
import pytest
import rasterio

from backend.agent.registry import registry
from backend.agent.schema import SpecialistInput, TaskType
from backend.evaluation.semantic_fixtures import generate_semantic_fixtures
from backend.models.change_vqa import ChangeVQASpecialist


def test_semantic_fixture_generation():
    """Verify that generate_semantic_fixtures creates all 5 pairs with valid CRS."""
    fixtures = generate_semantic_fixtures()
    assert len(fixtures) == 5

    categories = [f.category for f in fixtures]
    assert "URBAN" in categories
    assert "FOREST_VEGETATION" in categories
    assert "WATER_AQUATIC" in categories
    assert "AGRICULTURE" in categories
    assert "ZERO_CHANGE" in categories

    for f in fixtures:
        assert f.t1_path.exists()
        assert f.t2_path.exists()
        with rasterio.open(f.t1_path) as src:
            assert src.crs is not None
            assert src.count == 3
            assert src.width == 256
            assert src.height == 256


def test_zero_change_semantic_anti_hallucination():
    """Verify zero-change fixture produces zero transitions and bypasses VLM."""
    fixtures = generate_semantic_fixtures()
    zero_case = next(f for f in fixtures if f.category == "ZERO_CHANGE")

    specialist = ChangeVQASpecialist()
    inp = SpecialistInput(
        task=TaskType.CHANGE_VQA,
        query=zero_case.query,
        primary_image_path=str(zero_case.t1_path),
        secondary_image_path=str(zero_case.t2_path),
        parameters={"changed_pixels": 0, "change_ratio_pct": 0.0, "regions": []},
    )

    out = specialist.predict(inp)
    assert out.success is True
    assert out.parameters_used["bypassed_vlm_for_zero_change"] is True
    sem = out.evidence.semantic_interpretation
    assert sem.temporal_direction == "no_change"
    assert sem.predominant_transition == "none"
    assert len(sem.transitions) == 0
    assert sem.semantic_uncertainty == 0.0


def test_spatial_vs_semantic_separation_concept():
    """Verify that spatial grounding and semantic correctness are distinct fields in results."""
    specialist = ChangeVQASpecialist()
    inp = SpecialistInput(
        task=TaskType.CHANGE_VQA,
        query="What changed?",
        primary_image_path="tests/fixtures/semantic/t1_forest.tif",
        secondary_image_path="tests/fixtures/semantic/t2_forest.tif",
        parameters={
            "changed_pixels": 9102,
            "change_ratio_pct": 13.89,
            "total_clusters": 3,
            "regions": [
                {
                    "region_name": "changed_region_01",
                    "bbox_pixel": (64, 64, 192, 192),
                    "details": {"area_pixels": 9102},
                }
            ],
        },
    )
    out = specialist.predict(inp)
    assert out.evidence.semantic_interpretation is not None
    # Confidence in StandardResultContract is decoupled from semantic_uncertainty
    assert "semantic_uncertainty" in out.parameters_used
    assert out.evidence.semantic_interpretation.semantic_uncertainty is not None
