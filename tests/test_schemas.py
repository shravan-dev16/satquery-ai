"""Unit tests for Pydantic data schemas and contracts.

Verifies schema creation, serialization, validation, and adherence to
the Standard Result Contract (Rule 8).
"""

import pytest
from pydantic import ValidationError

from backend.agent.schema import (
    BoundingBox,
    ComplementarityReport,
    ConfidenceBreakdown,
    EvidenceBundle,
    EvidenceImage,
    EvidenceMask,
    ExecutionTrace,
    ImageMetadata,
    ModalityType,
    ModelCapability,
    ModelExecutionRecord,
    PairCompatibility,
    SpecialistInput,
    SpecialistOutput,
    StandardResultContract,
    StepStatus,
    TaskType,
    TraceStep,
    ValidationResult,
    ZonalStatistic,
)


def test_confidence_breakdown_label():
    """Verify confidence breakdown is labeled as a heuristic, not a calibrated probability."""
    cb = ConfidenceBreakdown(
        input_quality_score=0.95,
        spatial_alignment_score=0.90,
        model_confidence_score=0.85,
        consistency_penalty=0.0,
    )
    assert cb.heuristic_name == "evidence-weighted confidence heuristic"
    assert cb.is_calibrated_probability is False
    assert 0.0 <= cb.input_quality_score <= 1.0


def test_confidence_breakdown_validation():
    """Verify out-of-range confidence scores raise validation errors."""
    with pytest.raises(ValidationError):
        ConfidenceBreakdown(
            input_quality_score=1.5,  # Invalid: > 1.0
            spatial_alignment_score=0.9,
            model_confidence_score=0.8,
            consistency_penalty=0.0,
        )


def test_standard_result_contract_roundtrip():
    """Verify StandardResultContract serialization and deserialization."""
    contract = StandardResultContract(
        task="bitemporal_change_analysis",
        status="success",
        answer="Significant new construction detected in northern sector.",
        confidence=0.88,
        confidence_breakdown=ConfidenceBreakdown(
            input_quality_score=0.95,
            spatial_alignment_score=0.92,
            model_confidence_score=0.89,
            consistency_penalty=0.0,
        ),
        evidence=EvidenceBundle(
            images=[
                EvidenceImage(
                    role="before_image",
                    url="/static/t1.png",
                    width=256,
                    height=256,
                    crs="EPSG:32643",
                    bounds=(725000.0, 3127440.0, 727560.0, 3130000.0),
                )
            ],
            masks=[
                EvidenceMask(
                    mask_id="mask_001",
                    label="Changed Area",
                    url="/static/mask.png",
                    palette={"0": "#00000000", "1": "#FF0033CC"},
                )
            ],
            boxes=[
                BoundingBox(
                    box_id="box_1",
                    label="Industrial Building",
                    confidence=0.94,
                    coordinates_normalized=(0.35, 0.35, 0.65, 0.65),
                    coordinates_pixel=(90, 90, 170, 170),
                    geojson={
                        "type": "Polygon",
                        "coordinates": [[[726000.0, 3128000.0], [727000.0, 3128000.0], [727000.0, 3127000.0], [726000.0, 3128000.0]]],
                    },
                )
            ],
            statistics=[
                ZonalStatistic(
                    metric_name="changed_area_m2",
                    display_name="Changed Area",
                    value=64000.0,
                    unit="m^2",
                )
            ],
        ),
        models=[
            ModelExecutionRecord(
                identifier="CHANGE_DETECT",
                model_name="TinyCD",
                version="1.0.0",
                execution_time_ms=120,
            )
        ],
        parameters={"threshold": 0.5},
        warnings=[],
        execution_trace=[
            TraceStep(
                step_number=1,
                step_name="Validation",
                status=StepStatus.COMPLETED,
                details="Input verified",
                duration_ms=15,
            )
        ],
        execution_time_ms=135,
    )

    json_str = contract.model_dump_json()
    reconstructed = StandardResultContract.model_validate_json(json_str)

    assert reconstructed.task == "bitemporal_change_analysis"
    assert reconstructed.confidence == 0.88
    assert len(reconstructed.evidence.boxes) == 1
    assert reconstructed.evidence.boxes[0].confidence == 0.94
    assert reconstructed.evidence.statistics[0].value == 64000.0


def test_validation_result_schema():
    """Verify ValidationResult structure."""
    val = ValidationResult(
        valid=True,
        images=[
            ImageMetadata(
                filename="optical_s2_sample.tif",
                format="GeoTIFF",
                width=256,
                height=256,
                band_count=3,
                crs="EPSG:32643",
                resolution=(10.0, 10.0),
                bounds=(725000.0, 3127440.0, 727560.0, 3130000.0),
                nodata=0.0,
                dtype="uint16",
                modality=ModalityType.OPTICAL,
            )
        ],
        compatibility=PairCompatibility(
            pair_supported=True,
            spatial_overlap_ratio=1.0,
            spatial_overlap_percentage=100.0,
            crs_match=True,
            reprojection_required=False,
            temporal_ordering="valid",
        ),
        errors=[],
        warnings=[],
    )
    assert val.valid is True
    assert val.images[0].band_count == 3
    assert val.compatibility.spatial_overlap_percentage == 100.0


def test_specialist_input_output():
    """Verify SpecialistInput and SpecialistOutput models."""
    inp = SpecialistInput(
        task=TaskType.VQA,
        query="Is there a port?",
        primary_image_path="/path/to/img.tif",
        parameters={"max_tokens": 128},
    )
    assert inp.task == TaskType.VQA
    assert inp.query == "Is there a port?"

    out = SpecialistOutput(
        specialist_id="RS_VQA",
        success=True,
        answer_text="No port infrastructure visible.",
        confidence=0.91,
    )
    assert out.specialist_id == "RS_VQA"
    assert out.success is True
    assert out.confidence == 0.91
