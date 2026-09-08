"""Unit and integration tests for Evidence Normalization and Fusion (Milestone M8).

Adheres to AGENTS.md:
- Rule 1 (Evidence-first remote sensing AI analyst)
- Rule 8 (Standard Result Contract)
- Rule 18 (Evidence provenance preservation)
- Rule 25 (Testing Requirements)
"""

import pytest
from backend.agent.schema import (
    BoundingBox,
    ComplementarityReport,
    DetectedRegion,
    EvidenceBundle,
    EvidenceItem,
    EvidenceMask,
    ModalityType,
    SemanticChangeInterpretation,
    SemanticTransition,
    SpecialistOutput,
    ZonalStatistic,
)
from backend.evidence.fusion import EvidenceFuser, EvidenceNormalizer


def test_evidence_normalizer_vqa():
    """Verify single-image VQA output is normalized into EvidenceItem with provenance."""
    output = SpecialistOutput(
        specialist_id="RS_VQA",
        success=True,
        answer_text="The scene contains an active commercial seaport with container cranes.",
        confidence=0.92,
        evidence=EvidenceBundle(),
        execution_time_ms=85,
    )
    items = EvidenceNormalizer.normalize_vqa(
        output=output,
        query="What infrastructure is visible?",
        metadata={"filename": "seaport.tif", "crs": "EPSG:32618"},
    )
    assert len(items) == 1
    item = items[0]
    assert item.source_specialist == "RS_VQA"
    assert item.evidence_type == "visual_narrative"
    assert item.modality == ModalityType.OPTICAL
    assert "commercial seaport" in item.claim
    assert item.specialist_confidence == 0.92
    assert item.is_uncertain is False
    assert item.provenance["query"] == "What infrastructure is visible?"
    assert item.provenance["filename"] == "seaport.tif"


def test_evidence_normalizer_grounding_with_boxes():
    """Verify grounding bounding boxes are normalized with strict coordinate provenance."""
    output = SpecialistOutput(
        specialist_id="RS_GROUND",
        success=True,
        answer_text="Detected water body",
        confidence=0.88,
        evidence=EvidenceBundle(
            boxes=[
                BoundingBox(
                    box_id="box_water_01",
                    label="water body",
                    confidence=0.88,
                    model_score=0.88,
                    is_degenerate=False,
                    coordinates_normalized=(0.1, 0.1, 0.5, 0.5),
                    coordinates_pixel=(25, 25, 125, 125),
                    geojson={"type": "Polygon", "coordinates": []},
                )
            ]
        ),
        execution_time_ms=120,
    )
    items = EvidenceNormalizer.normalize_grounding(
        output=output,
        query="Where is the water body?",
        metadata={"filename": "coastal.tif", "crs": "EPSG:32643"},
    )
    assert len(items) == 1
    item = items[0]
    assert item.source_specialist == "RS_GROUND"
    assert item.evidence_type == "bounding_box"
    assert item.region_id == "box_water_01"
    assert item.geometry["coordinates_pixel"] == [25, 25, 125, 125]
    assert item.specialist_confidence == 0.88
    assert item.is_uncertain is False
    assert item.provenance["crs"] == "EPSG:32643"


def test_evidence_normalizer_grounding_negative_detection():
    """Verify empty detection produces explicit negative evidence item with uncertainty flag."""
    output = SpecialistOutput(
        specialist_id="RS_GROUND",
        success=True,
        answer_text="No objects detected",
        confidence=0.10,
        evidence=EvidenceBundle(boxes=[]),
        execution_time_ms=60,
    )
    items = EvidenceNormalizer.normalize_grounding(
        output=output,
        query="Where is the airport runway?",
        metadata={"filename": "forest.tif"},
    )
    assert len(items) == 1
    item = items[0]
    assert item.evidence_type == "negative_detection"
    assert item.is_uncertain is True
    assert "No spatial regions matching referring expression" in item.claim


def test_evidence_normalizer_change_detection():
    """Verify change detection mask, zonal statistics, and clusters are normalized."""
    output = SpecialistOutput(
        specialist_id="CHANGE_DETECT",
        success=True,
        answer_text="Change detected: 4,500 pixels",
        confidence=0.91,
        evidence=EvidenceBundle(
            statistics=[
                ZonalStatistic(metric_name="changed_pixels", display_name="Changed Pixels", value=4500, unit="pixels"),
                ZonalStatistic(metric_name="physical_area_ha", display_name="Physical Area", value=4.5, unit="ha"),
            ],
            regions=[
                DetectedRegion(
                    region_name="cluster_001",
                    label="change_cluster",
                    bbox_pixel=(10, 10, 80, 80),
                    confidence=0.91,
                    details={"area_pixels": 4500, "cluster_index": 1},
                )
            ],
        ),
        parameters_used={
            "changed_pixels": 4500,
            "change_ratio_pct": 3.25,
            "total_clusters": 1,
            "width": 512,
            "height": 512,
            "candidate": "tinycd",
        },
        execution_time_ms=95,
    )
    items = EvidenceNormalizer.normalize_change_detection(
        output=output,
        metadata={"filename_t1": "t1.tif", "filename_t2": "t2.tif", "crs": "EPSG:32643"},
    )
    # Expect 1 mask summary + 2 stats + 1 region = 4 items
    assert len(items) == 4
    types = [i.evidence_type for i in items]
    assert "spatial_mask" in types
    assert "zonal_statistics" in types
    assert "spatial_cluster" in types

    mask_item = next(i for i in items if i.evidence_type == "spatial_mask")
    assert mask_item.metrics["changed_pixels"] == 4500
    assert mask_item.provenance["candidate"] == "tinycd"

    stat_item = next(i for i in items if i.metrics.get("metric_name") == "physical_area_ha")
    assert stat_item.metrics["value"] == 4.5
    assert stat_item.provenance["crs"] == "EPSG:32643"


def test_evidence_normalizer_change_vqa():
    """Verify semantic change interpretation and individual transitions are normalized."""
    output = SpecialistOutput(
        specialist_id="CHANGE_VQA",
        success=True,
        confidence=0.85,
        evidence=EvidenceBundle(
            semantic_interpretation=SemanticChangeInterpretation(
                summary="Vegetation clearing converted to residential construction.",
                temporal_direction="increased",
                predominant_transition="forest -> residential_construction",
                transitions=[
                    SemanticTransition(
                        transition_id="trans_01",
                        from_class="forest",
                        to_class="residential_construction",
                        description="New buildings constructed on cleared forested patch.",
                        region_id="cluster_001",
                        bbox_pixel=(10, 10, 80, 80),
                        semantic_confidence=0.85,
                    )
                ],
                supporting_regions=["cluster_001"],
                supporting_model="Qwen/Qwen2-VL-2B-Instruct",
                semantic_uncertainty=0.15,
            )
        ),
        execution_time_ms=450,
    )
    items = EvidenceNormalizer.normalize_change_vqa(
        output=output,
        metadata={"filename_t1": "t1.tif", "filename_t2": "t2.tif"},
    )
    assert len(items) == 2  # 1 summary + 1 transition
    types = [i.evidence_type for i in items]
    assert "semantic_interpretation" in types
    assert "semantic_transition" in types

    trans_item = next(i for i in items if i.evidence_type == "semantic_transition")
    assert trans_item.region_id == "cluster_001"
    assert trans_item.metrics["from_class"] == "forest"
    assert trans_item.metrics["to_class"] == "residential_construction"
    assert trans_item.provenance["transition_id"] == "trans_01"


def test_evidence_normalizer_optical_sar():
    """Verify Optical-SAR joint analysis outputs are normalized with dual-sensor cue breakdown."""
    output = SpecialistOutput(
        specialist_id="OPTICAL_SAR_FUSION",
        success=True,
        answer_text="Joint analysis identified built structures and surface water.",
        confidence=0.89,
        evidence=EvidenceBundle(
            complementarity_report=ComplementarityReport(
                optical_limitations="Cloud shadow obscures southern section.",
                sar_penetration="All-weather SAR penetrates shadows and reveals high double-bounce built structures.",
                structural_contrast="SAR roughness separates smooth pavement from rough buildings.",
                layers=[{"sensor": "Optical", "layer": "RGB"}, {"sensor": "SAR", "layer": "Backscatter"}],
            ),
            regions=[
                DetectedRegion(
                    region_name="fused_reg_001",
                    label="built_structure",
                    bbox_pixel=(50, 50, 150, 150),
                    confidence=0.92,
                    details={
                        "area_pixels": 8500,
                        "optical_evidence": {"mean_brightness": 0.45, "mean_exg": -0.02},
                        "sar_evidence": {"mean_backscatter": 0.78, "roughness": 0.22},
                        "joint_evidence": {"confidence": 0.92},
                    },
                )
            ],
        ),
        parameters_used={"class_distribution": {"built_structure": 60.0, "water_body": 40.0}},
        execution_time_ms=180,
    )
    items = EvidenceNormalizer.normalize_optical_sar(
        output=output,
        metadata={"filename_optical": "opt.tif", "filename_sar": "sar.tif"},
    )
    assert len(items) == 3  # 1 joint class + 1 complementarity + 1 region
    types = [i.evidence_type for i in items]
    assert "cross_modal_joint_class" in types
    assert "cross_modal_complementarity" in types
    assert "cross_modal_region" in types

    reg_item = next(i for i in items if i.evidence_type == "cross_modal_region")
    assert reg_item.metrics["optical_evidence"]["mean_brightness"] == 0.45
    assert reg_item.metrics["sar_evidence"]["mean_backscatter"] == 0.78
    assert reg_item.region_id == "fused_reg_001"


def test_evidence_fuser_deduplication_and_indexing():
    """Verify EvidenceFuser appends, deduplicates, and enables specialist/region querying."""
    bundle = EvidenceBundle()
    items1 = [
        EvidenceItem(
            item_id="item_001",
            source_specialist="CHANGE_DETECT",
            evidence_type="spatial_mask",
            modality=ModalityType.BITEMPORAL,
            claim="Changed pixels = 2500",
            region_id="cluster_A",
            specialist_confidence=0.95,
        )
    ]
    items2 = [
        EvidenceItem(
            item_id="item_001",  # Duplicate ID
            source_specialist="CHANGE_DETECT",
            evidence_type="spatial_mask",
            modality=ModalityType.BITEMPORAL,
            claim="Changed pixels = 2500 duplicate",
            specialist_confidence=0.95,
        ),
        EvidenceItem(
            item_id="item_002",
            source_specialist="CHANGE_VQA",
            evidence_type="semantic_transition",
            modality=ModalityType.BITEMPORAL,
            claim="bare_land -> built_up",
            region_id="cluster_A",
            specialist_confidence=0.88,
        ),
    ]

    EvidenceFuser.fuse(bundle, items1)
    assert len(bundle.fused_items) == 1

    EvidenceFuser.fuse(bundle, items2)
    # Deduplication should ensure exactly 2 unique items
    assert len(bundle.fused_items) == 2

    by_spec = EvidenceFuser.get_items_by_specialist(bundle, "CHANGE_DETECT")
    assert len(by_spec) == 1
    assert by_spec[0].item_id == "item_001"

    by_reg = EvidenceFuser.get_items_by_region(bundle, "cluster_A")
    assert len(by_reg) == 2

    by_type = EvidenceFuser.get_items_by_type(bundle, "semantic_transition")
    assert len(by_type) == 1
    assert by_type[0].item_id == "item_002"
