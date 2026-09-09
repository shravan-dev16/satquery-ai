"""Comprehensive Unit and Integration Test Suite for Milestone M11 Reporting.

Verifies:
1. Single-image VQA report packaging
2. Grounding report packaging (bounding boxes & counts)
3. Bi-temporal change report packaging (metrics, masks, overlays)
4. Optical-SAR joint analysis report packaging (complementarity)
5. Combined multi-specialist workflow packaging
6. Confidence packaging (heuristic score, not calibrated probability)
7. Consistency packaging (status, conflicts, gating action)
8. Evidence provenance ledger
9. Visual artifact references
10. Auditable execution trace
11. Contradictory evidence handling (conflict visibility, gating)
12. Insufficient evidence handling (no forced certainty)
13. Missing optional evidence robustness
14. Specialist failure & partial output handling
15. Malformed contract fallback safety
16. JSON serialization & round-trip validation
17. Markdown export generation & formatting
18. End-to-end API integration (/api/v1/analyze returns report)
19. Report determinism and structural integrity
"""

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.agent.schema import (
    BoundingBox,
    ConfidenceBreakdown,
    ConfidenceLevel,
    ConsistencyReport,
    ConsistencyStatus,
    DetectedRegion,
    EvidenceBundle,
    EvidenceConflict,
    EvidenceImage,
    EvidenceItem,
    EvidenceMask,
    ModalityType,
    ModelCapability,
    ModelExecutionRecord,
    SpecialistInput,
    SpecialistOutput,
    StandardResultContract,
    StepStatus,
    TaskType,
    TraceStep,
    ZonalStatistic,
)
from backend.main import app
from backend.models.base import BaseSpecialist
from backend.reports import (
    AnalystReport,
    ReportBuilder,
    generate_markdown_report,
)
from backend.agent.registry import registry

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Fixtures & Mock Helpers
# ---------------------------------------------------------------------------

def create_sample_contract(
    task: str = "single_image_vqa",
    answer: str = "Mixed vegetation and low-density residential structures observed.",
    confidence: float = 0.88,
    confidence_level: str = "HIGH",
    evidence: EvidenceBundle = None,
    evidence_status: str = "CONSISTENT",
    models: list = None,
    parameters: dict = None,
    warnings: list = None,
    trace_steps: list = None,
) -> StandardResultContract:
    """Helper to build a realistic StandardResultContract."""
    if evidence is None:
        evidence = EvidenceBundle()
    if models is None:
        models = [
            ModelExecutionRecord(
                identifier="RS_VQA",
                model_name="Qwen2-VL-2B-Instruct-RS-LoRA",
                version="1.0.0",
                execution_time_ms=120,
            )
        ]
    if parameters is None:
        parameters = {"query": "What is in this scene?"}
    if warnings is None:
        warnings = []
    if trace_steps is None:
        trace_steps = [
            TraceStep(step_number=1, step_name="InputValidation", status=StepStatus.COMPLETED, details="Input validated", duration_ms=10),
            TraceStep(step_number=2, step_name="ModelExecution", status=StepStatus.COMPLETED, details="Model executed", duration_ms=120),
            TraceStep(step_number=3, step_name="EvidenceAssembly", status=StepStatus.COMPLETED, details="Evidence assembled", duration_ms=5),
        ]

    cb = ConfidenceBreakdown(
        overall_confidence=confidence,
        specialist_confidence=confidence,
        input_quality_score=0.95,
        spatial_alignment_score=1.0,
        model_confidence_score=confidence,
        confidence_level=ConfidenceLevel(confidence_level) if confidence_level in ConfidenceLevel._value2member_map_ else ConfidenceLevel.MEDIUM,
        confidence_factors=["High resolution input", "Strong specialist agreement"],
        confidence_warnings=[],
        is_calibrated_probability=False,
    )

    return StandardResultContract(
        task=task,
        status="success",
        answer=answer,
        confidence=confidence,
        confidence_level=confidence_level,
        confidence_breakdown=cb,
        evidence=evidence,
        evidence_status=evidence_status,
        models=models,
        parameters=parameters,
        warnings=warnings,
        execution_trace=trace_steps,
        execution_time_ms=135,
    )


# ---------------------------------------------------------------------------
# 1. Single-Image VQA Report
# ---------------------------------------------------------------------------

def test_single_image_vqa_report():
    """Verify single-image VQA produces a coherent AnalystReport."""
    contract = create_sample_contract()
    report = ReportBuilder.build(contract=contract, query="What is in this scene?", primary_filename="sample_optical.tif")

    assert isinstance(report, AnalystReport)
    assert report.task == "single_image_vqa"
    assert report.metadata.report_id.startswith("REP_")
    assert report.metadata.mission == "Smart India Hackathon 2026 / SIH26167"
    assert report.answer == contract.answer
    assert report.input_summary.query == "What is in this scene?"
    assert report.input_summary.input_count >= 1
    assert report.confidence.score == 0.88
    assert report.confidence.level == "HIGH"
    assert report.confidence.is_calibrated_probability is False
    assert len(report.execution_trace) == 3
    assert len(report.provenance) >= 1
    assert report.provenance[0].source_specialist == "RS_VQA"


# ---------------------------------------------------------------------------
# 2. Grounding Report
# ---------------------------------------------------------------------------

def test_grounding_report():
    """Verify text-guided grounding packages bounding box metrics and visual links."""
    boxes = [
        BoundingBox(
            box_id="box_001",
            label="water_body",
            confidence=0.91,
            coordinates_normalized=(0.1, 0.1, 0.4, 0.5),
            coordinates_pixel=(10, 10, 100, 120),
            is_degenerate=False,
        ),
        BoundingBox(
            box_id="box_002",
            label="water_body",
            confidence=0.85,
            coordinates_normalized=(0.5, 0.6, 0.8, 0.9),
            coordinates_pixel=(128, 150, 200, 230),
            is_degenerate=False,
        ),
    ]
    ev = EvidenceBundle(
        boxes=boxes,
        images=[EvidenceImage(role="primary", url="/api/v1/static/previews/grounding_preview.png", width=256, height=256, crs="EPSG:32643")],
    )
    contract = create_sample_contract(
        task="single_image_grounding",
        answer="Detected 2 water body features in the target raster.",
        confidence=0.89,
        evidence=ev,
        models=[ModelExecutionRecord(identifier="RS_GROUND", model_name="Grounding-DINO-Base", version="1.0.0", execution_time_ms=250)],
    )

    report = ReportBuilder.build(contract=contract, query="Where is the water body?")

    assert report.task == "single_image_grounding"
    assert report.statistics.metrics["detected_boxes_count"] == 2
    assert report.statistics.metrics["valid_boxes_count"] == 2
    assert report.statistics.metrics["max_box_confidence"] == 0.91
    assert any("valid spatial target" in note for note in report.statistics.summary_notes)
    assert len(report.visual_evidence) >= 1
    assert report.visual_evidence[0].path_or_url == "/api/v1/static/previews/grounding_preview.png"


# ---------------------------------------------------------------------------
# 3. Bi-Temporal Change Report
# ---------------------------------------------------------------------------

def test_bitemporal_change_report():
    """Verify bi-temporal change analysis packages physical metrics, masks, and overlays."""
    ev = EvidenceBundle(
        masks=[
            EvidenceMask(mask_id="mask_cd_001", label="Change Mask", url="/api/v1/static/previews/change_mask_001.png"),
        ],
        statistics=[
            ZonalStatistic(metric_name="changed_pixels", display_name="Changed Pixels", value=1024, unit="pixels"),
            ZonalStatistic(metric_name="physical_area_ha", display_name="Physical Change Area", value=1.024, unit="hectares"),
        ],
    )
    params = {
        "changed_pixels": 1024,
        "physical_area_ha": 1.024,
        "change_ratio": 0.0404,
        "overlay_path": "/api/v1/static/previews/change_overlay_001.png",
        "mask_path": "/api/v1/static/previews/change_mask_001.png",
        "temporal_ordering": "2024-01-10 -> 2024-06-15",
    }
    contract = create_sample_contract(
        task="bitemporal_change_detection",
        answer="Physical surface change detected across 1.02 hectares (4.04% of scene).",
        confidence=0.86,
        evidence=ev,
        parameters=params,
        models=[ModelExecutionRecord(identifier="CHANGE_DETECT", model_name="TinyCD-LEVIR-Adapted", version="1.0.0", execution_time_ms=180)],
    )

    report = ReportBuilder.build(
        contract=contract,
        query="What changed between T1 and T2?",
        primary_filename="time1_pre.tif",
        secondary_filename="time2_post.tif",
    )

    assert report.task == "bitemporal_change_detection"
    assert report.input_summary.input_count == 2
    assert report.input_summary.temporal_ordering == "2024-01-10 -> 2024-06-15"
    assert report.statistics.metrics["changed_pixels"] == 1024
    assert report.statistics.metrics["physical_area_ha"] == 1.024
    assert len(report.visual_evidence) >= 2  # mask and overlay
    urls = [v.path_or_url for v in report.visual_evidence]
    assert "/api/v1/static/previews/change_mask_001.png" in urls
    assert "/api/v1/static/previews/change_overlay_001.png" in urls


# ---------------------------------------------------------------------------
# 4. Optical-SAR Report
# ---------------------------------------------------------------------------

def test_optical_sar_report():
    """Verify optical-SAR joint analysis packages cross-modal complementarity."""
    from backend.agent.schema import ComplementarityReport
    comp = ComplementarityReport(
        optical_limitations="Dense cloud cover obscuring 28% of optical scene.",
        sar_penetration="C-band microwave radar penetrated cloud cover, resolving runway structures.",
        structural_contrast="SAR double-bounce backscatter identified high-density urban clusters.",
        layers=[
            {"layer_id": "sar_penetration", "label": "Cloud Penetration", "url": "/api/v1/static/previews/sar_penetration_001.png"},
            {"layer_id": "sar_roughness", "label": "Microwave Roughness", "url": "/api/v1/static/previews/sar_roughness_001.png"},
        ],
    )
    ev = EvidenceBundle(complementarity_report=comp)
    contract = create_sample_contract(
        task="optical_sar_analysis",
        answer="Joint optical-SAR analysis successfully resolved cloud-obscured infrastructure using microwave penetration.",
        confidence=0.84,
        evidence=ev,
        models=[ModelExecutionRecord(identifier="OPTICAL_SAR_FUSION", model_name="DualModalityPhysicsSpecialist", version="1.0.0", execution_time_ms=210)],
    )

    report = ReportBuilder.build(
        contract=contract,
        query="Analyze optical and SAR together.",
        primary_filename="optical_scene.tif",
        secondary_filename="sar_scene.tif",
    )

    assert report.task == "optical_sar_analysis"
    assert len(report.input_summary.images) == 2
    assert report.input_summary.images[0].role == "optical"
    assert report.input_summary.images[1].role == "sar"
    assert any("Dual-modality synthesis" in note for note in report.statistics.summary_notes)
    assert len(report.visual_evidence) >= 2
    vis_ids = [v.id for v in report.visual_evidence]
    assert "sar_penetration" in vis_ids


# ---------------------------------------------------------------------------
# 5. Combined Multi-Specialist Workflow
# ---------------------------------------------------------------------------

def test_combined_multi_specialist_report():
    """Verify combined workflow (Change Detection + Change VQA + Optical SAR) preserves unified provenance."""
    fused_items = [
        EvidenceItem(
            item_id="ev_cd_001",
            source_specialist="CHANGE_DETECT",
            evidence_type="spatial_mask",
            modality=ModalityType.OPTICAL,
            claim="Surface change detected in northern quad: 520 pixels.",
            metrics={"changed_pixels": 520, "area_ha": 0.52},
            specialist_confidence=0.88,
            provenance={"sensor": "Sentinel-2", "method": "TinyCD"},
        ),
        EvidenceItem(
            item_id="ev_vqa_001",
            source_specialist="CHANGE_VQA",
            evidence_type="semantic_transition",
            modality=ModalityType.OPTICAL,
            claim="Identified urban construction on previously bare land.",
            specialist_confidence=0.82,
            provenance={"vlm": "Qwen2-VL-2B-Instruct-RS-LoRA"},
        ),
    ]
    ev = EvidenceBundle(
        fused_items=fused_items,
        consistency_report=ConsistencyReport(
            status=ConsistencyStatus.CONSISTENT,
            is_gated=False,
            gating_action="allow",
            summary_narrative="Both physical change and semantic interpretation confirm urban expansion.",
        ),
    )
    models = [
        ModelExecutionRecord(identifier="CHANGE_DETECT", model_name="TinyCD", version="1.0.0", execution_time_ms=150),
        ModelExecutionRecord(identifier="CHANGE_VQA", model_name="Qwen2-VL-RS-LoRA", version="1.0.0", execution_time_ms=800),
    ]
    contract = create_sample_contract(
        task="bitemporal_change_vqa",
        answer="Urban expansion confirmed in northern sector: 0.52 ha of bare land converted to structures.",
        confidence=0.85,
        evidence=ev,
        models=models,
    )

    report = ReportBuilder.build(contract=contract, query="Has built-up area increased?")

    assert len(report.provenance) == 2
    specialists_in_provenance = {p.source_specialist for p in report.provenance}
    assert "CHANGE_DETECT" in specialists_in_provenance
    assert "CHANGE_VQA" in specialists_in_provenance
    assert report.consistency.status == "CONSISTENT"
    assert report.consistency.is_gated is False


# ---------------------------------------------------------------------------
# 6. Confidence Packaging (Rule 11)
# ---------------------------------------------------------------------------

def test_confidence_packaging():
    """Verify confidence packaging preserves M9 heuristic nature and strictly denies calibrated probability."""
    contract = create_sample_contract(confidence=0.74, confidence_level="MEDIUM")
    report = ReportBuilder.build(contract=contract)

    assert report.confidence.score == 0.74
    assert report.confidence.level == "MEDIUM"
    assert report.confidence.is_calibrated_probability is False
    assert "heuristic" in report.confidence.method.lower()
    assert report.confidence.breakdown is not None
    assert report.confidence.breakdown.is_calibrated_probability is False


# ---------------------------------------------------------------------------
# 7. Consistency Packaging (M8)
# ---------------------------------------------------------------------------

def test_consistency_packaging():
    """Verify consistency packaging reflects M8 status, gating action, and summary."""
    ev = EvidenceBundle(
        consistency_report=ConsistencyReport(
            status=ConsistencyStatus.PARTIALLY_CONSISTENT,
            is_gated=True,
            gating_action="qualify",
            conflicts_count=1,
            summary_narrative="Minor boundary discrepancy between physical mask and semantic narrative.",
        )
    )
    contract = create_sample_contract(evidence=ev, evidence_status="PARTIALLY_CONSISTENT")
    report = ReportBuilder.build(contract=contract)

    assert report.consistency.status == "PARTIALLY_CONSISTENT"
    assert report.consistency.is_gated is True
    assert report.consistency.gating_action == "qualify"
    assert "boundary discrepancy" in report.consistency.summary_narrative


# ---------------------------------------------------------------------------
# 8. Evidence Provenance Ledger
# ---------------------------------------------------------------------------

def test_evidence_provenance_ledger():
    """Verify every claim in fused items maps to its specialist, inputs, and confidence."""
    item = EvidenceItem(
        item_id="ev_001",
        source_specialist="RS_GROUND",
        evidence_type="bounding_box",
        modality=ModalityType.OPTICAL,
        claim="Detected commercial aircraft at apron coordinate [120, 45, 180, 110].",
        specialist_confidence=0.94,
        provenance={"sensor": "Cartosat-2", "crs": "EPSG:32643"},
    )
    ev = EvidenceBundle(fused_items=[item])
    contract = create_sample_contract(
        task="single_image_grounding",
        evidence=ev,
        models=[ModelExecutionRecord(identifier="RS_GROUND", model_name="Grounding-DINO", version="1.0.0", execution_time_ms=190)],
    )

    report = ReportBuilder.build(contract=contract, primary_filename="cartosat_scene.tif")

    assert len(report.provenance) == 1
    p = report.provenance[0]
    assert p.evidence_id == "ev_001"
    assert p.source_specialist == "RS_GROUND"
    assert p.model_name == "Grounding-DINO"
    assert p.raw_confidence == 0.94
    assert "cartosat_scene.tif" in p.input_files
    assert p.provenance_details["sensor"] == "Cartosat-2"


# ---------------------------------------------------------------------------
# 9. Visual Artifact References
# ---------------------------------------------------------------------------

def test_visual_artifact_references():
    """Verify visual evidence references contain clean paths, labels, and formats."""
    ev = EvidenceBundle(
        images=[EvidenceImage(role="primary", url="/api/v1/static/previews/input_rgb.png", width=256, height=256, crs="EPSG:32643")],
        masks=[EvidenceMask(mask_id="mask_1", label="Water Mask", url="/api/v1/static/previews/water_mask.png", format="image/png")],
    )
    contract = create_sample_contract(evidence=ev)
    report = ReportBuilder.build(contract=contract)

    assert len(report.visual_evidence) == 2
    types = {v.type for v in report.visual_evidence}
    assert "source_image" in types
    assert "mask" in types
    for v in report.visual_evidence:
        assert v.path_or_url.startswith("/api/v1/static/previews/")
        assert v.format == "image/png"


# ---------------------------------------------------------------------------
# 10. Execution Trace Preservation
# ---------------------------------------------------------------------------

def test_execution_trace_preservation():
    """Verify execution trace steps maintain chronological order and timings."""
    contract = create_sample_contract()
    report = ReportBuilder.build(contract=contract)

    assert len(report.execution_trace) == 3
    step_numbers = [s.step_number for s in report.execution_trace]
    assert step_numbers == [1, 2, 3]
    assert report.execution_trace[0].step_name == "InputValidation"
    assert report.execution_trace[1].step_name == "ModelExecution"
    assert report.execution_trace[2].step_name == "EvidenceAssembly"


# ---------------------------------------------------------------------------
# 11. Contradictory Evidence Handling (Rule 10)
# ---------------------------------------------------------------------------

def test_contradictory_evidence_handling():
    """Verify contradictions detected by M8 are transparently exposed with conflicts and gating."""
    conflict = EvidenceConflict(
        conflict_id="conf_001",
        rule_violated="C1_ZERO_CHANGE_CONTRADICTION",
        severity="critical",
        description="Change detector measured 0 pixels changed, but language narrative claims major urban expansion.",
        conflicting_sources=["CHANGE_DETECT", "CHANGE_VQA"],
        details={"detector_pixels": 0, "claimed_expansion": True},
    )
    cr = ConsistencyReport(
        status=ConsistencyStatus.CONTRADICTORY,
        is_gated=True,
        gating_action="flag_contradiction",
        conflicts=[conflict],
        summary_narrative="Significant contradiction detected between physical change detector and semantic interpretation.",
    )
    ev = EvidenceBundle(consistency_report=cr)
    contract = create_sample_contract(
        task="bitemporal_change_vqa",
        answer="[CONTRADICTION DETECTED] Change detector measured 0 pixels, contradicting claimed urban expansion.",
        confidence=0.35,
        confidence_level="LOW",
        evidence=ev,
        evidence_status="CONTRADICTORY",
        warnings=["Consistency violation: C1_ZERO_CHANGE_CONTRADICTION"],
    )

    report = ReportBuilder.build(contract=contract)

    assert report.consistency.status == "CONTRADICTORY"
    assert report.consistency.is_gated is True
    assert report.consistency.gating_action == "flag_contradiction"
    assert report.consistency.conflicts_count == 1
    assert len(report.consistency.conflicts) == 1
    assert report.consistency.conflicts[0].rule_violated == "C1_ZERO_CHANGE_CONTRADICTION"
    assert report.confidence.score <= 0.40
    assert report.confidence.level == "LOW"


# ---------------------------------------------------------------------------
# 12. Insufficient Evidence Handling (Rule 31)
# ---------------------------------------------------------------------------

def test_insufficient_evidence_handling():
    """Verify system does not manufacture certainty when evidence is insufficient."""
    cr = ConsistencyReport(
        status=ConsistencyStatus.INSUFFICIENT_EVIDENCE,
        is_gated=True,
        gating_action="state_insufficient",
        summary_narrative="Insufficient spatial overlap to support valid change analysis.",
    )
    ev = EvidenceBundle(consistency_report=cr)
    contract = create_sample_contract(
        task="bitemporal_change_detection",
        answer="Insufficient valid pixel overlap (22.5%) between images to compute reliable change detection.",
        confidence=0.20,
        confidence_level="UNSUPPORTED",
        evidence=ev,
        evidence_status="INSUFFICIENT_EVIDENCE",
        warnings=["Low spatial overlap: 22.5%"],
    )

    report = ReportBuilder.build(contract=contract)

    assert report.consistency.status == "INSUFFICIENT_EVIDENCE"
    assert report.confidence.level == "UNSUPPORTED"
    assert report.confidence.score <= 0.25
    assert any("Low spatial overlap" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# 13. Missing Optional Evidence Robustness
# ---------------------------------------------------------------------------

def test_missing_optional_evidence_robustness():
    """Verify ReportBuilder handles completely empty EvidenceBundle gracefully."""
    empty_ev = EvidenceBundle(images=[], masks=[], boxes=[], statistics=[], regions=[])
    contract = create_sample_contract(evidence=empty_ev)
    report = ReportBuilder.build(contract=contract)

    assert report.visual_evidence == []
    assert report.statistics.zonal_statistics == []
    assert report.statistics.metrics == {}
    assert report.consistency.status == "CONSISTENT"


# ---------------------------------------------------------------------------
# 14. Specialist Failure & Partial Output Handling
# ---------------------------------------------------------------------------

def test_specialist_failure_handling():
    """Verify partial status in contract is safely reflected in the report."""
    contract = create_sample_contract(
        task="single_image_caption",
        answer="Scene contains terrain elements but caption generation timed out.",
        confidence=0.45,
        confidence_level="LOW",
        warnings=["Specialist timeout after 5000ms"],
    )
    contract.status = "partial"

    report = ReportBuilder.build(contract=contract)

    assert report.task == "single_image_caption"
    assert report.confidence.score == 0.45
    assert any("timeout" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# 15. Malformed / Partial Output Fallback Safety
# ---------------------------------------------------------------------------

def test_malformed_contract_fallback_safety():
    """Verify ReportBuilder does not crash on malformed contract attributes."""
    class WeirdBrokenContract:
        task = "broken_task"
        answer = "broken answer"
        confidence = 0.5
        confidence_level = "MEDIUM"
        confidence_breakdown = None
        evidence = None
        evidence_status = None
        models = None
        parameters = {}
        warnings = []
        execution_trace = []
        execution_time_ms = 50

    report = ReportBuilder.build(contract=WeirdBrokenContract(), query="test")
    assert isinstance(report, AnalystReport)
    assert report.metadata.report_id.startswith("REP_")
    assert report.task == "broken_task"
    assert report.confidence.score == 0.5


# ---------------------------------------------------------------------------
# 16. JSON Serialization & Round-Trip
# ---------------------------------------------------------------------------

def test_json_serialization_roundtrip():
    """Verify report serializes cleanly to JSON and reconstructs into AnalystReport."""
    contract = create_sample_contract()
    report = ReportBuilder.build(contract=contract, query="Identify features")

    json_str = report.to_json(indent=2)
    assert isinstance(json_str, str)
    data = json.loads(json_str)

    assert data["metadata"]["mission"] == "Smart India Hackathon 2026 / SIH26167"
    assert data["task"] == "single_image_vqa"
    assert data["confidence"]["is_calibrated_probability"] is False

    # Round trip validation
    reconstructed = AnalystReport.model_validate(data)
    assert reconstructed.task == report.task
    assert reconstructed.confidence.score == report.confidence.score


# ---------------------------------------------------------------------------
# 17. Markdown Export
# ---------------------------------------------------------------------------

def test_markdown_export():
    """Verify to_markdown() generates an executive briefing with all required sections."""
    contract = create_sample_contract()
    report = ReportBuilder.build(contract=contract, query="Identify features", primary_filename="s2_optical.tif")

    md = report.to_markdown()
    assert isinstance(md, str)
    assert "# SatQuery AI — Executive Analysis Report" in md
    assert "Smart India Hackathon 2026 / SIH26167" in md
    assert "Indian Space Research Organisation (ISRO)" in md
    assert "## 1. Executive Summary" in md
    assert "## 2. System Confidence & Consistency Audit" in md
    assert "not calibrated probability" in md
    assert "## 3. Ingested Imagery & Pre-Flight Validation" in md
    assert "## 6. Specialist Model Provenance Ledger" in md
    assert "## 7. Auditable Execution Trace" in md


# ---------------------------------------------------------------------------
# 18. End-to-End API Integration
# ---------------------------------------------------------------------------

def test_api_analyze_returns_m11_report(optical_sample_path: Path):
    """Verify POST /api/v1/analyze returns StandardResultContract with enriched M11 report field."""
    with open(optical_sample_path, "rb") as f:
        response = client.post(
            "/api/v1/analyze",
            data={"query": "What type of land cover is present?"},
            files={"image_primary": ("optical_s2_sample.tif", f, "image/tiff")},
        )
    assert response.status_code == 200
    data = response.json()

    # Verify standard contract fields remain intact
    assert any(term in data["answer"].lower() for term in ["vegetation", "green", "forest", "land cover"])
    assert "confidence" in data
    assert "confidence_breakdown" in data
    assert "execution_trace" in data

    # Verify M11 report field is present and structured
    assert "report" in data
    assert data["report"] is not None
    rep = data["report"]
    assert rep["metadata"]["mission"] == "Smart India Hackathon 2026 / SIH26167"
    assert rep["task"] == data["task"]
    assert rep["answer"] == data["answer"]
    assert rep["confidence"]["score"] == data["confidence"]
    assert rep["confidence"]["is_calibrated_probability"] is False
    assert len(rep["execution_trace"]) == len(data["execution_trace"])
    assert len(rep["provenance"]) >= 1


# ---------------------------------------------------------------------------
# 19. Report Determinism
# ---------------------------------------------------------------------------

def test_report_determinism():
    """Verify reports built from identical contracts produce deterministic metrics and structures."""
    contract = create_sample_contract()
    rep1 = ReportBuilder.build(contract=contract, query="test query")
    rep2 = ReportBuilder.build(contract=contract, query="test query")

    assert rep1.task == rep2.task
    assert rep1.answer == rep2.answer
    assert rep1.confidence.score == rep2.confidence.score
    assert rep1.confidence.level == rep2.confidence.level
    assert rep1.statistics.metrics == rep2.statistics.metrics
    assert len(rep1.execution_trace) == len(rep2.execution_trace)
