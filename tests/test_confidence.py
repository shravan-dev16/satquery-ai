"""Unit and integration tests for M9 Defensible System Confidence Engine.

Adheres to AGENTS.md:
- Rule 10 (Consistency Checking influences Confidence)
- Rule 11 (Confidence Principle: Defensible, non-naive, explainable, no arbitrary scores)
- Rule 12 (Auditable Execution Trace: Operational telemetry without hidden CoT)
- Rule 23 (No Fake Implementations: Grounded calculation, honest calibration disclosure)
- Rule 31 (Never manufacture certainty)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pytest
from starlette.testclient import TestClient

from backend.agent.executor import AgentExecutor
from backend.agent.planner import AgentPlanner
from backend.agent.registry import registry
from backend.agent.router import AgentRouter
from backend.agent.schema import (
    BoundingBox,
    ComplementarityReport,
    ConfidenceBreakdown,
    ConfidenceLevel,
    ConsistencyReport,
    ConsistencyStatus,
    DetectedRegion,
    EvidenceBundle,
    EvidenceConflict,
    EvidenceItem,
    EvidenceMask,
    ModalityType,
    RoutingDecision,
    SemanticChangeInterpretation,
    SemanticTransition,
    SpecialistOutput,
    TaskType,
    ZonalStatistic,
)
from backend.evidence.confidence import ConfidenceEngine
from backend.evidence.consistency import ConsistencyChecker
from backend.main import app

client = TestClient(app)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_SAMPLE = FIXTURES_DIR / "real_rs_sample.tif"
OPTICAL_SAMPLE = FIXTURES_DIR / "optical_s2_sample.tif"
SAR_SAMPLE = FIXTURES_DIR / "sar_s1_sample.tif"
T1_BITEMPORAL = FIXTURES_DIR / "bitemporal" / "time1_pre_change.tif"
T2_BITEMPORAL = FIXTURES_DIR / "bitemporal" / "time2_post_change.tif"


@dataclass
class MockGeoMeta:
    crs: Optional[str] = None
    is_projected: bool = False


# ===========================================================================
# 1. HIGH-QUALITY CONSISTENT RESULT
# ===========================================================================
def test_high_quality_consistent_result():
    """Verify that a consistent, well-aligned, high-specialist input yields HIGH confidence with 0.0 penalty."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        conflict_count=0,
        critical_conflicts=0,
        summary_narrative="All specialist evidence is consistent.",
        evidence_quality_score=0.92,
    )
    bundle = EvidenceBundle(
        images=[],
        masks=[EvidenceMask(mask_id="m1", label="change_mask", url="/api/v1/previews/m1.png")],
        boxes=[
            BoundingBox(
                box_id="b1",
                label="change",
                coordinates_pixel=(10, 10, 50, 50),
                coordinates_normalized=(0.04, 0.04, 0.20, 0.20),
                confidence=0.90,
            )
        ],
        statistics=[ZonalStatistic(metric_name="changed_area_m2", display_name="Changed Area", value=4500.0, unit="m2")],
        regions=[DetectedRegion(region_name="reg_1", label="change_cluster", confidence=0.88)],
    )
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.92,
        execution_status="success",
        success=True,
    )
    geo_meta = MockGeoMeta(crs="EPSG:32643", is_projected=True)

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="bitemporal_change_detection",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        geo_meta=geo_meta,
        spatial_alignment_score=0.95,
    )

    assert isinstance(result, ConfidenceBreakdown)
    assert result.confidence_level == ConfidenceLevel.HIGH
    assert overall_conf >= 0.78
    assert result.consistency_penalty == 0.0
    assert "consistent_specialist_outputs" in result.confidence_factors
    assert "valid_geospatial_alignment" in result.confidence_factors
    assert "verified_spatial_crs" in result.confidence_factors
    assert len(result.confidence_warnings) == 0


# ===========================================================================
# 2. PARTIALLY CONSISTENT RESULT
# ===========================================================================
def test_partially_consistent_result():
    """Verify that PARTIALLY_CONSISTENT status caps confidence at <= 0.75 and applies 0.12 penalty."""
    report = ConsistencyReport(
        status=ConsistencyStatus.PARTIALLY_CONSISTENT,
        conflict_count=1,
        critical_conflicts=0,
        summary_narrative="Minor semantic variance detected across tools.",
        evidence_quality_score=0.85,
    )
    bundle = EvidenceBundle(
        images=[],
        masks=[EvidenceMask(mask_id="m1", label="change_mask", url="/api/v1/previews/m1.png")],
        boxes=[
            BoundingBox(
                box_id="b1",
                label="change",
                coordinates_pixel=(10, 10, 50, 50),
                coordinates_normalized=(0.04, 0.04, 0.20, 0.20),
                confidence=0.85,
            )
        ],
    )
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.90,
        execution_status="success",
        success=True,
    )

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="bitemporal_change_detection",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        spatial_alignment_score=0.90,
    )

    assert result.confidence_level == ConfidenceLevel.MEDIUM
    assert overall_conf <= 0.75
    assert result.consistency_penalty == 0.12
    assert any("partially consistent" in w.lower() for w in result.confidence_warnings)


# ===========================================================================
# 3. UNCERTAIN RESULT
# ===========================================================================
def test_uncertain_result():
    """Verify that UNCERTAIN status caps confidence at <= 0.60 and applies 0.20 penalty."""
    report = ConsistencyReport(
        status=ConsistencyStatus.UNCERTAIN,
        conflict_count=1,
        critical_conflicts=0,
        summary_narrative="Semantic model expressed high classification uncertainty.",
        evidence_quality_score=0.70,
    )
    bundle = EvidenceBundle(
        images=[],
        masks=[EvidenceMask(mask_id="m1", label="change_mask", url="/api/v1/previews/m1.png")],
    )
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.80,
        execution_status="success",
        success=True,
    )

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="bitemporal_change_detection",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        spatial_alignment_score=0.80,
    )

    assert overall_conf <= 0.60
    assert result.consistency_penalty == 0.20
    assert result.confidence_level in [ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]
    assert any("uncertain" in w.lower() for w in result.confidence_warnings)


# ===========================================================================
# 4. INSUFFICIENT EVIDENCE
# ===========================================================================
def test_insufficient_evidence():
    """Verify that INSUFFICIENT_EVIDENCE status caps confidence at <= 0.40 and applies 0.35 penalty."""
    report = ConsistencyReport(
        status=ConsistencyStatus.INSUFFICIENT_EVIDENCE,
        conflict_count=0,
        critical_conflicts=0,
        summary_narrative="Insufficient evidence available to substantiate finding.",
        evidence_quality_score=0.20,
    )
    bundle = EvidenceBundle(images=[], masks=[], boxes=[], statistics=[], regions=[])
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.70,
        execution_status="success",
        success=True,
    )

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="bitemporal_change_detection",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        spatial_alignment_score=0.50,
    )

    assert overall_conf <= 0.40
    assert result.consistency_penalty == 0.35
    assert result.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.UNSUPPORTED]
    assert any("insufficient" in w.lower() for w in result.confidence_warnings)


# ===========================================================================
# 5. CONTRADICTORY RESULT
# ===========================================================================
def test_contradictory_result():
    """Verify that CONTRADICTORY status caps confidence at <= 0.35, applies 0.50 penalty, and forces UNSUPPORTED."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONTRADICTORY,
        conflict_count=1,
        critical_conflicts=1,
        summary_narrative="Critical contradiction: detector found 0 pixels changed but VQA claimed change.",
        evidence_quality_score=0.40,
    )
    bundle = EvidenceBundle(images=[], masks=[], boxes=[])
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.75,
        execution_status="success",
        success=True,
    )

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="bitemporal_change_detection",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
    )

    assert result.confidence_level == ConfidenceLevel.UNSUPPORTED
    assert overall_conf <= 0.35
    assert result.consistency_penalty == 0.50
    assert any("contradiction" in w.lower() for w in result.confidence_warnings)


# ===========================================================================
# 6. CRITICAL CONTRADICTION CAPS CONFIDENCE (NON-NAIVE DOMINANCE)
# ===========================================================================
def test_critical_contradiction_caps_confidence():
    """Verify that high raw specialist confidence (0.99) is dominated by critical contradiction."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONTRADICTORY,
        conflict_count=2,
        critical_conflicts=2,
        summary_narrative="Critical conflict across specialists.",
        evidence_quality_score=0.90,
    )
    bundle = EvidenceBundle(
        images=[],
        masks=[EvidenceMask(mask_id="m1", label="change_mask", url="/m1.png")],
        boxes=[
            BoundingBox(
                box_id="b1",
                label="change",
                coordinates_pixel=(0, 0, 10, 10),
                coordinates_normalized=(0.0, 0.0, 0.04, 0.04),
                confidence=0.98,
            )
        ],
    )
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.99,  # Extremely high raw model score
        execution_status="success",
        success=True,
    )

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="bitemporal_change_detection",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        spatial_alignment_score=1.0,  # Perfect alignment
    )

    assert overall_conf <= 0.35
    assert result.confidence_level == ConfidenceLevel.UNSUPPORTED
    assert result.calculation_details["capped_by_status"] is True
    assert result.calculation_details["status_cap"] == 0.35


# ===========================================================================
# 7. SPECIALIST CONFIDENCE != SYSTEM CONFIDENCE
# ===========================================================================
def test_specialist_confidence_not_equal_system_confidence():
    """Verify that specialist confidence and system confidence remain distinct numerical values."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        conflict_count=0,
        critical_conflicts=0,
        summary_narrative="Consistent",
        evidence_quality_score=0.70,
    )
    specialist_output = SpecialistOutput(
        specialist_id="rs_vqa_specialist",
        task="single_image_vqa",
        confidence=0.95,
        execution_status="success",
        success=True,
    )
    bundle = EvidenceBundle(images=[])

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="single_image_vqa",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        spatial_alignment_score=0.60,
    )

    assert result.specialist_confidence == 0.95
    assert overall_conf != result.specialist_confidence
    assert overall_conf < 0.95


# ===========================================================================
# 8. MISSING EVIDENCE
# ===========================================================================
def test_missing_evidence():
    """Verify that completely empty evidence bundle lowers evidence score and reduces overall confidence."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        conflict_count=0,
        critical_conflicts=0,
        evidence_quality_score=0.30,
    )
    bundle = EvidenceBundle(images=[], masks=[], boxes=[], statistics=[], regions=[])
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.80,
        execution_status="success",
        success=True,
    )

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="bitemporal_change_detection",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        spatial_alignment_score=0.50,
    )

    assert result.evidence_quality_score <= 0.25
    assert any("sparse" in w.lower() or "missing" in w.lower() for w in result.confidence_warnings)


# ===========================================================================
# 9. INVALID ALIGNMENT
# ===========================================================================
def test_invalid_alignment():
    """Verify that poor or unverified spatial alignment reduces alignment score and warns."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        conflict_count=0,
        critical_conflicts=0,
        evidence_quality_score=0.80,
    )
    bundle = EvidenceBundle(
        images=[],
        masks=[EvidenceMask(mask_id="m1", label="change_mask", url="/m1.png")],
    )
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.85,
        execution_status="success",
        success=True,
    )
    geo_meta = MockGeoMeta(crs=None, is_projected=False)

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="bitemporal_change_detection",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        geo_meta=geo_meta,
        spatial_alignment_score=0.30,  # Weak alignment
    )

    assert result.spatial_alignment_score == 0.30
    assert any("spatial alignment" in w.lower() or "crs" in w.lower() for w in result.confidence_warnings)


# ===========================================================================
# 10. ZERO-CHANGE WORKFLOW
# ===========================================================================
def test_zero_change_workflow():
    """Verify that verified zero-change (0 pixels changed + consistent VQA) is treated as valid evidence."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        conflict_count=0,
        critical_conflicts=0,
        summary_narrative="Change detector and VQA both confirm zero surface change.",
        evidence_quality_score=0.90,
    )
    bundle = EvidenceBundle(
        images=[],
        masks=[],
        boxes=[],
        statistics=[ZonalStatistic(metric_name="changed_area_m2", display_name="Changed Area", value=0.0, unit="m2")],
        regions=[],
    )
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.92,
        execution_status="success",
        success=True,
    )

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="bitemporal_change_detection",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        spatial_alignment_score=0.90,
        parameters={"changed_pixels": 0, "change_ratio_pct": 0.0},
    )

    assert result.evidence_quality_score >= 0.85
    assert "verified_zero_change_state" in result.confidence_factors
    assert result.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]


# ===========================================================================
# 11. OPTICAL / SAR AGREEMENT
# ===========================================================================
def test_optical_sar_agreement():
    """Verify that joint Optical+SAR agreement with low unknown area yields high confidence."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        conflict_count=0,
        critical_conflicts=0,
        summary_narrative="Optical and SAR complementary signatures agree.",
        evidence_quality_score=0.90,
    )
    bundle = EvidenceBundle(
        images=[],
        masks=[EvidenceMask(mask_id="opt_sar_m", label="fused_mask", url="/m.png")],
        complementarity_report=ComplementarityReport(
            optical_limitations="Shadowed areas resolved by SAR.",
            sar_penetration="Confirmed structural footprint.",
            structural_contrast="High contrast between optical reflectance and SAR backscatter.",
        ),
    )
    specialist_output = SpecialistOutput(
        specialist_id="optical_sar_fusion_specialist",
        task="optical_sar_analysis",
        confidence=0.88,
        execution_status="success",
        success=True,
    )

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="optical_sar_analysis",
        specialist_output=specialist_output,
        consistency_report=report,
        evidence_bundle=bundle,
        spatial_alignment_score=0.92,
        parameters={"unknown_ratio_pct": 5.0},
    )

    assert result.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
    assert result.consistency_penalty == 0.0
    assert "consistent_specialist_outputs" in result.confidence_factors


# ===========================================================================
# 12. OPTICAL / SAR CONFLICT
# ===========================================================================
def test_optical_sar_conflict():
    """Verify that cross-modal conflict (Rule C6: water label with high backscatter) caps confidence at <= 0.35."""
    bundle = EvidenceBundle(
        regions=[
            DetectedRegion(
                region_name="reg_water_conflict",
                label="water_body",
                details={
                    "optical_evidence": {"mean_exg": 0.0},
                    "sar_evidence": {"mean_backscatter": 0.80, "roughness": 0.30},
                },
            )
        ],
        fused_items=[
            EvidenceItem(
                item_id="ev_01",
                source_specialist="OPTICAL_SAR_FUSION",
                evidence_type="cross_modal_joint_class",
                modality=ModalityType.CROSS_MODAL,
                claim="Conflicting water vs high backscatter",
                specialist_confidence=0.85,
            )
        ],
    )
    consistency_rep = ConsistencyReport(
        status=ConsistencyStatus.CONTRADICTORY,
        conflict_count=1,
        critical_conflicts=1,
        summary_narrative="Critical contradiction: optical indicates calm open water but SAR backscatter indicates heavy double-bounce infrastructure.",
        evidence_quality_score=0.35,
    )

    specialist_output = SpecialistOutput(
        specialist_id="optical_sar_fusion_specialist",
        task="optical_sar_analysis",
        confidence=0.85,
        execution_status="success",
        success=True,
    )

    overall_conf, result, level, factors, warnings = ConfidenceEngine.evaluate(
        task="optical_sar_analysis",
        specialist_output=specialist_output,
        consistency_report=consistency_rep,
        evidence_bundle=bundle,
        spatial_alignment_score=0.90,
    )

    assert result.confidence_level == ConfidenceLevel.UNSUPPORTED
    assert overall_conf <= 0.35
    assert result.consistency_penalty == 0.50


# ===========================================================================
# 13. SINGLE-IMAGE EVIDENCE SUFFICIENCY & DEGENERACY
# ===========================================================================
def test_single_image_evidence_sufficiency():
    """Verify that single-image grounding is rewarded for precise boxes and penalized for degenerate full-scene box."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        conflict_count=0,
        critical_conflicts=0,
        evidence_quality_score=0.85,
    )

    # Case A: Precise bounding box
    bundle_precise = EvidenceBundle(
        images=[],
        boxes=[
            BoundingBox(
                box_id="b1",
                label="aircraft",
                coordinates_pixel=(50, 50, 80, 80),
                coordinates_normalized=(0.20, 0.20, 0.31, 0.31),
                confidence=0.90,
            )
        ],
    )
    spec_a = SpecialistOutput(
        specialist_id="rs_grounding_specialist",
        task="single_image_grounding",
        confidence=0.88,
        execution_status="success",
        success=True,
    )
    conf_a, res_a, _, _, _ = ConfidenceEngine.evaluate(
        task="single_image_grounding",
        specialist_output=spec_a,
        consistency_report=report,
        evidence_bundle=bundle_precise,
    )
    assert "grounded_bounding_box_available" in res_a.confidence_factors

    # Case B: Degenerate bounding box covering 99% of 256x256 image
    bundle_degenerate = EvidenceBundle(
        images=[],
        boxes=[
            BoundingBox(
                box_id="b_deg",
                label="aircraft",
                coordinates_pixel=(1, 1, 255, 255),
                coordinates_normalized=(0.004, 0.004, 0.996, 0.996),
                confidence=0.88,
            )
        ],
    )
    conf_b, res_b, _, _, _ = ConfidenceEngine.evaluate(
        task="single_image_grounding",
        specialist_output=spec_a,
        consistency_report=report,
        evidence_bundle=bundle_degenerate,
    )
    assert conf_b < conf_a
    assert any("degenerate" in w.lower() for w in res_b.confidence_warnings)


# ===========================================================================
# 14. CONFIDENCE LEVEL MAPPING
# ===========================================================================
def test_confidence_level_mapping():
    """Verify exact threshold boundaries for ConfidenceLevel enum."""
    assert ConfidenceEngine.determine_level(0.95) == ConfidenceLevel.HIGH
    assert ConfidenceEngine.determine_level(0.78) == ConfidenceLevel.HIGH
    assert ConfidenceEngine.determine_level(0.779) == ConfidenceLevel.MEDIUM
    assert ConfidenceEngine.determine_level(0.55) == ConfidenceLevel.MEDIUM
    assert ConfidenceEngine.determine_level(0.549) == ConfidenceLevel.LOW
    assert ConfidenceEngine.determine_level(0.35) == ConfidenceLevel.LOW
    assert ConfidenceEngine.determine_level(0.349) == ConfidenceLevel.UNSUPPORTED
    assert ConfidenceEngine.determine_level(0.00) == ConfidenceLevel.UNSUPPORTED


# ===========================================================================
# 15. API INTEGRATION
# ===========================================================================
def test_api_integration():
    """Verify that POST /api/v1/analyze returns M9 confidence_breakdown, confidence_level, and system confidence."""
    with open(T1_BITEMPORAL, "rb") as f1, open(T2_BITEMPORAL, "rb") as f2:
        files = {
            "image_primary": ("t1.tif", f1, "image/tiff"),
            "image_secondary": ("t2.tif", f2, "image/tiff"),
        }
        data = {
            "query": "Detect urban changes between these two dates",
        }
        resp = client.post("/api/v1/analyze", files=files, data=data)

    assert resp.status_code == 200, f"API returned error: {resp.text}"
    body = resp.json()

    assert "confidence" in body
    assert "confidence_level" in body
    assert body["confidence_level"] in ["HIGH", "MEDIUM", "LOW", "UNSUPPORTED"]
    assert "confidence_breakdown" in body

    cb = body["confidence_breakdown"]
    assert "overall_confidence" in cb
    assert "specialist_confidence" in cb
    assert "input_quality_score" in cb
    assert "spatial_alignment_score" in cb
    assert "evidence_quality_score" in cb
    assert "consistency_penalty" in cb
    assert "confidence_factors" in cb
    assert "confidence_warnings" in cb
    assert "calculation_details" in cb
    assert cb["calculation_details"]["is_calibrated_probability"] is False


# ===========================================================================
# 16. TRACE INTEGRATION
# ===========================================================================
def test_trace_integration():
    """Verify that execution_trace contains operational telemetry about confidence calculation."""
    with open(REAL_SAMPLE, "rb") as f:
        files = {"image_primary": ("test.tif", f, "image/tiff")}
        data = {"query": "What is the land use type in this satellite image?"}
        resp = client.post("/api/v1/analyze", files=files, data=data)

    assert resp.status_code == 200
    body = resp.json()
    trace = body.get("execution_trace", [])
    assert len(trace) >= 4

    conf_step = next((s for s in trace if s["step_name"] == "ConfidenceCalculation"), None)
    if conf_step:
        meta = conf_step["metadata"]
        assert "confidence_level" in meta
        assert "is_capped" in meta
        assert "is_calibrated" in meta
        assert "inputs_considered" in meta
    else:
        assembly_step = next((s for s in trace if s["step_name"] == "EvidenceAssembly"), None)
        assert assembly_step is not None
        assert "confidence_calculation" in assembly_step["metadata"]


# ===========================================================================
# 17. REPRODUCIBILITY
# ===========================================================================
def test_reproducibility():
    """Verify that running ConfidenceEngine multiple times with identical inputs produces bitwise identical results."""
    report = ConsistencyReport(
        status=ConsistencyStatus.CONSISTENT,
        conflict_count=0,
        critical_conflicts=0,
        summary_narrative="Consistent",
        evidence_quality_score=0.88,
    )
    bundle = EvidenceBundle(
        images=[],
        masks=[EvidenceMask(mask_id="m1", label="change_mask", url="/m1.png")],
        boxes=[
            BoundingBox(
                box_id="b1",
                label="change",
                coordinates_pixel=(10, 10, 50, 50),
                coordinates_normalized=(0.04, 0.04, 0.20, 0.20),
                confidence=0.85,
            )
        ],
    )
    specialist_output = SpecialistOutput(
        specialist_id="rs_change_detector",
        task="bitemporal_change_detection",
        confidence=0.91,
        execution_status="success",
        success=True,
    )

    runs = [
        ConfidenceEngine.evaluate(
            task="bitemporal_change_detection",
            specialist_output=specialist_output,
            consistency_report=report,
            evidence_bundle=bundle,
            spatial_alignment_score=0.92,
        )
        for _ in range(5)
    ]

    for r in runs[1:]:
        assert r[0] == runs[0][0]  # overall_confidence
        assert r[1].confidence_level == runs[0][1].confidence_level
        assert r[1].consistency_penalty == runs[0][1].consistency_penalty
        assert r[1].confidence_factors == runs[0][1].confidence_factors


# ===========================================================================
# 18. FULL M0-M8 REGRESSION
# ===========================================================================
def test_full_m0_m8_regression():
    """Verify that agentic executor and full pipeline with M0-M8 outputs function cleanly with M9 confidence engine."""
    with open(T1_BITEMPORAL, "rb") as f1, open(T2_BITEMPORAL, "rb") as f2:
        files = {
            "image_primary": ("t1.tif", f1, "image/tiff"),
            "image_secondary": ("t2.tif", f2, "image/tiff"),
        }
        data = {"query": "Assess surface change between these two acquisitions"}
        resp = client.post("/api/v1/analyze", files=files, data=data)

    assert resp.status_code == 200
    contract = resp.json()

    # M0 & M1 Verification
    assert "answer" in contract
    assert "confidence" in contract
    assert 0.0 <= contract["confidence"] <= 1.0

    # M3 / M4 Verification (Parameters)
    assert "changed_pixels" in contract["parameters"]
    assert "change_ratio_pct" in contract["parameters"]

    # M4 & M5 Verification (Bitemporal change evidence & semantic interpretation)
    assert contract["task"] == "bitemporal_change_detection"
    assert "evidence" in contract
    assert "semantic_interpretation" in contract["evidence"]

    # M7 Verification (Agentic trace)
    assert len(contract["execution_trace"]) >= 4

    # M8 Verification (Evidence status & Consistency report)
    assert "evidence_status" in contract
    assert "consistency_report" in contract["evidence"]
    assert contract["evidence"]["consistency_report"] is not None

    # M9 Verification (Confidence breakdown & Level)
    assert "confidence_level" in contract
    assert contract["confidence_level"] in ["HIGH", "MEDIUM", "LOW", "UNSUPPORTED"]
    assert contract["confidence_breakdown"] is not None
    assert contract["confidence_breakdown"]["overall_confidence"] == contract["confidence"]
