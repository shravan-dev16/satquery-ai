"""Unit and integration tests for Consistency Checking, Rules C1-C8, and Evidence Gating (Milestone M8).

Adheres to AGENTS.md:
- Rule 10 (Consistency Checking)
- Rule 11 (Specialist confidence separated from final system confidence)
- Rule 24 (Graceful fallbacks and transparent conflict exposure)
- Rule 31 (Never manufacture certainty)
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.agent.executor import AgentExecutor
from backend.agent.planner import AgentPlanner
from backend.agent.registry import registry
from backend.agent.router import AgentRouter
from backend.agent.schema import (
    BoundingBox,
    ConsistencyReport,
    ConsistencyStatus,
    DetectedRegion,
    EvidenceBundle,
    EvidenceConflict,
    EvidenceItem,
    ModalityType,
    RoutingDecision,
    SemanticChangeInterpretation,
    SemanticTransition,
    SpecialistOutput,
    TaskType,
    ZonalStatistic,
)
from backend.evidence.consistency import ConsistencyChecker, EvidenceGater
from backend.main import app

client = TestClient(app)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_SAMPLE = FIXTURES_DIR / "real_rs_sample.tif"
SYNTH_OPTICAL = FIXTURES_DIR / "synthetic_optical_sample.tif"
SYNTH_SAR = FIXTURES_DIR / "synthetic_sar_sample.tif"


# ===========================================================================
# 1. RULE C1: ZERO-CHANGE CONTRADICTION
# ===========================================================================
def test_rule_c1_zero_change_contradiction():
    """Verify that when detector finds 0 changed pixels but semantic VQA claims change, status is CONTRADICTORY."""
    bundle = EvidenceBundle(
        regions=[],
        semantic_interpretation=SemanticChangeInterpretation(
            summary="Extensive urban expansion observed across the scene.",
            temporal_direction="increased",
            predominant_transition="bare_land -> built_up",
            transitions=[
                SemanticTransition(
                    transition_id="trans_01",
                    from_class="bare_land",
                    to_class="built_up",
                    description="Multiple new residential complexes built.",
                    region_id="cluster_nonexistent",
                )
            ],
        ),
        fused_items=[
            EvidenceItem(
                item_id="ev_01",
                source_specialist="CHANGE_DETECT",
                evidence_type="spatial_mask",
                modality=ModalityType.BITEMPORAL,
                claim="Detected 0 changed pixels.",
                metrics={"changed_pixels": 0},
                specialist_confidence=0.95,
            ),
            EvidenceItem(
                item_id="ev_02",
                source_specialist="CHANGE_VQA",
                evidence_type="semantic_interpretation",
                modality=ModalityType.BITEMPORAL,
                claim="Urban expansion observed.",
                metrics={"temporal_direction": "increased"},
                specialist_confidence=0.85,
            ),
        ],
    )

    report = ConsistencyChecker.check_bitemporal(
        evidence=bundle,
        change_params={"changed_pixels": 0, "total_clusters": 0},
        temporal_ordering="valid",
        query="What changed between these two dates?",
    )

    assert report.status == ConsistencyStatus.CONTRADICTORY
    assert report.is_gated is True
    assert report.gating_action == "flag_contradiction"
    assert any(c.rule_violated == "C1_ZERO_CHANGE_CONTRADICTION" for c in report.conflicts)

    # Verify gating protects the final answer
    raw_answer = "Extensive urban expansion observed across the scene."
    gated = EvidenceGater.gate_answer(raw_answer, report)
    assert "[CONTRADICTION DETECTED]" in gated
    assert "0 changed pixels" in gated
    assert "cannot be verified as factual" in gated


# ===========================================================================
# 2. RULE C2: REGION SUPPORT
# ===========================================================================
def test_rule_c2_unsupported_region():
    """Verify semantic transitions referencing non-existent regions are flagged as unsupported."""
    bundle = EvidenceBundle(
        regions=[
            DetectedRegion(region_name="cluster_valid_01", label="change", bbox_pixel=(10, 10, 50, 50))
        ],
        semantic_interpretation=SemanticChangeInterpretation(
            summary="Construction detected in northern sector.",
            temporal_direction="modified",
            predominant_transition="bare_land -> commercial",
            transitions=[
                SemanticTransition(
                    transition_id="trans_01",
                    from_class="bare_land",
                    to_class="commercial",
                    description="Commercial structure built.",
                    region_id="cluster_phantom_99",  # Does not exist in detector regions
                )
            ],
        ),
        fused_items=[
            EvidenceItem(
                item_id="ev_01",
                source_specialist="CHANGE_DETECT",
                evidence_type="spatial_mask",
                modality=ModalityType.BITEMPORAL,
                claim="Detected 2,500 changed pixels in cluster_valid_01.",
                metrics={"changed_pixels": 2500},
                specialist_confidence=0.90,
            )
        ],
    )

    report = ConsistencyChecker.check_bitemporal(
        evidence=bundle,
        change_params={"changed_pixels": 2500, "total_clusters": 1},
        temporal_ordering="valid",
        query="What changed?",
    )

    assert report.status in (ConsistencyStatus.PARTIALLY_CONSISTENT, ConsistencyStatus.UNCERTAIN)
    assert any(c.rule_violated == "C2_UNSUPPORTED_REGION" for c in report.conflicts)
    assert "cluster_phantom_99" in report.conflicts[0].description


# ===========================================================================
# 3. RULE C3: NONZERO CHANGE SUPPORT
# ===========================================================================
def test_rule_c3_nonzero_change_support():
    """Verify semantic transitions without nonzero physical changed pixels trigger contradiction."""
    bundle = EvidenceBundle(
        regions=[],
        semantic_interpretation=SemanticChangeInterpretation(
            summary="Subtle alteration.",
            temporal_direction="modified",
            predominant_transition="vegetation -> soil",
            transitions=[
                SemanticTransition(
                    transition_id="t1",
                    from_class="vegetation",
                    to_class="soil",
                    description="Soil cleared.",
                )
            ],
        ),
        fused_items=[],
    )

    report = ConsistencyChecker.check_bitemporal(
        evidence=bundle,
        change_params={"changed_pixels": 0, "total_clusters": 0},
        temporal_ordering="valid",
        query="What changed?",
    )
    assert report.status == ConsistencyStatus.CONTRADICTORY


# ===========================================================================
# 4. RULE C4: AREA CONSISTENCY
# ===========================================================================
def test_rule_c4_area_consistency():
    """Verify claimed area is checked against measured detector zonal statistics within 10% tolerance."""
    bundle = EvidenceBundle(
        statistics=[
            ZonalStatistic(metric_name="physical_area_ha", display_name="Physical Area", value=10.0, unit="ha")
        ],
        semantic_interpretation=SemanticChangeInterpretation(
            summary="Area conversion.",
            temporal_direction="modified",
            predominant_transition="crop -> soil",
            transitions=[
                SemanticTransition(
                    transition_id="t1",
                    from_class="crop",
                    to_class="soil",
                    description="Farmland altered.",
                    details={"claimed_area_ha": 15.0},  # 50% discrepancy (>10% tolerance)
                )
            ],
        ),
    )

    report = ConsistencyChecker.check_bitemporal(
        evidence=bundle,
        change_params={"changed_pixels": 10000, "total_clusters": 1},
        temporal_ordering="valid",
    )
    assert any(c.rule_violated == "C4_AREA_CONSISTENCY" for c in report.conflicts)


# ===========================================================================
# 5. RULE C5: TEMPORAL DIRECTION & EVIDENCE SUPPORT (User Correction 2)
# ===========================================================================
def test_rule_c5_temporal_chronology_and_evidence_direction():
    """Verify Rule C5: T1 < T2 validates chronology, while direction is checked against T1->T2 evidence."""
    # Sub-case A: Invalid chronological ordering
    bundle_a = EvidenceBundle()
    report_a = ConsistencyChecker.check_bitemporal(
        evidence=bundle_a,
        change_params={"changed_pixels": 1000},
        temporal_ordering="reversed",
    )
    assert any(c.rule_violated == "C5_TEMPORAL_CHRONOLOGY_INVALID" for c in report_a.conflicts)
    assert report_a.status == ConsistencyStatus.CONTRADICTORY

    # Sub-case B: Direction claims 'increased' but all transitions show vegetation clearing to bare soil
    bundle_b = EvidenceBundle(
        semantic_interpretation=SemanticChangeInterpretation(
            summary="Change detected.",
            temporal_direction="increased",
            predominant_transition="forest -> bare_ground",
            transitions=[
                SemanticTransition(
                    transition_id="t1",
                    from_class="forest",
                    to_class="bare_ground",
                    description="Extensive forest clearing leaving bare ground.",
                )
            ],
        )
    )
    report_b = ConsistencyChecker.check_bitemporal(
        evidence=bundle_b,
        change_params={"changed_pixels": 3000},
        temporal_ordering="valid",
    )
    assert any(c.rule_violated == "C5_TEMPORAL_DIRECTION_CONFLICT" for c in report_b.conflicts)


# ===========================================================================
# 6. RULE C6: OPTICAL/SAR AGREEMENT
# ===========================================================================
def test_rule_c6_optical_sar_agreement_and_conflict():
    """Verify Rule C6 flags cross-modal discrepancies between optical spectral and SAR physical cues."""
    # Sub-case A: Agreement on built structure
    bundle_agree = EvidenceBundle(
        regions=[
            DetectedRegion(
                region_name="reg_agree",
                label="built_structure",
                details={
                    "optical_evidence": {"mean_brightness": 0.40, "mean_exg": -0.01},
                    "sar_evidence": {"mean_backscatter": 0.65, "roughness": 0.18},  # Strong double bounce
                },
            )
        ]
    )
    report_agree = ConsistencyChecker.check_optical_sar(bundle_agree, query="Identify built-up areas")
    assert report_agree.status == ConsistencyStatus.CONSISTENT
    assert len(report_agree.conflicts) == 0

    # Sub-case B: Conflict - optically vegetation, but extreme SAR backscatter and double bounce
    bundle_conflict = EvidenceBundle(
        regions=[
            DetectedRegion(
                region_name="reg_conflict",
                label="vegetation_or_cropland",
                details={
                    "optical_evidence": {"mean_brightness": 0.30, "mean_exg": 0.22},  # High greenness
                    "sar_evidence": {"mean_backscatter": 0.85, "roughness": 0.25},  # Double bounce!
                },
            )
        ]
    )
    report_conflict = ConsistencyChecker.check_optical_sar(bundle_conflict, query="Analyze cross-modal features")
    assert report_conflict.status in (ConsistencyStatus.PARTIALLY_CONSISTENT, ConsistencyStatus.UNCERTAIN)
    assert any(c.rule_violated == "C6_OPTICAL_SAR_DISAGREEMENT" for c in report_conflict.conflicts)


# ===========================================================================
# 7. RULE C8: EVIDENCE SUFFICIENCY (User Correction 3)
# ===========================================================================
def test_rule_c8_single_image_sufficiency_and_degeneracy():
    """Verify single image checks sufficiency and box degeneracy without fake cross-model agreement."""
    # Sub-case A: Empty boxes for explicit referring query
    bundle_empty = EvidenceBundle(boxes=[])
    report_empty = ConsistencyChecker.check_single_image(
        evidence=bundle_empty,
        task_name="grounding",
        query="Where is the water reservoir?",
        image_metadata={"filename": "test.tif", "crs": "EPSG:32618"},
    )
    assert report_empty.status == ConsistencyStatus.INSUFFICIENT_EVIDENCE
    assert report_empty.is_gated is True
    assert report_empty.gating_action == "state_insufficient"

    # Sub-case B: Degenerate full-image box
    bundle_degen = EvidenceBundle(
        boxes=[
            BoundingBox(
                box_id="b1",
                label="water",
                confidence=0.20,
                is_degenerate=True,
                degenerate_reason="Box covers >=98% of entire image area",
                coordinates_normalized=(0, 0, 1, 1),
                coordinates_pixel=(0, 0, 512, 512),
            )
        ]
    )
    report_degen = ConsistencyChecker.check_single_image(
        evidence=bundle_degen,
        task_name="grounding",
        query="Where is the water?",
    )
    assert any(c.rule_violated == "C8_DEGENERATE_DETECTION" for c in report_degen.conflicts)


# ===========================================================================
# 8. SEPARATION OF CONFIDENCE (User Correction 1)
# ===========================================================================
def test_separation_of_confidence_m8_boundary():
    """Verify M8 does NOT deduct numerical consistency penalties; penalty remains 0.0 for M9."""
    decision = AgentRouter.route(
        query="What is in this image?",
        primary_path=REAL_SAMPLE,
    )
    plan = AgentPlanner.create_plan(decision, primary_path=REAL_SAMPLE)
    result = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=REAL_SAMPLE,
        query="What is in this image?",
        primary_filename="real_rs_sample.tif",
    )

    # Numerical consistency penalty must remain 0.0 (belongs exclusively to M9)
    assert result.confidence_breakdown.consistency_penalty == 0.0

    # Structured/qualitative consistency report must be present
    assert result.evidence.consistency_report is not None
    assert result.evidence_status is not None
    assert result.evidence_status in ("CONSISTENT", "PARTIALLY_CONSISTENT", "UNCERTAIN", "CONTRADICTORY", "INSUFFICIENT_EVIDENCE")


# ===========================================================================
# 9. EXECUTION TRACE INTEGRATION (Phase 12)
# ===========================================================================
def test_execution_trace_includes_m8_steps():
    """Verify execution trace contains EvidenceNormalization, ConsistencyCheck, EvidenceFusion, EvidenceAssembly."""
    decision = AgentRouter.route(
        query="What is in this scene?",
        primary_path=REAL_SAMPLE,
    )
    plan = AgentPlanner.create_plan(decision, primary_path=REAL_SAMPLE)
    result = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=REAL_SAMPLE,
        query="What is in this scene?",
        primary_filename="real_rs_sample.tif",
    )

    step_names = [s.step_name for s in result.execution_trace]
    assert "EvidenceNormalization" in step_names
    assert "ConsistencyCheck" in step_names
    assert "EvidenceFusion" in step_names
    assert "EvidenceAssembly" in step_names

    # Check step ordering: Normalization precedes ConsistencyCheck which precedes EvidenceFusion
    idx_norm = step_names.index("EvidenceNormalization")
    idx_cons = step_names.index("ConsistencyCheck")
    idx_fuse = step_names.index("EvidenceFusion")
    idx_assemb = step_names.index("EvidenceAssembly")
    assert idx_norm < idx_cons < idx_fuse < idx_assemb


# ===========================================================================
# 10. API END-TO-END VERIFICATION
# ===========================================================================
def test_api_analyze_returns_m8_evidence_status_and_items():
    """Verify POST /api/v1/analyze returns evidence_status and populated fused_items."""
    with open(REAL_SAMPLE, "rb") as f:
        resp = client.post(
            "/api/v1/analyze",
            data={"query": "Describe the features in this remote sensing image."},
            files={"image_primary": ("real_rs_sample.tif", f, "image/tiff")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "evidence_status" in data
    assert data["evidence_status"] is not None
    assert "fused_items" in data["evidence"]
    assert len(data["evidence"]["fused_items"]) >= 1
    assert data["evidence"]["consistency_report"] is not None
    assert data["evidence"]["consistency_report"]["status"] == data["evidence_status"]
