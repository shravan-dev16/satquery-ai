"""M7 Routing and Agentic Orchestration Test Matrix.

Verifies:
1. Single image -> VQA (RS_VQA)
2. Single image -> Grounding (RS_GROUND)
3. Bi-temporal pair -> Change Analysis (multi-stage)
4. Bi-temporal query requiring semantic interpretation
5. Optical + SAR -> Optical-SAR specialist (OPTICAL_SAR_FUSION)
6. Two optical images -> rejected for optical-SAR joint analysis
7. Two SAR images -> rejected for optical-SAR joint analysis
8. Single image + temporal question -> validation failure (cannot satisfy bi-temporal)
9. Optical + SAR + temporal-only query -> rejected (sensor physics mismatch)
10. Query wording variations map to the same task
11. Specialist lookup resolves from ModelRegistry (not hardcoded classes)
12. Multi-stage change plan is ordered correctly (CHANGE_DETECT before CHANGE_VQA)
13. Trace contains actual execution order and real durations
14. API automatic routing operates without manual task hint
15. Manual candidate override (task_hint) works for debugging
16. Ambiguous/impossible combinations return structured failure
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.agent.executor import AgentExecutor, ExecutionError
from backend.agent.planner import AgentPlanner
from backend.agent.registry import ModelRegistry, registry
from backend.agent.router import AgentRouter
from backend.agent.schema import (
    EvidenceBundle,
    ExecutionPlan,
    ModelCapability,
    PlanStep,
    RoutingDecision,
    SpecialistInput,
    SpecialistOutput,
    StandardResultContract,
    StepStatus,
    TaskType,
    ModalityType,
)
from backend.main import app
from backend.models.base import BaseSpecialist

client = TestClient(app)

OPTICAL_PATH = Path("tests/fixtures/optical_s2_sample.tif")
T1_PATH = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
T2_PATH = Path("tests/fixtures/bitemporal/time2_post_change.tif")
OPT_FUSION_PATH = Path("tests/fixtures/optical_sar/optical_multimodal_01.tif")
SAR_FUSION_PATH = Path("tests/fixtures/optical_sar/sar_multimodal_01.tif")


# ===========================================================================
# 1. SINGLE IMAGE -> VQA
# ===========================================================================
def test_single_image_vqa_routing():
    """Verify single optical image with analytical question routes to RS_VQA."""
    decision = AgentRouter.route(
        query="What type of land cover is visible across this scene?",
        primary_path=OPTICAL_PATH,
    )
    assert decision.is_valid is True
    assert decision.task == TaskType.VQA
    assert decision.target_specialist_ids == ["RS_VQA"]
    assert "single" in decision.input_configuration

    plan = AgentPlanner.create_plan(decision, primary_path=OPTICAL_PATH)
    assert len(plan.steps) >= 3
    action_types = [s.action_type for s in plan.steps]
    assert "specialist_predict" in action_types
    assert "assemble_evidence" in action_types


# ===========================================================================
# 2. SINGLE IMAGE -> GROUNDING
# ===========================================================================
def test_single_image_grounding_routing():
    """Verify single optical image with spatial query routes to RS_GROUND."""
    queries = [
        "Where is the water body?",
        "Locate the runway in this image",
        "Highlight the forested region",
        "Ground bounding box for the settlement",
    ]
    for q in queries:
        decision = AgentRouter.route(query=q, primary_path=OPTICAL_PATH)
        assert decision.is_valid is True
        assert decision.task == TaskType.GROUNDING
        assert decision.target_specialist_ids == ["RS_GROUND"]
        assert decision.intent_category == "region_grounding"


# ===========================================================================
# 3. BI-TEMPORAL PAIR -> CHANGE ANALYSIS
# ===========================================================================
def test_bitemporal_pair_change_detection_routing():
    """Verify bi-temporal optical pair routes to CHANGE_DETECT for binary change localization."""
    decision = AgentRouter.route(
        query="What changed between these two dates?",
        primary_path=T1_PATH,
        secondary_path=T2_PATH,
    )
    assert decision.is_valid is True
    assert decision.task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA)
    assert "CHANGE_DETECT" in decision.target_specialist_ids
    assert decision.input_configuration == "bitemporal_optical_pair"


# ===========================================================================
# 4. BI-TEMPORAL QUERY REQUIRING SEMANTIC INTERPRETATION
# ===========================================================================
def test_bitemporal_semantic_interpretation_routing():
    """Verify semantic change questions route to multi-stage CHANGE_VQA."""
    semantic_queries = [
        "Describe what type of land-cover change occurred between these dates.",
        "Has the built-up area increased, decreased, or remained unchanged?",
        "Explain how vegetation cover transformed over time.",
        "What kind of change happened in this urban area?",
    ]
    for q in semantic_queries:
        decision = AgentRouter.route(
            query=q,
            primary_path=T1_PATH,
            secondary_path=T2_PATH,
        )
        assert decision.is_valid is True
        assert decision.task == TaskType.CHANGE_VQA
        assert decision.is_multi_stage is True
        assert decision.target_specialist_ids == ["CHANGE_DETECT", "CHANGE_VQA"]

        plan = AgentPlanner.create_plan(decision, primary_path=T1_PATH, secondary_path=T2_PATH)
        assert plan.is_multi_stage is True
        specialist_steps = [s for s in plan.steps if s.action_type == "specialist_predict"]
        assert len(specialist_steps) == 2
        assert specialist_steps[0].specialist_id == "CHANGE_DETECT"
        assert specialist_steps[1].specialist_id == "CHANGE_VQA"


# ===========================================================================
# 5. OPTICAL + SAR -> OPTICAL-SAR SPECIALIST
# ===========================================================================
def test_optical_sar_joint_routing():
    """Verify Optical + SAR pair routes to OPTICAL_SAR_FUSION specialist."""
    queries = [
        "Identify built-up areas using both optical and SAR information.",
        "Use SAR backscatter and optical reflectance jointly to map surface water.",
        "Cross-modal analysis for urban infrastructure.",
    ]
    for q in queries:
        decision = AgentRouter.route(
            query=q,
            primary_path=OPT_FUSION_PATH,
            secondary_path=SAR_FUSION_PATH,
        )
        assert decision.is_valid is True
        assert decision.task == TaskType.OPTICAL_SAR_ANALYSIS
        assert decision.target_specialist_ids == ["OPTICAL_SAR_FUSION"]
        assert decision.input_configuration == "cross_modal_optical_sar"


# ===========================================================================
# 6. TWO OPTICAL -> NOT OPTICAL-SAR
# ===========================================================================
def test_two_optical_not_optical_sar():
    """Verify cross-modal SAR request on two optical rasters is cleanly rejected."""
    decision = AgentRouter.route(
        query="Use optical and SAR imagery together to classify buildings.",
        primary_path=T1_PATH,
        secondary_path=T2_PATH,
    )
    assert decision.is_valid is False
    assert "Optical" in decision.rejection_reason
    assert "SAR" in decision.rejection_reason
    assert decision.target_specialist_ids == []


# ===========================================================================
# 7. TWO SAR -> NOT OPTICAL-SAR
# ===========================================================================
def test_two_sar_not_optical_sar():
    """Verify cross-modal request on two SAR rasters is cleanly rejected."""
    decision = AgentRouter.route(
        query="Use optical and SAR fusion to classify terrain.",
        primary_path=SAR_FUSION_PATH,
        secondary_path=SAR_FUSION_PATH,
    )
    assert decision.is_valid is False
    assert "both uploaded images are SAR" in decision.rejection_reason


# ===========================================================================
# 8. SINGLE IMAGE + TEMPORAL QUESTION -> VALIDATION FAILURE
# ===========================================================================
def test_single_image_temporal_question_validation_failure():
    """Verify single raster with temporal change question is rejected with structured reason."""
    decision = AgentRouter.route(
        query="What changed between these two dates?",
        primary_path=OPTICAL_PATH,
        secondary_path=None,
    )
    assert decision.is_valid is False
    assert "requires both Time 1" in decision.rejection_reason


# ===========================================================================
# 9. OPTICAL + SAR + TEMPORAL-ONLY QUERY -> REJECTED
# ===========================================================================
def test_optical_sar_temporal_query_rejected():
    """Verify pure temporal change query on Optical+SAR cross-modal pair is rejected (physics mismatch)."""
    decision = AgentRouter.route(
        query="What changed between 2020 and 2024?",
        primary_path=OPT_FUSION_PATH,
        secondary_path=SAR_FUSION_PATH,
    )
    assert decision.is_valid is False
    assert "sensor physics mismatch" in decision.rejection_reason.lower()


# ===========================================================================
# 10. QUERY WORDING VARIATIONS MAP TO SAME TASK
# ===========================================================================
def test_query_wording_variations_same_task():
    """Verify diverse phrasing robustly maps to uniform task classifications."""
    grounding_phrasings = [
        "Where is the lake?",
        "Locate where the water is",
        "Find bounding box for the river",
        "Ground water bodies",
        "Segment the forest area",
    ]
    for q in grounding_phrasings:
        dec = AgentRouter.route(q, primary_path=OPTICAL_PATH)
        assert dec.task == TaskType.GROUNDING, f"Failed on query: '{q}'"

    vqa_phrasings = [
        "What is the predominant land cover?",
        "Is this an urban or rural region?",
        "Does this image contain cloud cover?",
        "Analyze the geographic features in this scene",
    ]
    for q in vqa_phrasings:
        dec = AgentRouter.route(q, primary_path=OPTICAL_PATH)
        assert dec.task == TaskType.VQA, f"Failed on query: '{q}'"


# ===========================================================================
# 11. SPECIALIST LOOKUP COMES FROM MODEL REGISTRY
# ===========================================================================
class CustomRegistrySpecialist(BaseSpecialist):
    """Custom specialist registered dynamically to verify registry-based dispatching."""

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            identifier="CUSTOM_DYNAMIC_SPEC",
            name="Custom Dynamic Specialist",
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
            specialist_id="CUSTOM_DYNAMIC_SPEC",
            success=True,
            answer_text="Dynamically resolved specialist executed successfully.",
            confidence=0.99,
            evidence=EvidenceBundle(),
            parameters_used={"resolved_via": "ModelRegistry"},
            execution_time_ms=10,
        )

    def health_check(self) -> bool:
        return True


def test_specialist_lookup_from_model_registry():
    """Verify AgentExecutor resolves specialists via registry rather than hardcoding class instances."""
    custom_spec = CustomRegistrySpecialist()
    registry.register(custom_spec)

    try:
        decision = RoutingDecision(
            task=TaskType.VQA,
            target_specialist_ids=["CUSTOM_DYNAMIC_SPEC"],
            intent_category="test_lookup",
            input_configuration="single_optical",
            is_valid=True,
            reasoning="Testing ModelRegistry dynamic resolution",
        )
        plan = AgentPlanner.create_plan(decision, primary_path=OPTICAL_PATH)

        result = AgentExecutor.execute(
            plan=plan,
            decision=decision,
            primary_path=OPTICAL_PATH,
            query="Test query",
            primary_filename="test_opt.tif",
        )

        assert result.status == "success"
        assert result.answer == "Dynamically resolved specialist executed successfully."
        assert len(result.models) == 1
        assert result.models[0].identifier == "CUSTOM_DYNAMIC_SPEC"
    finally:
        registry.unregister("CUSTOM_DYNAMIC_SPEC")


# ===========================================================================
# 12. MULTI-STAGE CHANGE PLAN IS ORDERED CORRECTLY
# ===========================================================================
def test_multi_stage_change_plan_ordered_correctly():
    """Verify Planner orders TinyCD (WHERE/HOW MUCH) before ChangeVQA (WHAT KIND)."""
    decision = RoutingDecision(
        task=TaskType.CHANGE_VQA,
        target_specialist_ids=["CHANGE_DETECT", "CHANGE_VQA"],
        intent_category="semantic_change_vqa",
        input_configuration="bitemporal_optical_pair",
        is_multi_stage=True,
        reasoning="Test multi-stage sequencing order",
    )
    plan = AgentPlanner.create_plan(decision, primary_path=T1_PATH, secondary_path=T2_PATH)

    cd_idx = next(i for i, s in enumerate(plan.steps) if s.specialist_id == "CHANGE_DETECT")
    cvqa_idx = next(i for i, s in enumerate(plan.steps) if s.specialist_id == "CHANGE_VQA")

    assert cd_idx < cvqa_idx, "CHANGE_DETECT must precede CHANGE_VQA in execution plan"


# ===========================================================================
# 13. TRACE CONTAINS ACTUAL EXECUTION ORDER
# ===========================================================================
def test_trace_contains_actual_execution_order():
    """Verify execution trace captures sequenced operations with monotonically increasing step numbers."""
    decision = AgentRouter.route(
        query="What is the terrain type?",
        primary_path=OPTICAL_PATH,
    )
    plan = AgentPlanner.create_plan(decision, primary_path=OPTICAL_PATH)
    result = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=OPTICAL_PATH,
        query="What is the terrain type?",
        primary_filename="sample.tif",
    )

    trace = result.execution_trace
    assert len(trace) >= 4

    step_nums = [s.step_number for s in trace]
    assert step_nums == list(range(1, len(trace) + 1))

    for s in trace:
        assert s.status == StepStatus.COMPLETED
        assert s.duration_ms >= 0


# ===========================================================================
# 14. API AUTOMATIC ROUTING
# ===========================================================================
def test_api_automatic_routing():
    """Verify POST /api/v1/analyze automatically infers task and specialist without user selection."""
    with open(OPTICAL_PATH, "rb") as f:
        res = client.post(
            "/api/v1/analyze",
            data={"query": "Where is the water body?"},
            files={"image_primary": ("optical.tif", f, "image/tiff")},
        )
    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "single_image_grounding"
    assert data["status"] == "success"
    assert "boxes" in data["evidence"]


# ===========================================================================
# 15. MANUAL CANDIDATE OVERRIDE DOES NOT BREAK
# ===========================================================================
def test_manual_candidate_override_does_not_break():
    """Verify debug overrides (task_hint='tinycd_raw' or 'cva_raw') remain backward compatible."""
    with open(T1_PATH, "rb") as f1, open(T2_PATH, "rb") as f2:
        res = client.post(
            "/api/v1/analyze",
            data={
                "query": "Detect changes",
                "task_hint": "cva_raw",
            },
            files={
                "image_primary": ("t1.tif", f1, "image/tiff"),
                "image_secondary": ("t2.tif", f2, "image/tiff"),
            },
        )
    assert res.status_code == 200
    data = res.json()
    assert data["task"] == "bitemporal_change_detection"
    assert data["parameters"]["candidate_model"] in ("cva", "CVA Baseline")


# ===========================================================================
# 16. AMBIGUOUS OR IMPOSSIBLE COMBINATIONS RETURN STRUCTURED FAILURE
# ===========================================================================
def test_ambiguous_or_impossible_combinations_fail_cleanly():
    """Verify impossible task/input combination yields HTTP 400 with actionable explanation."""
    with open(OPTICAL_PATH, "rb") as f:
        res = client.post(
            "/api/v1/analyze",
            data={"query": "What changed between these two dates?"},
            files={"image_primary": ("single_opt.tif", f, "image/tiff")},
        )
    assert res.status_code == 400
    assert "requires both" in res.json()["detail"]
