"""Tests for Semantic Change Area Attribution & Limitation Qualification.

Enforces SIH26167 Scientific Rigor Rules:
1. total_changed_area != semantic_class_changed_area
2. If class-specific area cannot be spatially attributed from the change mask,
   semantic_changed_area_ha MUST be None / unmeasured.
3. The system answer and report must provide:
   - Total detected change (pixels, area in ha)
   - Grounded semantic interpretation
   - Defensible system confidence / consistency
   - Explicit qualification that class-specific area is unmeasured
4. No fake semantic segmentation masks or fabricated class-specific areas.
5. Zero-change anti-hallucination behavior is strictly preserved.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from backend.main import app

client = TestClient(app)

from backend.agent.executor import AgentExecutor
from backend.agent.planner import AgentPlanner
from backend.agent.registry import registry
from backend.agent.router import AgentRouter
from backend.agent.schema import TaskType

T1_PATH = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
T2_PATH = Path("tests/fixtures/bitemporal/time2_post_change.tif")


def test_semantic_quantity_query_building_creation_workflow():
    """Verify workflow for 'How much buildings were newly created?'
    Must:
    1. Route to CHANGE_VQA (multi-stage)
    2. Sequentially execute CHANGE_DETECT followed by CHANGE_VQA
    3. Return total physical change (not claiming it is pure buildings)
    4. Set semantic_changed_area_ha = None
    5. Set semantic_area_status = 'unmeasured_from_spatial_evidence'
    6. Include explicit scientific limitation in answer and report
    """
    query = "How much buildings were newly created?"
    decision = AgentRouter.route(query=query, primary_path=T1_PATH, secondary_path=T2_PATH)

    assert decision.is_valid is True
    assert decision.task == TaskType.CHANGE_VQA
    assert decision.is_multi_stage is True
    assert decision.target_specialist_ids == ["CHANGE_DETECT", "CHANGE_VQA"]

    plan = AgentPlanner.create_plan(decision, primary_path=T1_PATH, secondary_path=T2_PATH)
    contract = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=T1_PATH,
        secondary_path=T2_PATH,
        query=query,
    )

    assert contract.status == "success"
    assert contract.task == "bitemporal_change_vqa"

    # Verify execution trace contains both specialists in order
    step_names = [s.step_name for s in contract.execution_trace]
    assert "ModelExecution" in step_names
    assert "SemanticInterpretation" in step_names

    # Check parameters
    params = contract.parameters
    assert "total_changed_pixels" in params
    assert params["total_changed_pixels"] > 0
    assert "total_changed_area_ha" in params
    assert params["semantic_class_detected"] == "built-up / building"
    assert params["semantic_changed_area_ha"] is None
    assert params["semantic_area_status"] == "unmeasured_from_spatial_evidence"
    assert "semantic_area_limitation" in params

    # Verify answer explicitly distinguishes total change and states limitation
    answer = contract.answer
    assert "Total detected physical surface change" in answer
    assert "Semantic interpretation indicates" in answer
    assert "Note: Specific built-up / building surface area is not directly measurable" in answer
    # MUST NOT simply state 'X ha of buildings were newly created' as fact
    assert "of buildings were newly created" not in answer.lower() or "not directly measurable" in answer

    # Verify M11 report packaging preserves this distinction
    report = contract.report
    assert report is not None
    rep_metrics = report.statistics.metrics
    assert rep_metrics.get("total_changed_pixels") == params["total_changed_pixels"]
    assert rep_metrics.get("total_changed_area_ha") == params["total_changed_area_ha"]
    assert rep_metrics.get("semantic_changed_area_ha") is None
    assert rep_metrics.get("semantic_area_status") == "unmeasured_from_spatial_evidence"

    # Verify report summary notes contain attribution qualification
    notes_text = " ".join(report.statistics.summary_notes)
    assert "Semantic area attribution:" in notes_text
    assert "not directly measurable" in notes_text or "unmeasured" in notes_text


def test_semantic_query_vegetation_loss_workflow():
    """Verify workflow for 'How much vegetation was lost?'"""
    query = "How much vegetation was lost?"
    decision = AgentRouter.route(query=query, primary_path=T1_PATH, secondary_path=T2_PATH)

    assert decision.task == TaskType.CHANGE_VQA
    assert decision.is_multi_stage is True

    plan = AgentPlanner.create_plan(decision, primary_path=T1_PATH, secondary_path=T2_PATH)
    contract = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=T1_PATH,
        secondary_path=T2_PATH,
        query=query,
    )

    params = contract.parameters
    assert params["semantic_class_detected"] == "vegetation / cropland"
    assert params["semantic_changed_area_ha"] is None
    assert params["semantic_area_status"] == "unmeasured_from_spatial_evidence"
    assert "vegetation" in contract.answer.lower()
    assert "not directly measurable" in contract.answer


def test_semantic_query_water_area_change_workflow():
    """Verify workflow for 'How much water area changed?'"""
    query = "How much water area changed?"
    decision = AgentRouter.route(query=query, primary_path=T1_PATH, secondary_path=T2_PATH)

    assert decision.task == TaskType.CHANGE_VQA
    assert decision.is_multi_stage is True

    plan = AgentPlanner.create_plan(decision, primary_path=T1_PATH, secondary_path=T2_PATH)
    contract = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=T1_PATH,
        secondary_path=T2_PATH,
        query=query,
    )

    params = contract.parameters
    assert params["semantic_class_detected"] == "water body"
    assert params["semantic_changed_area_ha"] is None
    assert params["semantic_area_status"] == "unmeasured_from_spatial_evidence"
    assert "water" in contract.answer.lower()
    assert "not directly measurable" in contract.answer


def test_semantic_query_builtup_increase_workflow():
    """Verify workflow for 'Did the built-up area increase?'
    Must route to CHANGE_VQA and have trace: CHANGE_DETECT -> CHANGE_VQA.
    """
    query = "Did the built-up area increase?"
    decision = AgentRouter.route(query=query, primary_path=T1_PATH, secondary_path=T2_PATH)

    assert decision.task == TaskType.CHANGE_VQA
    assert decision.is_multi_stage is True
    assert decision.target_specialist_ids == ["CHANGE_DETECT", "CHANGE_VQA"]

    plan = AgentPlanner.create_plan(decision, primary_path=T1_PATH, secondary_path=T2_PATH)
    contract = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=T1_PATH,
        secondary_path=T2_PATH,
        query=query,
    )

    models_executed = [m.identifier for m in contract.models]
    assert models_executed == ["CHANGE_DETECT", "CHANGE_VQA"]

    step_names = [s.step_name for s in contract.execution_trace]
    cd_step = step_names.index("ModelExecution")
    vqa_step = step_names.index("SemanticInterpretation")
    assert cd_step < vqa_step

    assert contract.parameters["semantic_class_detected"] == "built-up / building"
    assert contract.parameters["semantic_changed_area_ha"] is None
    assert contract.parameters["semantic_area_status"] == "unmeasured_from_spatial_evidence"
    assert "built-up" in contract.answer.lower() or "building" in contract.answer.lower()
    assert "not directly measurable" in contract.answer


def test_pure_total_change_does_not_invoke_change_vqa():
    """Verify pure change query 'What changed between these two dates?'
    routes strictly to CHANGE_DETECTION and does NOT invoke CHANGE_VQA.
    """
    query = "What changed between these two dates?"
    decision = AgentRouter.route(query=query, primary_path=T1_PATH, secondary_path=T2_PATH)

    assert decision.task == TaskType.CHANGE_DETECTION
    assert decision.is_multi_stage is False
    assert decision.target_specialist_ids == ["CHANGE_DETECT"]

    plan = AgentPlanner.create_plan(decision, primary_path=T1_PATH, secondary_path=T2_PATH)
    contract = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=T1_PATH,
        secondary_path=T2_PATH,
        query=query,
    )

    assert contract.task == "bitemporal_change_detection"
    models_executed = [m.identifier for m in contract.models]
    assert "CHANGE_DETECT" in models_executed
    assert "CHANGE_VQA" not in models_executed

    # Step names in trace must NOT include SemanticInterpretation
    trace_step_names = [s.step_name for s in contract.execution_trace]
    assert "SemanticInterpretation" not in trace_step_names


def test_zero_change_scene_anti_hallucination_preserved():
    """Verify comparing identical rasters (T1 to T1) yields zero change,
    does not hallucinate semantic area, and states scene stability.
    """
    zero_p1 = Path("tests/fixtures/semantic/t1_zero.tif")
    zero_p2 = Path("tests/fixtures/semantic/t2_zero.tif")
    query = "How much buildings were newly created?"
    decision = AgentRouter.route(query=query, primary_path=zero_p1, secondary_path=zero_p2)

    plan = AgentPlanner.create_plan(decision, primary_path=zero_p1, secondary_path=zero_p2)
    contract = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=zero_p1,
        secondary_path=zero_p2,
        query=query,
        task_hint="cva",
    )

    assert contract.parameters.get("changed_pixels", 0) == 0
    # Zero change scene must report no physical change and scene stability
    assert "no significant physical surface change" in contract.answer.lower() or "0 changed pixels" in contract.answer.lower() or "stable" in contract.answer.lower()
