"""Verification script for Milestone M11: Real End-to-End Analysis Workflows.

Runs 5 real end-to-end workflows through the SatQuery AI agent pipeline
and validates that each produces an auditable, structured AnalystReport.
"""

from pathlib import Path
import sys
import time

from backend.agent import AgentPlanner, AgentRouter, AgentExecutor
from backend.main import register_default_specialists

def run_workflow(name: str, query: str, p1: Path, p2: Path = None):
    print(f"\n{'='*70}")
    print(f"RUNNING WORKFLOW: {name}")
    print(f"Query: \"{query}\"")
    print(f"Primary: {p1}")
    if p2:
        print(f"Secondary: {p2}")
    print(f"{'='*70}")

    start = time.perf_counter()
    decision = AgentRouter.route(
        query=query,
        primary_path=p1,
        secondary_path=p2,
        primary_filename=p1.name,
        secondary_filename=p2.name if p2 else None,
    )
    print(f"-> Routed Task: {decision.task.value} (Intent: {decision.intent_category})")
    print(f"-> Target Specialists: {decision.target_specialist_ids}")

    plan = AgentPlanner.create_plan(
        decision=decision,
        primary_path=p1,
        secondary_path=p2,
        query=query,
    )
    print(f"-> Execution Plan Steps: {[s.step_name for s in plan.steps]}")

    result = AgentExecutor.execute(
        plan=plan,
        decision=decision,
        primary_path=p1,
        secondary_path=p2,
        query=query,
        primary_filename=p1.name,
        secondary_filename=p2.name if p2 else None,
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"-> Execution Completed in {elapsed:.1f} ms. Status: {result.status}")
    print(f"-> Answer: {result.answer[:160]}...")
    print(f"-> System Confidence: {result.confidence:.2f} ({result.confidence_level})")

    # M11 Report Inspection
    rep = result.report
    assert rep is not None, "Report field must not be None!"
    print(f"\n[M11 AUDIT REPORT INSPECTION]")
    print(f"  Report ID: {rep.metadata.report_id}")
    print(f"  Mission: {rep.metadata.mission}")
    print(f"  Input Count: {rep.input_summary.input_count}")
    print(f"  Visual Evidence Items: {len(rep.visual_evidence)}")
    for v in rep.visual_evidence:
        print(f"    - [{v.type}] {v.label}: {v.path_or_url}")
    print(f"  Quantitative Metrics: {rep.statistics.metrics}")
    print(f"  Summary Notes: {rep.statistics.summary_notes}")
    print(f"  Consistency Status: {rep.consistency.status} (Gated: {rep.consistency.is_gated})")
    print(f"  Provenance Ledger Entries: {len(rep.provenance)}")
    for p in rep.provenance:
        print(f"    - Tool: {p.source_specialist} | Model: {p.model_name} | Conf: {p.raw_confidence}")
    print(f"  Execution Trace Steps: {len(rep.execution_trace)}")

    # Verify JSON and Markdown serialization
    json_repr = rep.to_json()
    assert len(json_repr) > 100
    md_repr = rep.to_markdown()
    assert len(md_repr) > 100
    print(f"  Workflow {name}: SUCCESS [OK]")
    return rep

def main():
    register_default_specialists()

    p_optical = Path("tests/fixtures/optical_s2_sample.tif")
    p_t1 = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
    p_t2 = Path("tests/fixtures/bitemporal/time2_post_change.tif")
    p_opt_sar_opt = Path("tests/fixtures/optical_sar/optical_multimodal_01.tif")
    p_opt_sar_sar = Path("tests/fixtures/optical_sar/sar_multimodal_01.tif")

    # 1. Single-Image VQA
    run_workflow(
        name="1. Single-Image VQA",
        query="What type of land cover and structures are visible in this scene?",
        p1=p_optical,
    )

    # 2. Text-Guided Grounding
    run_workflow(
        name="2. Text-Guided Grounding",
        query="Locate and ground the water body or river in this image.",
        p1=p_optical,
    )

    # 3. Bi-Temporal Physical Change Detection
    run_workflow(
        name="3. Bi-Temporal Change Detection",
        query="Detect physical surface changes between T1 and T2.",
        p1=p_t1,
        p2=p_t2,
    )

    # 4. Optical-SAR Joint Analysis
    run_workflow(
        name="4. Optical + SAR Cross-Modal Analysis",
        query="Perform joint optical and SAR analysis to resolve surface features.",
        p1=p_opt_sar_opt,
        p2=p_opt_sar_sar,
    )

    # 5. Combined Multi-Specialist Workflow (Killer Workflow)
    run_workflow(
        name="5. Combined Multi-Specialist Workflow",
        query="Has the built-up area increased, decreased, or remained unchanged between these dates?",
        p1=p_t1,
        p2=p_t2,
    )

    print("\n" + "="*70)
    print("ALL 5 REAL WORKFLOWS VERIFIED SUCCESSFULLY WITH M11 REPORTING! [OK]")
    print("="*70)

if __name__ == "__main__":
    main()
