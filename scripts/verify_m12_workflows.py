"""Verification script for Milestone M12: Full UI + Product Integration.

Tests all 12 operational workflows against the live FastAPI application:
1. Single-image VQA (RS_VQA)
2. Text-guided region grounding (RS_GROUND)
3. Pure bi-temporal change detection (CHANGE_DETECT)
4. Building semantic change (CHANGE_DETECT -> CHANGE_VQA)
5. Semantic quantity query (Limitation-aware attribution)
6. Vegetation canopy loss (Forest change)
7. Water reservoir change (Water expansion)
8. Optical + SAR cross-modal fusion (OPTICAL_SAR_FUSION)
9. PNG-to-PNG bi-temporal change
10. JPEG-to-JPEG compressed bi-temporal change
11. GeoTIFF-to-GeoTIFF bi-temporal change (Metric hectares / m²)
12. Misaligned / risk test pair (Reduced confidence & warnings)
"""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

from backend.main import app
from backend.reports import generate_markdown_report

client = TestClient(app)

WORKFLOWS = [
    {
        "id": "wf-01-vqa",
        "name": "Single-Image VQA (VRSBench)",
        "p1": "datasets/ui_demo/vqa_sample_1.png",
        "p2": None,
        "query": "What is present in this image?",
        "expected_task_substr": "vqa",
        "expect_metric_area": False,
    },
    {
        "id": "wf-02-grounding",
        "name": "Text-Guided Region Grounding",
        "p1": "datasets/ui_demo/grounding_harbor.png",
        "p2": None,
        "query": "Where is the harbor?",
        "expected_task_substr": "grounding",
        "expect_metric_area": False,
    },
    {
        "id": "wf-03-change-pure",
        "name": "Pure Bi-Temporal Change Detection",
        "p1": "datasets/ui_demo/change_pure_t1.tif",
        "p2": "datasets/ui_demo/change_pure_t2.tif",
        "query": "What changed between these two dates?",
        "expected_task_substr": "change_detection",
        "expect_metric_area": True,
    },
    {
        "id": "wf-04-change-urban",
        "name": "Building Semantic Change",
        "p1": "datasets/ui_demo/change_urban_t1.tif",
        "p2": "datasets/ui_demo/change_urban_t2.tif",
        "query": "Did the built-up area increase?",
        "expected_task_substr": "change_vqa",
        "expect_metric_area": True,
    },
    {
        "id": "wf-05-change-quantity",
        "name": "Semantic Quantity Attribution Check",
        "p1": "datasets/ui_demo/change_urban_t1.tif",
        "p2": "datasets/ui_demo/change_urban_t2.tif",
        "query": "How much buildings were newly created?",
        "expected_task_substr": "change_vqa",
        "expect_metric_area": True,
    },
    {
        "id": "wf-06-change-vegetation",
        "name": "Vegetation Canopy Loss",
        "p1": "datasets/ui_demo/change_forest_t1.tif",
        "p2": "datasets/ui_demo/change_forest_t2.tif",
        "query": "How much vegetation was lost?",
        "expected_task_substr": "change_vqa",
        "expect_metric_area": True,
    },
    {
        "id": "wf-07-change-water",
        "name": "Water Reservoir Change",
        "p1": "datasets/ui_demo/change_water_t1.tif",
        "p2": "datasets/ui_demo/change_water_t2.tif",
        "query": "How much water area changed?",
        "expected_task_substr": "change_vqa",
        "expect_metric_area": True,
    },
    {
        "id": "wf-08-optical-sar",
        "name": "Optical + SAR Cross-Modal Fusion",
        "p1": "datasets/ui_demo/optical_multimodal.tif",
        "p2": "datasets/ui_demo/sar_multimodal.tif",
        "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "expected_task_substr": "optical_sar",
        "expect_metric_area": False,
    },
    {
        "id": "wf-09-png-bitemporal",
        "name": "PNG-to-PNG Bi-Temporal Change",
        "p1": "datasets/ui_demo/change_png_t1.png",
        "p2": "datasets/ui_demo/change_png_t2.png",
        "query": "What changed between these two dates?",
        "expected_task_substr": "change",
        "expect_metric_area": False,  # PNG has no CRS -> pixel space only
    },
    {
        "id": "wf-10-jpeg-bitemporal",
        "name": "JPEG Compressed Bi-Temporal Change",
        "p1": "datasets/ui_demo/change_jpeg_t1.jpg",
        "p2": "datasets/ui_demo/change_jpeg_t2.jpg",
        "query": "What changed between these two dates?",
        "expected_task_substr": "change",
        "expect_metric_area": False,  # JPEG has no CRS -> pixel space only
    },
    {
        "id": "wf-11-geotiff-bitemporal",
        "name": "GeoTIFF Bi-Temporal (Metric Area)",
        "p1": "tests/fixtures/bitemporal/time1_pre_change.tif",
        "p2": "tests/fixtures/bitemporal/time2_post_change.tif",
        "query": "What changed between these two dates?",
        "expected_task_substr": "change",
        "expect_metric_area": True,  # Has projected CRS -> ha and m²
    },
    {
        "id": "wf-12-misaligned-risk",
        "name": "Misaligned / Risk Test Pair",
        "p1": "datasets/ui_demo/misaligned_t1.png",
        "p2": "datasets/ui_demo/misaligned_t2.png",
        "query": "What changed between these two dates?",
        "expected_task_substr": "change",
        "expect_metric_area": False,
        "expect_reduced_confidence_or_warning": True,
    },
]


def run_all_workflows():
    print("=" * 70)
    print("SATQUERY AI — M12 COMPREHENSIVE WORKFLOW VERIFICATION SUITE")
    print("=" * 70)

    results = []
    total_start = time.perf_counter()

    for idx, wf in enumerate(WORKFLOWS, 1):
        print(f"\n[{idx}/12] Executing: {wf['name']} ({wf['id']})...")
        p1_path = Path(wf["p1"])
        if not p1_path.exists():
            print(f"  ❌ FAILED: Primary image not found: {p1_path}")
            sys.exit(1)

        files = {"image_primary": (p1_path.name, open(p1_path, "rb"), "image/tiff" if p1_path.suffix.startswith(".tif") else "image/png")}

        if wf["p2"]:
            p2_path = Path(wf["p2"])
            if not p2_path.exists():
                print(f"  ❌ FAILED: Secondary image not found: {p2_path}")
                sys.exit(1)
            files["image_secondary"] = (p2_path.name, open(p2_path, "rb"), "image/tiff" if p2_path.suffix.startswith(".tif") else "image/png")

        data = {"query": wf["query"]}

        t0 = time.perf_counter()
        resp = client.post("/api/v1/analyze", data=data, files=files)
        elapsed = time.perf_counter() - t0

        # Close file handles
        for f in files.values():
            f[1].close()

        if resp.status_code != 200:
            print(f"  ❌ HTTP {resp.status_code}: {resp.text}")
            sys.exit(1)

        body = resp.json()
        task = body.get("task", "")
        answer = body.get("answer", "")
        conf = body.get("confidence", 0.0)
        conf_level = body.get("confidence_level", "")
        warnings = body.get("warnings", [])
        trace = body.get("execution_trace", [])
        report = body.get("report", None)
        params = body.get("parameters", {})
        evidence = body.get("evidence", {})

        # Assertions
        assert wf["expected_task_substr"] in task.lower(), f"Task mismatch: expected {wf['expected_task_substr']}, got {task}"
        assert answer and len(answer) > 5, "Empty answer returned"
        assert 0.0 <= conf <= 1.0, f"Invalid confidence: {conf}"
        assert conf_level in ("HIGH", "MEDIUM", "LOW", "UNSUPPORTED"), f"Invalid level: {conf_level}"
        assert len(trace) > 0, "No trace steps returned"
        assert report is not None, "Report missing from contract"

        # Check path sanitization: no C:\ or /home/ in answer
        assert "C:\\Users" not in answer, f"Server path leaked in answer: {answer}"
        assert "/home/" not in answer, f"Server path leaked in answer: {answer}"

        # Check metric area behavior
        stat_ha = next((s for s in evidence.get("statistics", []) if s.get("metric_name") == "changed_area_hectares"), None)
        stat_m2 = next((s for s in evidence.get("statistics", []) if s.get("metric_name") == "changed_area_m2"), None)
        if wf["expect_metric_area"]:
            assert stat_ha and stat_ha.get("value") is not None, f"Expected metric hectares for {wf['id']}"
            assert stat_m2 and stat_m2.get("value") is not None, f"Expected metric m2 for {wf['id']}"
        else:
            # Should be null or absent
            val_ha = stat_ha.get("value") if stat_ha else None
            assert val_ha is None, f"Fabricated hectares detected on non-georeferenced input: {val_ha}"

        # For misaligned pair, check that warnings or reduced confidence is emitted
        if wf.get("expect_reduced_confidence_or_warning"):
            has_warn = len(warnings) > 0 or len(body.get("confidence_breakdown", {}).get("confidence_warnings", [])) > 0
            has_reduced_conf = conf < 0.90
            assert has_warn or has_reduced_conf, f"Misaligned pair did not trigger warning or reduced confidence! Conf={conf}"

        # Markdown report generation verification
        from backend.reports.schema import AnalystReport
        analyst_rpt = AnalystReport.model_validate(report)
        md_text = generate_markdown_report(analyst_rpt)
        assert len(md_text) > 100, "Generated markdown report is too short"
        assert "# SatQuery AI" in md_text, "Markdown report header missing"

        print(f"  [OK] Route: {task}")
        print(f"  [OK] Answer: {answer[:80]}...")
        print(f"  [OK] Confidence: {conf_level} ({conf:.2f})")
        print(f"  [OK] Trace Steps: {len(trace)} steps ({elapsed*1000:.1f} ms)")
        if warnings:
            print(f"  [OK] Advisories ({len(warnings)}): {warnings[0][:60]}...")
        if stat_ha and stat_ha.get("value") is not None:
            print(f"  [OK] Metric Area: {stat_ha['value']:.2f} ha ({stat_m2['value']:.0f} m²)")
        else:
            print(f"  [OK] Metric Area: Unmeasured (pixel-space only)")

        results.append({
            "id": wf["id"],
            "name": wf["name"],
            "task": task,
            "confidence": conf,
            "confidence_level": conf_level,
            "latency_ms": int(elapsed * 1000),
            "status": "PASSED",
        })

    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 70)
    print(f"ALL 12 OPERATIONAL WORKFLOWS PASSED IN {total_time:.2f}s")
    print("=" * 70)

    for r in results:
        print(f"[{r['status']}] {r['id']:<24} {r['name']:<36} Route={r['task']:<28} Conf={r['confidence_level']} ({r['confidence']:.2f}) Latency={r['latency_ms']}ms")

    return results


if __name__ == "__main__":
    run_all_workflows()
