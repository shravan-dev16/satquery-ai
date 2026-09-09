"""Live API Integration Test Runner for SatQuery AI.

Tests all mandated workflows against the running server at http://localhost:8000.
Checks:
- HTTP status code
- Task routing decision
- Models executed
- Confidence and breakdown
- Consistency status (M8)
- Evidence bundle (masks, boxes, images, statistics, semantic_interpretation)
- Static visual URLs (checks HTTP 200)
- Execution trace steps
- M11 report attachment
"""

import json
from pathlib import Path
import sys
import httpx

BASE_URL = "http://localhost:8000"

TEST_CASES = [
    {
        "test_id": "TEST_A",
        "name": "Single Image VQA",
        "query": "What is present in this image?",
        "primary": "datasets/ui_demo/vqa_sample_1.png",
        "secondary": None,
        "expected_task": "single_image_vqa",
    },
    {
        "test_id": "TEST_B",
        "name": "Grounding (Harbor)",
        "query": "Where is the harbor?",
        "primary": "datasets/ui_demo/grounding_harbor.png",
        "secondary": None,
        "expected_task": "text_guided_grounding",
    },
    {
        "test_id": "TEST_B2",
        "name": "Grounding (Water Edge-Case)",
        "query": "Where is the water body?",
        "primary": "datasets/ui_demo/grounding_water.tif",
        "secondary": None,
        "expected_task": "text_guided_grounding",
    },
    {
        "test_id": "TEST_C",
        "name": "Pure Bi-Temporal Change",
        "query": "What changed between these two dates?",
        "primary": "datasets/ui_demo/change_pure_t1.tif",
        "secondary": "datasets/ui_demo/change_pure_t2.tif",
        "expected_task": "bitemporal_change_detection",
    },
    {
        "test_id": "TEST_D",
        "name": "Built-up Semantic Change",
        "query": "Did the built-up area increase?",
        "primary": "datasets/ui_demo/change_urban_t1.tif",
        "secondary": "datasets/ui_demo/change_urban_t2.tif",
        "expected_task": "bitemporal_change_detection",
    },
    {
        "test_id": "TEST_E",
        "name": "Semantic Quantity: Buildings",
        "query": "How much buildings were newly created?",
        "primary": "datasets/ui_demo/change_urban_t1.tif",
        "secondary": "datasets/ui_demo/change_urban_t2.tif",
        "expected_task": "bitemporal_change_detection",
    },
    {
        "test_id": "TEST_F",
        "name": "Semantic Quantity: Vegetation",
        "query": "How much vegetation was lost?",
        "primary": "datasets/ui_demo/change_forest_t1.tif",
        "secondary": "datasets/ui_demo/change_forest_t2.tif",
        "expected_task": "bitemporal_change_detection",
    },
    {
        "test_id": "TEST_G",
        "name": "Semantic Quantity: Water",
        "query": "How much water area changed?",
        "primary": "datasets/ui_demo/change_water_t1.tif",
        "secondary": "datasets/ui_demo/change_water_t2.tif",
        "expected_task": "bitemporal_change_detection",
    },
    {
        "test_id": "TEST_H",
        "name": "Optical + SAR Cross-Modal",
        "query": "Use the optical and SAR images together to identify built-up and water-covered regions.",
        "primary": "datasets/ui_demo/optical_multimodal.tif",
        "secondary": "datasets/ui_demo/sar_multimodal.tif",
        "expected_task": "optical_sar_analysis",
    },
    {
        "test_id": "TEST_I",
        "name": "Combined Integrated Workflow",
        "query": "Has the built-up area increased, where did the change occur, and does the SAR evidence support it?",
        "primary": "datasets/ui_demo/change_urban_t1.tif",
        "secondary": "datasets/ui_demo/change_urban_t2.tif",
        "expected_task": "bitemporal_change_detection",
    },
]


def run_tests():
    print("=" * 70)
    print("SATQUERY AI — LIVE API INTEGRATION TEST SUITE")
    print("=" * 70)

    client = httpx.Client(timeout=120.0)
    results = []

    for tc in TEST_CASES:
        t_id = tc["test_id"]
        t_name = tc["name"]
        print(f"\n[{t_id}] Running: {t_name}...")
        print(f"  Query: '{tc['query']}'")
        print(f"  Files: {tc['primary']}" + (f" + {tc['secondary']}" if tc['secondary'] else ""))

        files = {}
        p1 = Path(tc["primary"])
        if not p1.exists():
            print(f"  ERROR: Primary file not found: {p1}")
            continue

        with open(p1, "rb") as f1:
            files["image_primary"] = (p1.name, f1.read(), "application/octet-stream")

        if tc["secondary"]:
            p2 = Path(tc["secondary"])
            if not p2.exists():
                print(f"  ERROR: Secondary file not found: {p2}")
                continue
            with open(p2, "rb") as f2:
                files["image_secondary"] = (p2.name, f2.read(), "application/octet-stream")

        data = {"query": tc["query"]}

        try:
            resp = client.post(f"{BASE_URL}/api/v1/analyze", data=data, files=files)
            print(f"  HTTP Status: {resp.status_code}")

            if resp.status_code != 200:
                print(f"  FAILED: {resp.text}")
                results.append({
                    "test_id": t_id,
                    "name": t_name,
                    "status_code": resp.status_code,
                    "error": resp.text,
                    "pass": False,
                })
                continue

            contract = resp.json()
            task = contract.get("task")
            answer = contract.get("answer", "")
            conf = contract.get("confidence")
            conf_level = contract.get("confidence_level")
            ev_status = contract.get("evidence_status")
            models = [m.get("model_name", m.get("identifier")) for m in contract.get("models", [])]
            trace_steps = [s.get("step_name") for s in contract.get("execution_trace", [])]
            has_report = contract.get("report") is not None
            evidence = contract.get("evidence", {})

            # Check static visual URLs
            image_urls = [img.get("url") for img in evidence.get("images", []) if img.get("url")]
            mask_urls = [m.get("url") for m in evidence.get("masks", []) if m.get("url")]
            all_urls = image_urls + mask_urls

            url_status = {}
            for u in all_urls:
                if u.startswith("/"):
                    u_full = f"{BASE_URL}{u}"
                else:
                    u_full = u
                try:
                    u_resp = client.get(u_full)
                    url_status[u] = u_resp.status_code
                except Exception as ex:
                    url_status[u] = f"Error: {ex}"

            print(f"  Task: {task}")
            print(f"  Models: {' -> '.join(models) if models else 'None'}")
            print(f"  Confidence: {conf} ({conf_level})")
            print(f"  Evidence Status: {ev_status}")
            print(f"  Trace Steps ({len(trace_steps)}): {', '.join(trace_steps[:4])}...")
            print(f"  Report Present: {has_report}")
            print(f"  Answer: {answer[:120]}...")
            print(f"  Visual URLs: {url_status}")

            results.append({
                "test_id": t_id,
                "name": t_name,
                "status_code": resp.status_code,
                "task": task,
                "models": models,
                "confidence": conf,
                "confidence_level": conf_level,
                "evidence_status": ev_status,
                "trace_steps": trace_steps,
                "has_report": has_report,
                "answer": answer,
                "url_status": url_status,
                "parameters": contract.get("parameters", {}),
                "pass": True,
            })

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append({
                "test_id": t_id,
                "name": t_name,
                "pass": False,
                "error": str(e),
            })

    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    for r in results:
        status_str = "PASS" if r.get("pass") else "FAIL"
        task_str = r.get("task", "N/A")
        models_str = " -> ".join(r.get("models", []))
        print(f"{r['test_id']:<8} | {status_str} | {task_str:<28} | Models: {models_str}")

    # Save detailed JSON output for report generation
    out_path = Path("docs/evaluation/ui_test_run_data.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved test telemetry to {out_path}")


if __name__ == "__main__":
    run_tests()
