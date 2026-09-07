"""Milestone M3 Bi-Temporal Change Detection Smoke Test.

Executes end-to-end bi-temporal change detection on the georeferenced synthetic fixture:
- T1: tests/fixtures/bitemporal/time1_pre_change.tif
- T2: tests/fixtures/bitemporal/time2_post_change.tif

Verifies:
1. Ingestion & Alignment
2. Execution of ChangeDetectionSpecialist
3. Recovery of the known synthetic change region (80x80 px at [90, 90, 170, 170])
4. Correct physical area computation in m^2 and ha (EPSG:32643, 10m/px -> 640,000 m^2 / 64 ha)
5. Generation of binary mask and red overlay preview PNGs
6. StandardResultContract generation and 9-step execution trace via FastAPI analyze API
7. Saves output to docs/SMOKE_TEST_CHANGE_RESULTS.json
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np
from PIL import Image
from starlette.testclient import TestClient

from backend.agent.schema import SpecialistInput, TaskType
from backend.main import app
from backend.models.change import ChangeDetectionSpecialist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test_change")


def run_smoke_test():
    t1_path = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
    t2_path = Path("tests/fixtures/bitemporal/time2_post_change.tif")

    assert t1_path.exists(), f"Missing fixture: {t1_path}"
    assert t2_path.exists(), f"Missing fixture: {t2_path}"

    logger.info("1. Executing ChangeDetectionSpecialist direct inference...")
    specialist = ChangeDetectionSpecialist(candidate="cva")
    specialist.load()

    spec_input = SpecialistInput(
        task=TaskType.CHANGE_DETECTION,
        query="Identify physical surface changes between these two satellite observations.",
        primary_image_path=str(t1_path),
        secondary_image_path=str(t2_path),
        parameters={"candidate": "cva"},
        metadata={"crs": "EPSG:32643"},
    )

    t0 = time.perf_counter()
    output = specialist.predict(spec_input)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    logger.info("Direct inference completed in %.1f ms. Success: %s", elapsed_ms, output.success)
    assert output.success, "Specialist execution failed!"
    assert output.evidence.masks, "No mask returned!"
    assert output.evidence.boxes, "No bounding boxes returned!"

    params = output.parameters_used
    changed_px = params["changed_pixels"]
    logger.info("Detected changed pixels: %d (Expected ground truth: ~6400 px)", changed_px)
    assert changed_px > 5000, f"Detected pixels ({changed_px}) significantly deviates from true block (6400)!"

    # Verify physical area stats
    stats_dict = {s.metric_name: s.value for s in output.evidence.statistics}
    logger.info("Extracted statistics: %s", stats_dict)
    assert "changed_area_m2" in stats_dict, "Missing changed_area_m2 in projected raster!"
    assert "changed_area_hectares" in stats_dict, "Missing changed_area_hectares!"

    area_m2 = stats_dict["changed_area_m2"]
    area_ha = stats_dict["changed_area_hectares"]
    logger.info("Physical Area: %.1f m^2 (%.2f ha)", area_m2, area_ha)
    assert abs(area_ha - 64.0) < 5.0, f"Area in hectares ({area_ha}) deviated unexpectedly from 64 ha!"

    # Verify bounding box coordinates [xmin, ymin, xmax, ymax]
    main_box = output.evidence.boxes[0]
    bx = main_box.coordinates_pixel
    logger.info("Detected main bounding box [xmin, ymin, xmax, ymax]: %s", bx)
    # True box is [90, 90, 169, 169]
    assert 85 <= bx[0] <= 95 and 85 <= bx[1] <= 95, f"Box minimum ({bx[:2]}) misaligned with true block!"
    assert 165 <= bx[2] <= 175 and 165 <= bx[3] <= 175, f"Box maximum ({bx[2:]}) misaligned with true block!"

    # 2. Test FastAPI End-to-End Endpoint
    logger.info("2. Testing FastAPI POST /api/v1/analyze with bi-temporal images...")
    client = TestClient(app)

    with open(t1_path, "rb") as f1, open(t2_path, "rb") as f2:
        response = client.post(
            "/api/v1/analyze",
            data={"query": "What changed between these two dates?"},
            files={
                "image_primary": ("time1.tif", f1, "image/tiff"),
                "image_secondary": ("time2.tif", f2, "image/tiff"),
            },
        )

    logger.info("API Response status code: %d", response.status_code)
    assert response.status_code == 200, f"API failed with {response.status_code}: {response.text}"

    api_data = response.json()
    assert api_data["task"] == "bitemporal_change_detection"
    assert api_data["status"] == "success"
    assert "execution_trace" in api_data
    trace_step_names = [s["step_name"] for s in api_data["execution_trace"]]
    logger.info("Execution trace steps: %s", trace_step_names)

    expected_steps = [
        "InputValidation",
        "TemporalValidation",
        "SpatialCompatibility",
        "Alignment",
        "SpecialistSelection",
        "ModelExecution",
        "ChangeMaskValidation",
        "Statistics",
        "EvidenceAssembly",
    ]
    for step in expected_steps:
        assert step in trace_step_names, f"Missing execution trace step: {step}"

    # 3. Save Smoke Test Results
    result_record = {
        "status": "PASSED",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_type": "synthetic_pipeline_smoke_test",
        "input_t1": str(t1_path),
        "input_t2": str(t2_path),
        "known_change_region": {
            "bbox_pixel": [90, 90, 169, 169],
            "expected_pixels": 6400,
            "expected_area_m2": 640000.0,
            "expected_area_hectares": 64.0,
        },
        "measured_results": {
            "detected_pixels": changed_px,
            "detected_area_m2": area_m2,
            "detected_area_ha": area_ha,
            "detected_bbox_pixel": list(bx),
            "geojson": main_box.geojson,
            "confidence": api_data["confidence"],
            "execution_time_ms": api_data["execution_time_ms"],
        },
        "execution_trace": trace_step_names,
        "preview_artifacts": {
            "mask_url": [m["url"] for m in api_data["evidence"]["masks"]],
            "overlay_url": [img["url"] for img in api_data["evidence"]["images"] if img["role"] == "change_overlay"],
        },
    }

    out_file = Path("docs/SMOKE_TEST_CHANGE_RESULTS.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)

    logger.info("Saved smoke test results to: %s", out_file)
    logger.info("ALL SMOKE TEST ASSERTIONS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_smoke_test()
