"""Real Remote-Sensing Grounding Model Smoke Test & Multi-Query Evaluation.

Executes a live end-to-end smoke test on the real satellite test image:
real GeoTIFF -> preprocessing -> Grounding specialist (Grounding DINO) -> spatial bounding box -> GeoJSON projection.

Evaluates multiple realistic remote-sensing queries:
- "Where is the water body?" -> "water body."
- "Find the road." -> "road."
- "Locate the built-up area." -> "built-up area."
- "Detect the runway." -> "runway."

Records:
- Image identifier and provenance
- Query & normalized query
- Predicted boxes, scores, and degeneracy flags
- Inference latency & peak VRAM
- Warnings (e.g. absent feature warnings)
- Saves to docs/SMOKE_TEST_GROUNDING_RESULTS.json
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure workspace root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from backend.agent.schema import SpecialistInput, TaskType
from backend.models.grounding import GroundingDinoSpecialist, QueryNormalizer
from backend.preprocessing.geotiff import GeoTIFFReader


def run_grounding_smoke_test():
    real_image_path = Path("tests/fixtures/real_rs_sample.tif")
    if not real_image_path.exists():
        raise FileNotFoundError(f"Real remote-sensing sample not found at: {real_image_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==================================================")
    print("SATQUERY AI — M2.1 GROUNDING DINO SMOKE TEST")
    print(f"Compute Device: {device}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    print("==================================================")

    # 1. Measure Model Initialization
    init_start = time.perf_counter()
    specialist = GroundingDinoSpecialist(device=device)
    specialist.load()
    init_duration_ms = int((time.perf_counter() - init_start) * 1000)

    mem_after_load_mb = 0.0
    res_after_load_mb = 0.0
    if device == "cuda":
        mem_after_load_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        res_after_load_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"Model Init Time: {init_duration_ms} ms")
        print(f"VRAM Allocated after load: {mem_after_load_mb:.2f} MB")
        print(f"VRAM Reserved after load: {res_after_load_mb:.2f} MB")

    # 2. Inspect real image metadata
    raw_meta = GeoTIFFReader.inspect(real_image_path)
    print(f"\nReal Image Sample: {real_image_path.name}")
    print(f"  Provenance: Rasterio repository GeoTIFF test image used for geospatial pipeline smoke testing.")
    print(f"  Dimensions: {raw_meta['width']} x {raw_meta['height']}")
    print(f"  Bands: {raw_meta['band_count']}")
    print(f"  CRS: {raw_meta['crs']}")

    # 3. Test queries required by M2.1
    test_queries = [
        "Where is the water body?",
        "Find the road.",
        "Locate the built-up area.",
        "Detect the runway.",
    ]

    results_per_query: List[Dict[str, Any]] = []

    for q in test_queries:
        norm_q = QueryNormalizer.normalize(q)
        print(f"\n--- Testing Query: '{q}' (Normalized: '{norm_q}') ---")

        inputs = SpecialistInput(
            task=TaskType.GROUNDING,
            query=q,
            primary_image_path=str(real_image_path),
            parameters={"box_threshold": 0.25, "text_threshold": 0.25},
            metadata=raw_meta,
        )

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        output = specialist.predict(inputs)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        peak_alloc_mb = 0.0
        peak_res_mb = 0.0
        if device == "cuda":
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_res_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

        boxes = output.evidence.boxes
        print(f"  Summary: {output.answer_text}")
        print(f"  Confidence: {output.confidence}")
        print(f"  Latency: {latency_ms} ms")
        print(f"  Valid Boxes Count: {len(boxes)}")
        for i, b in enumerate(boxes[:3]):
            print(f"    Box {i+1}: pixel={b.coordinates_pixel}, score={b.model_score}, label='{b.label}', degen={b.is_degenerate}")
        if output.warnings:
            print(f"  Warnings: {output.warnings}")

        q_record = {
            "image_identifier": real_image_path.name,
            "query": q,
            "normalized_query": norm_q,
            "latency_ms": latency_ms,
            "peak_vram_allocated_mb": round(peak_alloc_mb, 2),
            "peak_vram_reserved_mb": round(peak_res_mb, 2),
            "valid_boxes_count": len(boxes),
            "boxes": [
                {
                    "box_id": b.box_id,
                    "label": b.label,
                    "pixel_bbox": list(b.coordinates_pixel),
                    "normalized_bbox": list(b.coordinates_normalized),
                    "model_score": b.model_score,
                    "confidence": b.confidence,
                    "is_degenerate": b.is_degenerate,
                    "degenerate_reason": b.degenerate_reason,
                    "geojson": b.geojson,
                }
                for b in boxes
            ],
            "answer_summary": output.answer_text,
            "warnings": output.warnings,
        }
        results_per_query.append(q_record)

    # 4. Verify Cleanup
    specialist.cleanup()
    post_cleanup_alloc_mb = 0.0
    if device == "cuda":
        post_cleanup_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"\nVRAM Allocated after cleanup: {post_cleanup_alloc_mb:.2f} MB")

    # 5. Output Summary JSON
    summary = {
        "status": "success",
        "model_id": specialist.model_id,
        "specialist_role": "dedicated zero-shot grounding detector candidate (IDEA-Research/grounding-dino-base)",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "image": {
            "name": real_image_path.name,
            "provenance": "Rasterio repository GeoTIFF test image used for geospatial pipeline smoke testing.",
            "source_url": "https://raw.githubusercontent.com/rasterio/rasterio/master/tests/data/RGB.byte.tif",
            "license": "BSD 3-Clause",
            "crs": raw_meta["crs"],
            "dimensions": [raw_meta["width"], raw_meta["height"]],
        },
        "initialization_time_ms": init_duration_ms,
        "vram_allocated_after_load_mb": round(mem_after_load_mb, 2),
        "vram_after_cleanup_mb": round(post_cleanup_alloc_mb, 2),
        "queries_evaluated": results_per_query,
    }

    out_file = Path("docs/SMOKE_TEST_GROUNDING_RESULTS.json")
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved smoke test multi-query results to {out_file}")
    return summary


if __name__ == "__main__":
    run_grounding_smoke_test()
