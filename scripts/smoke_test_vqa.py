"""Real Remote-Sensing VQA Model Smoke Test & GPU Memory Measurement.

Executes a live end-to-end smoke test on a legitimate real satellite image:
real GeoTIFF -> preprocessing -> VQA specialist -> structured answer -> StandardResultContract.

Records:
- Model initialization time
- Peak GPU allocated & reserved VRAM
- Inference latency
- Memory post-cleanup
"""

import gc
import json
import sys
import time
from pathlib import Path

# Ensure workspace root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from backend.agent.schema import SpecialistInput, TaskType
from backend.models.vqa import RemoteSensingVQASpecialist
from backend.preprocessing.geotiff import GeoTIFFReader


def run_smoke_test():
    real_image_path = Path("tests/fixtures/real_rs_sample.tif")
    if not real_image_path.exists():
        raise FileNotFoundError(f"Real remote-sensing sample not found at: {real_image_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==================================================")
    print(f"SATQUERY AI — M1 REAL VQA SMOKE TEST")
    print(f"Compute Device: {device}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    print(f"==================================================")

    # 1. Measure Model Initialization
    init_start = time.perf_counter()
    specialist = RemoteSensingVQASpecialist(device=device)
    specialist.load()
    init_duration_ms = int((time.perf_counter() - init_start) * 1000)

    mem_after_load_mb = 0.0
    if device == "cuda":
        mem_after_load_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        res_after_load_mb = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"Model Init Time: {init_duration_ms} ms")
        print(f"VRAM Allocated after load: {mem_after_load_mb:.2f} MB")
        print(f"VRAM Reserved after load: {res_after_load_mb:.2f} MB")

    # 2. Inspect real image metadata
    raw_meta = GeoTIFFReader.inspect(real_image_path)
    print(f"\nReal Image Sample: {real_image_path.name}")
    print(f"  Dimensions: {raw_meta['width']} x {raw_meta['height']}")
    print(f"  Bands: {raw_meta['band_count']}")
    print(f"  CRS: {raw_meta['crs']}")
    print(f"  Bounds: {raw_meta['bounds']}")

    # 3. Execute VQA Inference
    query = "Describe the prominent geographical features, coastlines, and land use visible in this satellite imagery."
    print(f"\nQuery: '{query}'")

    inputs = SpecialistInput(
        task=TaskType.VQA,
        query=query,
        primary_image_path=str(real_image_path),
        parameters={"temperature": 0.1, "max_new_tokens": 128},
        metadata=raw_meta,
    )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    inf_start = time.perf_counter()
    output = specialist.predict(inputs)
    inf_duration_ms = int((time.perf_counter() - inf_start) * 1000)

    peak_alloc_mb = 0.0
    peak_res_mb = 0.0
    if device == "cuda":
        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_res_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)

    print(f"\n--- VQA Specialist Output ---")
    print(f"Answer: {output.answer_text}")
    print(f"Confidence Heuristic: {output.confidence}")
    print(f"Execution Latency: {inf_duration_ms} ms")
    if device == "cuda":
        print(f"Peak VRAM Allocated during inference: {peak_alloc_mb:.2f} MB")
        print(f"Peak VRAM Reserved during inference: {peak_res_mb:.2f} MB")

    # 4. Verify Cleanup
    specialist.cleanup()
    post_cleanup_alloc_mb = 0.0
    if device == "cuda":
        post_cleanup_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"VRAM Allocated after cleanup: {post_cleanup_alloc_mb:.2f} MB (Expected ~0 MB)")

    # 5. Output Summary JSON
    summary = {
        "status": "success",
        "smoke_test_valid": bool(output.answer_text and len(output.answer_text) > 20),
        "model_id": specialist.model_id,
        "specialist_role": "initial VQA specialist candidate using zero-shot Qwen2-VL-2B-Instruct",
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
        "metrics": {
            "initialization_time_ms": init_duration_ms,
            "inference_latency_ms": inf_duration_ms,
            "vram_allocated_after_load_mb": round(mem_after_load_mb, 2),
            "peak_vram_allocated_mb": round(peak_alloc_mb, 2),
            "peak_vram_reserved_mb": round(peak_res_mb, 2),
            "vram_after_cleanup_mb": round(post_cleanup_alloc_mb, 2),
        },
        "query": query,
        "answer": output.answer_text,
        "confidence": output.confidence,
    }

    out_file = Path("docs/SMOKE_TEST_RESULTS.json")
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved smoke test metrics to {out_file}")
    return summary


if __name__ == "__main__":
    run_smoke_test()
