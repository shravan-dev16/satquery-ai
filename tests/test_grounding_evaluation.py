"""Fast unit tests for grounding evaluation logic, manifest integrity, and query normalization."""

import json
from pathlib import Path
import pytest

from backend.evidence.spatial import PixelBoundingBox
from backend.models.grounding import (
    QueryNormalizer,
    apply_nms,
    calculate_box_iou,
)


def test_iou_calculation_exact_cases():
    """Verify mathematical correctness of Intersection over Union (IoU) calculation."""
    # 1. Identical boxes -> IoU 1.0
    box_a = (10.0, 10.0, 50.0, 50.0)
    assert calculate_box_iou(box_a, box_a) == 1.0

    # 2. Non-overlapping boxes -> IoU 0.0
    box_b = (60.0, 60.0, 100.0, 100.0)
    assert calculate_box_iou(box_a, box_b) == 0.0

    # 3. Known geometric overlap:
    # box1: [0, 0, 20, 20], area = 400
    # box2: [10, 0, 30, 20], area = 400
    # intersection: [10, 0, 20, 20], area = 10 * 20 = 200
    # union: 400 + 400 - 200 = 600
    # IoU: 200 / 600 = 1/3 ~ 0.3333
    box1 = (0.0, 0.0, 20.0, 20.0)
    box2 = (10.0, 0.0, 30.0, 20.0)
    assert calculate_box_iou(box1, box2) == 0.3333

    # 4. Inverted/degenerate input
    box_inv = (50.0, 50.0, 10.0, 10.0)
    assert calculate_box_iou(box_a, box_inv) == 0.0


def test_query_normalization():
    """Verify deterministic conversion of user questions into detector entity phrases."""
    cases = [
        ("Where is the water body?", "water body."),
        ("Locate the runway.", "runway."),
        ("Find the road.", "road."),
        ("Detect an airplane.", "airplane."),
        ("Highlight the storage tank!", "storage tank."),
        ("Show me the harbor", "harbor."),
        ("Can you locate the bridge?", "bridge."),
        ("bridge", "bridge."),
    ]
    for raw, expected in cases:
        normalized = QueryNormalizer.normalize(raw)
        assert normalized == expected, f"Failed for query '{raw}': got '{normalized}', expected '{expected}'"


def test_degenerate_box_detection():
    """Verify degeneracy flags for full-image fallback and zero-area boxes."""
    # Full image box (100% coverage on 500x500 raster)
    full_box = PixelBoundingBox(xmin=0.0, ymin=0.0, xmax=500.0, ymax=500.0)
    is_degen, reason = full_box.check_degeneracy(width=500, height=500, threshold_pct=0.98)
    assert is_degen is True
    assert "full_image" in reason

    # Genuine small box (16% coverage)
    sub_box = PixelBoundingBox(xmin=100.0, ymin=100.0, xmax=300.0, ymax=300.0)
    is_degen, reason = sub_box.check_degeneracy(width=500, height=500, threshold_pct=0.98)
    assert is_degen is False
    assert reason is None

    # Zero-area box
    zero_box = PixelBoundingBox(xmin=50.0, ymin=50.0, xmax=50.0, ymax=100.0)
    is_degen, reason = zero_box.check_degeneracy(width=500, height=500)
    assert is_degen is True
    assert reason == "zero_area"


def test_nms_suppression():
    """Verify duplicate overlapping boxes are suppressed by NMS."""
    box_high = PixelBoundingBox(xmin=10.0, ymin=10.0, xmax=50.0, ymax=50.0, model_score=0.90)
    box_low_overlap = PixelBoundingBox(xmin=12.0, ymin=12.0, xmax=52.0, ymax=52.0, model_score=0.45)
    box_separate = PixelBoundingBox(xmin=100.0, ymin=100.0, xmax=150.0, ymax=150.0, model_score=0.85)

    kept = apply_nms([box_high, box_low_overlap, box_separate], iou_threshold=0.70)
    assert len(kept) == 2
    assert box_high in kept
    assert box_separate in kept
    assert box_low_overlap not in kept


def test_vrsbench_manifest_integrity():
    """Verify deterministic VRSBench evaluation manifest exists and all images are readable."""
    manifest_path = Path("docs/evaluation/vrsbench_grounding_subset.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated yet")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) >= 30, f"Expected at least 30 samples, got {len(data)}"

    for item in data:
        assert "sample_id" in item
        assert "referring_expression" in item
        assert len(item["referring_expression"]) > 5

        # Check image file exists
        img_path = Path(item["local_image_path"])
        assert img_path.exists(), f"Image not found: {img_path}"

        # Check bbox validity
        gt_pixel = item["ground_truth_bbox_pixel"]
        assert len(gt_pixel) == 4
        xmin, ymin, xmax, ymax = gt_pixel
        w, h = item["image_dimensions"]
        assert 0.0 <= xmin < xmax <= w, f"Invalid x bounds for {item['sample_id']}"
        assert 0.0 <= ymin < ymax <= h, f"Invalid y bounds for {item['sample_id']}"
        assert (xmax - xmin) * (ymax - ymin) > 0, "Zero ground truth area"


def test_coordinate_validity():
    """Verify coordinate validity filtering: negative, inverted, or non-finite values."""
    import math

    # 1. Normal valid box
    b_valid = PixelBoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=60.0)
    assert b_valid.area == (50.0 - 10.0) * (60.0 - 20.0)
    is_deg, reason = b_valid.check_degeneracy(100, 100)
    assert not is_deg

    # 2. Inverted box (xmin >= xmax)
    b_inv = PixelBoundingBox(xmin=50.0, ymin=20.0, xmax=10.0, ymax=60.0)
    assert b_inv.area == 0.0
    is_deg, reason = b_inv.check_degeneracy(100, 100)
    assert is_deg and reason == "inverted_coordinates"

    # 3. Non-finite values
    b_nan = PixelBoundingBox(xmin=float("nan"), ymin=20.0, xmax=50.0, ymax=60.0)
    assert b_nan.area == 0.0
    is_deg, reason = b_nan.check_degeneracy(100, 100)
    assert is_deg and reason == "non_finite_coordinates"


def test_metric_aggregation():
    """Verify statistical metric aggregation: mean, median, success@0.5, degenerate rate."""
    import numpy as np

    # Synthetic sample records
    ious = [0.85, 0.60, 0.40, 0.00, 0.00]
    is_degens = [False, False, False, False, True]
    is_empties = [False, False, False, True, False]

    n = len(ious)
    mean_iou = float(np.mean(ious))
    median_iou = float(np.median(ious))
    success_at_05 = sum(1 for iou in ious if iou >= 0.5) / n * 100.0
    degenerate_rate = sum(1 for d in is_degens if d) / n * 100.0
    empty_rate = sum(1 for e in is_empties if e) / n * 100.0

    assert round(mean_iou, 4) == 0.3700
    assert round(median_iou, 4) == 0.4000
    assert round(success_at_05, 1) == 40.0
    assert round(degenerate_rate, 1) == 20.0
    assert round(empty_rate, 1) == 20.0


def test_model_comparison_result_schema():
    """Verify the saved evaluation JSON adheres to the documented schema."""
    results_path = Path("docs/evaluation/grounding_evaluation_results.json")
    if not results_path.exists():
        pytest.skip("Evaluation results JSON not generated yet")

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "evaluation_dataset" in data
    assert "provenance" in data
    assert "sample_count" in data
    assert data["sample_count"] >= 30
    assert "models" in data
    assert "qwen2_vl_baseline" in data["models"]
    assert "grounding_dino" in data["models"]

    for model_key in ["qwen2_vl_baseline", "grounding_dino"]:
        m_info = data["models"][model_key]
        assert "model_name" in m_info
        assert "mean_iou" in m_info
        assert "median_iou" in m_info
        assert "success_at_0_5_pct" in m_info
        assert "valid_prediction_rate_pct" in m_info
        assert "degenerate_box_rate_pct" in m_info
        assert "empty_result_rate_pct" in m_info
        assert "mean_latency_ms" in m_info
        assert "peak_vram_allocated_mb" in m_info
        assert "sample_results" in m_info
        assert len(m_info["sample_results"]) == data["sample_count"]

