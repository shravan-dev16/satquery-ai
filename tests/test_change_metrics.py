"""Unit tests for change detection metrics, physical area, and spatial clustering."""

import numpy as np
import pytest

from backend.models.change import ChangeDetectionMetrics


def test_metrics_perfect_overlap():
    """Verify metrics return 1.0 when prediction perfectly matches ground truth."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:50, 20:50] = 255
    pred = gt.copy()

    m = ChangeDetectionMetrics.calculate(gt, pred)
    assert m["iou"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["is_empty"] is False
    assert m["is_full_image"] is False
    assert m["true_positive_pixels"] == 900
    assert m["false_positive_pixels"] == 0
    assert m["false_negative_pixels"] == 0


def test_metrics_disjoint_prediction():
    """Verify metrics return 0.0 when prediction is completely disjoint from ground truth."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255  # 400 pixels

    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[60:80, 60:80] = 255  # 400 pixels disjoint

    m = ChangeDetectionMetrics.calculate(gt, pred)
    assert m["iou"] == 0.0
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert m["true_positive_pixels"] == 0
    assert m["false_positive_pixels"] == 400
    assert m["false_negative_pixels"] == 400


def test_metrics_empty_prediction():
    """Verify empty prediction rate is correctly flagged."""
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[10:20, 10:20] = 255
    pred = np.zeros((50, 50), dtype=np.uint8)  # zero change predicted

    m = ChangeDetectionMetrics.calculate(gt, pred)
    assert m["is_empty"] is True
    assert m["iou"] == 0.0
    assert m["true_positive_pixels"] == 0


def test_metrics_full_image_prediction():
    """Verify degenerate full-image prediction (>98% change) is flagged as diagnostic."""
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.full((50, 50), 255, dtype=np.uint8)  # 100% change

    m = ChangeDetectionMetrics.calculate(gt, pred)
    assert m["is_full_image"] is True
    assert m["changed_pixel_ratio"] == 1.0


def test_metrics_known_fractional_case():
    """Verify exact mathematical calculation of IoU and F1 on known overlap."""
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[0:6, 0:6] = 255  # 36 pixels

    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[0:4, 0:6] = 255  # 24 pixels, all inside gt

    # tp = 24, fp = 0, fn = 12
    # precision = 24 / 24 = 1.0
    # recall = 24 / (24 + 12) = 24/36 = 0.6667
    # f1 = 2 * 1 * (2/3) / (1 + 2/3) = (4/3) / (5/3) = 0.8000
    # iou = 24 / (24 + 0 + 12) = 24/36 = 0.6667
    m = ChangeDetectionMetrics.calculate(gt, pred)
    assert abs(m["precision"] - 1.0) < 1e-4
    assert abs(m["recall"] - 0.6667) < 1e-4
    assert abs(m["f1"] - 0.8000) < 1e-4
    assert abs(m["iou"] - 0.6667) < 1e-4
