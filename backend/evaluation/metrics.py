"""Mathematical metrics and aggregation for bi-temporal change detection evaluation.

Provides:
- Exact pixel-level confusion matrix computation (TP, FP, FN, TN)
- Per-sample metrics (IoU, Precision, Recall, F1, FPR, Changed Pixel Ratio)
- Explicit edge-case handling (true negative agreement, empty predictions, full-image coverage)
- Separate, clearly labeled macro-averaging and pixel-global (micro) aggregation semantics
"""

from typing import List, Tuple
import numpy as np
from pydantic import BaseModel, Field


class ConfusionMatrix(BaseModel):
    """Exact pixel confusion counts for binary change detection."""

    tp: int = Field(..., description="True Positive pixels (correctly detected change)")
    fp: int = Field(..., description="False Positive pixels (false alarms on non-change)")
    fn: int = Field(..., description="False Negative pixels (missed changes)")
    tn: int = Field(..., description="True Negative pixels (correctly identified non-change)")
    total_pixels: int = Field(..., description="Total evaluated pixels (H x W)")


class ChangeMetrics(BaseModel):
    """Per-sample evaluation metrics for binary change detection."""

    iou: float = Field(..., description="Intersection over Union (Jaccard Index) for positive change class")
    precision: float = Field(..., description="Precision: TP / (TP + FP)")
    recall: float = Field(..., description="Recall / True Positive Rate: TP / (TP + FN)")
    f1: float = Field(..., description="F1 Score: Harmonic mean of precision and recall")
    false_positive_ratio: float = Field(..., description="Fallout / False Positive Ratio: FP / (FP + TN)")
    changed_pixel_ratio: float = Field(..., description="Fraction of image predicted as changed: (TP + FP) / Total")
    confusion: ConfusionMatrix = Field(..., description="Raw pixel confusion counts")
    is_empty: bool = Field(..., description="Whether prediction contains zero changed pixels")
    is_full_image: bool = Field(..., description="Whether prediction covers >= 98% of total pixels (degeneracy check)")


class MacroAggregateMetrics(BaseModel):
    """Sample-level macro-averaged metrics (unweighted mean across samples with distribution stats)."""

    sample_count: int
    mean_iou: float
    median_iou: float
    std_iou: float
    min_iou: float
    max_iou: float

    mean_precision: float
    median_precision: float
    std_precision: float
    min_precision: float
    max_precision: float

    mean_recall: float
    median_recall: float
    std_recall: float
    min_recall: float
    max_recall: float

    mean_f1: float
    median_f1: float
    std_f1: float
    min_f1: float
    max_f1: float

    mean_fpr: float
    empty_prediction_rate: float
    full_image_prediction_rate: float


class PixelGlobalAggregateMetrics(BaseModel):
    """Dataset-level micro-aggregated metrics (summed pixel confusion across all samples)."""

    sample_count: int
    total_pixels: int
    total_tp: int
    total_fp: int
    total_fn: int
    total_tn: int

    global_iou: float
    global_precision: float
    global_recall: float
    global_f1: float
    global_fpr: float
    global_changed_pixel_ratio: float


def calculate_confusion_matrix(gt_mask: np.ndarray, pred_mask: np.ndarray) -> ConfusionMatrix:
    """Calculates exact binary pixel confusion matrix between ground truth and prediction.

    Args:
        gt_mask: 2D numpy array where > 0 represents positive change.
        pred_mask: 2D numpy array where > 0 represents positive change.

    Raises:
        ValueError: If array dimensions do not match or are not 2D.
    """
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(f"Shape mismatch: ground truth shape {gt_mask.shape} != prediction shape {pred_mask.shape}")
    if gt_mask.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got ndim={gt_mask.ndim} (shape: {gt_mask.shape})")

    gt = (gt_mask > 0).astype(bool)
    pred = (pred_mask > 0).astype(bool)

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & (~gt)))
    fn = int(np.count_nonzero((~pred) & gt))
    tn = int(np.count_nonzero((~pred) & (~gt)))
    total_pixels = int(pred.size)

    return ConfusionMatrix(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        total_pixels=total_pixels,
    )


def calculate_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> ChangeMetrics:
    """Calculates comprehensive scientific metrics with explicit edge-case semantics.

    Edge-case conventions:
    - True Negative Agreement (both GT and Pred have 0 change pixels):
      IoU=1.0, Precision=1.0, Recall=1.0, F1=1.0, FPR=0.0, is_empty=True.
      (Both agree perfectly that zero change occurred).
    - Zero Change Predicted on Non-Empty GT:
      TP=0, FP=0, FN>0 -> Recall=0.0, Precision=0.0, IoU=0.0, F1=0.0, is_empty=True.
    - Change Predicted on Zero-Change GT (Pure False Alarm):
      TP=0, FP>0, FN=0 -> Precision=0.0, Recall=0.0, IoU=0.0, F1=0.0, FPR=FP/(FP+TN).
    - Full-Image Degeneracy:
      Predicted changed pixels >= 98% of total pixels -> is_full_image=True.
    """
    cm = calculate_confusion_matrix(gt_mask, pred_mask)

    total_pixels = cm.total_pixels
    tp = cm.tp
    fp = cm.fp
    fn = cm.fn
    tn = cm.tn

    # Check for True Negative Agreement (both masks completely empty)
    if (tp + fp + fn) == 0:
        return ChangeMetrics(
            iou=1.0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            false_positive_ratio=0.0,
            changed_pixel_ratio=0.0,
            confusion=cm,
            is_empty=True,
            is_full_image=False,
        )

    # Standard calculations with zero-division safety
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0.0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    iou = (tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    changed_pixel_ratio = (tp + fp) / float(total_pixels)

    is_empty = bool(tp + fp == 0)
    is_full_image = bool(changed_pixel_ratio >= 0.98)

    return ChangeMetrics(
        iou=round(float(iou), 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1=round(float(f1), 4),
        false_positive_ratio=round(float(fpr), 4),
        changed_pixel_ratio=round(float(changed_pixel_ratio), 4),
        confusion=cm,
        is_empty=is_empty,
        is_full_image=is_full_image,
    )


def aggregate_metrics(
    sample_metrics: List[ChangeMetrics],
) -> Tuple[MacroAggregateMetrics, PixelGlobalAggregateMetrics]:
    """Aggregates a list of per-sample metrics into both macro and pixel-global metrics.

    Args:
        sample_metrics: List of ChangeMetrics evaluated on individual image pairs.

    Returns:
        (MacroAggregateMetrics, PixelGlobalAggregateMetrics)
    """
    if not sample_metrics:
        raise ValueError("Cannot aggregate empty list of sample metrics.")

    n = len(sample_metrics)

    # Macro distribution stats
    ious = [m.iou for m in sample_metrics]
    precs = [m.precision for m in sample_metrics]
    recs = [m.recall for m in sample_metrics]
    f1s = [m.f1 for m in sample_metrics]
    fprs = [m.false_positive_ratio for m in sample_metrics]

    empty_count = sum(1 for m in sample_metrics if m.is_empty)
    full_count = sum(1 for m in sample_metrics if m.is_full_image)

    macro = MacroAggregateMetrics(
        sample_count=n,
        mean_iou=round(float(np.mean(ious)), 4),
        median_iou=round(float(np.median(ious)), 4),
        std_iou=round(float(np.std(ious)), 4),
        min_iou=round(float(np.min(ious)), 4),
        max_iou=round(float(np.max(ious)), 4),
        mean_precision=round(float(np.mean(precs)), 4),
        median_precision=round(float(np.median(precs)), 4),
        std_precision=round(float(np.std(precs)), 4),
        min_precision=round(float(np.min(precs)), 4),
        max_precision=round(float(np.max(precs)), 4),
        mean_recall=round(float(np.mean(recs)), 4),
        median_recall=round(float(np.median(recs)), 4),
        std_recall=round(float(np.std(recs)), 4),
        min_recall=round(float(np.min(recs)), 4),
        max_recall=round(float(np.max(recs)), 4),
        mean_f1=round(float(np.mean(f1s)), 4),
        median_f1=round(float(np.median(f1s)), 4),
        std_f1=round(float(np.std(f1s)), 4),
        min_f1=round(float(np.min(f1s)), 4),
        max_f1=round(float(np.max(f1s)), 4),
        mean_fpr=round(float(np.mean(fprs)), 4),
        empty_prediction_rate=round(float(empty_count / n), 4),
        full_image_prediction_rate=round(float(full_count / n), 4),
    )

    # Pixel-global (micro) counts
    total_tp = sum(m.confusion.tp for m in sample_metrics)
    total_fp = sum(m.confusion.fp for m in sample_metrics)
    total_fn = sum(m.confusion.fn for m in sample_metrics)
    total_tn = sum(m.confusion.tn for m in sample_metrics)
    total_pixels = sum(m.confusion.total_pixels for m in sample_metrics)

    global_prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    global_rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    if (global_prec + global_rec) > 0.0:
        global_f1 = (2.0 * global_prec * global_rec) / (global_prec + global_rec)
    else:
        global_f1 = 0.0

    global_iou = (total_tp / (total_tp + total_fp + total_fn)) if (total_tp + total_fp + total_fn) > 0 else 0.0
    global_fpr = (total_fp / (total_fp + total_tn)) if (total_fp + total_tn) > 0 else 0.0
    global_change_ratio = (total_tp + total_fp) / float(total_pixels)

    micro = PixelGlobalAggregateMetrics(
        sample_count=n,
        total_pixels=total_pixels,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_tn=total_tn,
        global_iou=round(float(global_iou), 4),
        global_precision=round(float(global_prec), 4),
        global_recall=round(float(global_rec), 4),
        global_f1=round(float(global_f1), 4),
        global_fpr=round(float(global_fpr), 4),
        global_changed_pixel_ratio=round(float(global_change_ratio), 4),
    )

    return macro, micro
