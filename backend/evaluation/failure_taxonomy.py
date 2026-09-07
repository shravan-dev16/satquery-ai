"""Structured failure taxonomy for bi-temporal remote-sensing change detection.

Provides:
- FailureCategory enumeration defining empirical failure types
- FailureDiagnosis model tracking per-sample failure category, severity, and rationale
- FailureTaxonomyClassifier assigning diagnosis through deterministic, unhallucinated heuristics
"""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from backend.evaluation.manifests import EvaluationSample, SceneCategory
from backend.evaluation.metrics import ChangeMetrics


class FailureCategory(str, Enum):
    """Standardized failure categories for change detection models."""

    MISSED_CHANGE = "missed_change"
    FALSE_POSITIVE = "false_positive"
    BOUNDARY_ERROR = "boundary_error"
    SMALL_CHANGE_MISSED = "small_change_missed"
    SEASONAL_VARIATION_FALSE_POSITIVE = "seasonal_variation_false_positive"
    SHADOW_FALSE_POSITIVE = "shadow_false_positive"
    CLOUD_RELATED_ERROR = "cloud_related_error"
    VEGETATION_ERROR = "vegetation_error"
    WATER_ERROR = "water_error"
    AGRICULTURE_ERROR = "agriculture_error"
    URBAN_ERROR = "urban_error"
    ALIGNMENT_RELATED_ERROR = "alignment_related_error"
    UNCLASSIFIED_FAILURE = "unclassified_failure"


class FailureDiagnosis(BaseModel):
    """Diagnosed failure for an evaluated sample with empirical rationale."""

    sample_id: str = Field(..., description="Evaluation sample identifier")
    category: FailureCategory = Field(..., description="Primary categorized failure mode")
    severity: float = Field(..., ge=0.0, le=1.0, description="Error severity metric in [0.0, 1.0]")
    rationale: str = Field(..., description="Empirically grounded explanation of error cause")
    metrics_summary: Dict[str, float] = Field(default_factory=dict, description="Key metrics relevant to diagnosis")


class FailureTaxonomyClassifier:
    """Assigns failure categories deterministically without unevidenced semantic inference."""

    @staticmethod
    def classify(
        sample: EvaluationSample,
        metrics: Optional[ChangeMetrics],
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> Optional[FailureDiagnosis]:
        """Diagnoses failure modes using deterministic confusion metrics and scene tags.

        Returns None if model performed reliably (F1 >= 0.85 and low FPR).
        """
        diag = diagnostics or {}

        # 1. Qualitative Evaluation Path (No Ground Truth Mask)
        if metrics is None or not sample.is_quantitative:
            pred_changed = diag.get("predicted_changed_pixels", 0)
            # Check nuisance stress conditions where zero change was expected
            if sample.nuisance_type == "registration" and pred_changed > 100:
                return FailureDiagnosis(
                    sample_id=sample.sample_id,
                    category=FailureCategory.ALIGNMENT_RELATED_ERROR,
                    severity=round(min(1.0, pred_changed / 10000.0), 4),
                    rationale=f"Registration offset generated {pred_changed} false positive edge pixels.",
                    metrics_summary={"predicted_changed_pixels": pred_changed},
                )
            if sample.nuisance_type in ("illumination", "shadow") and pred_changed > 100:
                return FailureDiagnosis(
                    sample_id=sample.sample_id,
                    category=FailureCategory.SHADOW_FALSE_POSITIVE,
                    severity=round(min(1.0, pred_changed / 10000.0), 4),
                    rationale=f"Illumination or shadow shift induced {pred_changed} false positive pixels.",
                    metrics_summary={"predicted_changed_pixels": pred_changed},
                )
            if sample.nuisance_type == "seasonal_color" and pred_changed > 100:
                return FailureDiagnosis(
                    sample_id=sample.sample_id,
                    category=FailureCategory.SEASONAL_VARIATION_FALSE_POSITIVE,
                    severity=round(min(1.0, pred_changed / 10000.0), 4),
                    rationale=f"Seasonal vegetation color shift produced {pred_changed} false positive pixels.",
                    metrics_summary={"predicted_changed_pixels": pred_changed},
                )
            # Cannot attribute failure on natural qualitative scenes without ground truth
            return None

        # 2. Quantitative Evaluation Path (Ground Truth Available)
        # Compliant detection threshold: high F1 and negligible false alarms
        if metrics.f1 >= 0.85 and metrics.false_positive_ratio < 0.01:
            return None

        summary = {
            "iou": metrics.iou,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "fpr": metrics.false_positive_ratio,
        }

        # Nuisance-specific classification
        if sample.nuisance_type == "registration" and metrics.false_positive_ratio > 0.005:
            return FailureDiagnosis(
                sample_id=sample.sample_id,
                category=FailureCategory.ALIGNMENT_RELATED_ERROR,
                severity=round(1.0 - metrics.f1, 4),
                rationale=f"Coregistration discrepancy produced edge false alarms (FPR={metrics.false_positive_ratio:.4f}).",
                metrics_summary=summary,
            )

        if sample.nuisance_type in ("illumination", "shadow") and metrics.false_positive_ratio > 0.01:
            return FailureDiagnosis(
                sample_id=sample.sample_id,
                category=FailureCategory.SHADOW_FALSE_POSITIVE,
                severity=round(1.0 - metrics.f1, 4),
                rationale=f"Sun angle / shadow variation resulted in false positives (FPR={metrics.false_positive_ratio:.4f}).",
                metrics_summary=summary,
            )

        if sample.nuisance_type == "seasonal_color" and metrics.false_positive_ratio > 0.01:
            return FailureDiagnosis(
                sample_id=sample.sample_id,
                category=FailureCategory.SEASONAL_VARIATION_FALSE_POSITIVE,
                severity=round(1.0 - metrics.f1, 4),
                rationale=f"Seasonal phenological variance mistaken for land-cover change (FPR={metrics.false_positive_ratio:.4f}).",
                metrics_summary=summary,
            )

        # Land-cover category specific classification
        if sample.scene_category == SceneCategory.FOREST and metrics.false_positive_ratio > 0.02:
            return FailureDiagnosis(
                sample_id=sample.sample_id,
                category=FailureCategory.VEGETATION_ERROR,
                severity=round(1.0 - metrics.f1, 4),
                rationale=f"Canopy seasonal discrepancy produced false alarms (FPR={metrics.false_positive_ratio:.4f}).",
                metrics_summary=summary,
            )

        if sample.scene_category == SceneCategory.AGRICULTURE and metrics.false_positive_ratio > 0.02:
            return FailureDiagnosis(
                sample_id=sample.sample_id,
                category=FailureCategory.AGRICULTURE_ERROR,
                severity=round(1.0 - metrics.f1, 4),
                rationale=f"Crop cycling or harvest mistaken for permanent change (FPR={metrics.false_positive_ratio:.4f}).",
                metrics_summary=summary,
            )

        if sample.scene_category == SceneCategory.WATER and metrics.false_positive_ratio > 0.02:
            return FailureDiagnosis(
                sample_id=sample.sample_id,
                category=FailureCategory.WATER_ERROR,
                severity=round(1.0 - metrics.f1, 4),
                rationale=f"Water surface glint or turbidity shift induced false alarms (FPR={metrics.false_positive_ratio:.4f}).",
                metrics_summary=summary,
            )

        # Metric profile classification
        if metrics.recall < 0.50 and metrics.precision >= 0.70:
            if metrics.confusion.tp < 1500 and metrics.confusion.fn > 0:
                return FailureDiagnosis(
                    sample_id=sample.sample_id,
                    category=FailureCategory.SMALL_CHANGE_MISSED,
                    severity=round(1.0 - metrics.recall, 4),
                    rationale=f"Small change structures undetected below receptive field (Recall={metrics.recall:.4f}).",
                    metrics_summary=summary,
                )
            return FailureDiagnosis(
                sample_id=sample.sample_id,
                category=FailureCategory.MISSED_CHANGE,
                severity=round(1.0 - metrics.recall, 4),
                rationale=f"Substantial change missed by detector (Recall={metrics.recall:.4f}, FN={metrics.confusion.fn}).",
                metrics_summary=summary,
            )

        if metrics.precision < 0.50 and metrics.recall >= 0.70:
            return FailureDiagnosis(
                sample_id=sample.sample_id,
                category=FailureCategory.FALSE_POSITIVE,
                severity=round(1.0 - metrics.precision, 4),
                rationale=f"High false alarm rate across non-change surface (Precision={metrics.precision:.4f}, FP={metrics.confusion.fp}).",
                metrics_summary=summary,
            )

        if metrics.recall < 0.70 and metrics.precision < 0.70:
            if metrics.confusion.tp > 0 and (
                metrics.confusion.fp > 0.25 * metrics.confusion.tp or metrics.confusion.fn > 0.25 * metrics.confusion.tp
            ):
                return FailureDiagnosis(
                    sample_id=sample.sample_id,
                    category=FailureCategory.BOUNDARY_ERROR,
                    severity=round(1.0 - metrics.f1, 4),
                    rationale=f"Boundary localization discrepancies around target structures (IoU={metrics.iou:.4f}).",
                    metrics_summary=summary,
                )
            return FailureDiagnosis(
                sample_id=sample.sample_id,
                category=FailureCategory.UNCLASSIFIED_FAILURE,
                severity=round(1.0 - metrics.f1, 4),
                rationale=f"General degradation without single dominant error mode (F1={metrics.f1:.4f}).",
                metrics_summary=summary,
            )

        return FailureDiagnosis(
            sample_id=sample.sample_id,
            category=FailureCategory.UNCLASSIFIED_FAILURE,
            severity=round(1.0 - metrics.f1, 4),
            rationale=f"Unclassified error (F1={metrics.f1:.4f}, IoU={metrics.iou:.4f}).",
            metrics_summary=summary,
        )
