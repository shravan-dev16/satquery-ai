"""Comprehensive unit and integration tests for M4 change detection benchmark evaluation."""

import json
from pathlib import Path
import numpy as np
import pytest

from backend.evaluation.change_benchmark import (
    BenchmarkSampleResult,
    ChangeBenchmarkHarness,
    ChangeBenchmarkReport,
)
from backend.evaluation.failure_taxonomy import (
    FailureCategory,
    FailureDiagnosis,
    FailureTaxonomyClassifier,
)
from backend.evaluation.manifests import (
    EvaluationManifest,
    EvaluationSample,
    SceneCategory,
)
from backend.evaluation.metrics import (
    ChangeMetrics,
    ConfusionMatrix,
    aggregate_metrics,
    calculate_confusion_matrix,
    calculate_metrics,
)
from backend.models.change import DeterministicCVASpecialist, TinyCDSpecialist


# =========================================================================
# 1. Metric Correctness and Edge Cases
# =========================================================================

def test_metrics_exact_overlap():
    """Verify metrics return 1.0 when prediction perfectly matches ground truth."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[25:75, 25:75] = 255  # 2500 pixels
    pred = gt.copy()

    m = calculate_metrics(gt, pred)
    assert m.iou == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.false_positive_ratio == 0.0
    assert m.confusion.tp == 2500
    assert m.confusion.fp == 0
    assert m.confusion.fn == 0
    assert m.confusion.tn == 7500
    assert m.is_empty is False
    assert m.is_full_image is False


def test_metrics_disjoint_prediction():
    """Verify metrics return 0.0 when prediction is completely disjoint from GT."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255  # 400 px
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[60:80, 60:80] = 255  # 400 px disjoint

    m = calculate_metrics(gt, pred)
    assert m.iou == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.confusion.tp == 0
    assert m.confusion.fp == 400
    assert m.confusion.fn == 400
    assert m.confusion.tn == 9200


def test_metrics_zero_change_true_negative_agreement():
    """Verify true negative agreement when both GT and Pred have zero changed pixels."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    pred = np.zeros((100, 100), dtype=np.uint8)

    m = calculate_metrics(gt, pred)
    assert m.iou == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.false_positive_ratio == 0.0
    assert m.is_empty is True
    assert m.confusion.tp == 0
    assert m.confusion.fp == 0
    assert m.confusion.fn == 0
    assert m.confusion.tn == 10000


def test_metrics_zero_change_prediction_on_positive_gt():
    """Verify empty prediction on positive change gives 0 recall/precision and is_empty=True."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:40, 20:40] = 255
    pred = np.zeros((100, 100), dtype=np.uint8)

    m = calculate_metrics(gt, pred)
    assert m.iou == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.is_empty is True
    assert m.confusion.fn == 400


def test_metrics_false_alarm_on_zero_change_gt():
    """Verify prediction on empty GT gives 0 precision/recall/f1 and positive FPR."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:30, 10:30] = 255  # 400 false alarms

    m = calculate_metrics(gt, pred)
    assert m.iou == 0.0
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.false_positive_ratio == round(400 / 10000.0, 4)
    assert m.is_empty is False


def test_metrics_full_image_degeneracy_flag():
    """Verify full-image prediction (>= 98% coverage) triggers is_full_image=True."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    pred = np.full((100, 100), 255, dtype=np.uint8)  # 100% coverage

    m = calculate_metrics(gt, pred)
    assert m.is_full_image is True
    assert m.changed_pixel_ratio == 1.0


def test_metrics_dimension_mismatch_raises_error():
    """Verify mismatched dimensions or non-2D arrays raise ValueError."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    pred = np.zeros((100, 50), dtype=np.uint8)

    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_metrics(gt, pred)

    with pytest.raises(ValueError, match="Expected 2D arrays"):
        calculate_metrics(np.zeros((10, 10, 3)), np.zeros((10, 10, 3)))


# =========================================================================
# 2. Metric Aggregation: Macro vs Pixel-Global
# =========================================================================

def test_metric_aggregation_macro_vs_pixel_global():
    """Verify explicit distinction and calculation between macro and pixel-global metrics."""
    # Sample 1: 10x10, TP=20, FP=0, FN=0, TN=80 -> IoU=1.0, F1=1.0
    m1 = ChangeMetrics(
        iou=1.0,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        false_positive_ratio=0.0,
        changed_pixel_ratio=0.2,
        confusion=ConfusionMatrix(tp=20, fp=0, fn=0, tn=80, total_pixels=100),
        is_empty=False,
        is_full_image=False,
    )
    # Sample 2: 10x10, TP=0, FP=20, FN=20, TN=60 -> IoU=0.0, F1=0.0
    m2 = ChangeMetrics(
        iou=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        false_positive_ratio=0.25,
        changed_pixel_ratio=0.2,
        confusion=ConfusionMatrix(tp=0, fp=20, fn=20, tn=60, total_pixels=100),
        is_empty=False,
        is_full_image=False,
    )

    macro, global_micro = aggregate_metrics([m1, m2])

    # Macro mean of [1.0, 0.0] = 0.5000
    assert macro.sample_count == 2
    assert macro.mean_iou == 0.5000
    assert macro.median_iou == 0.5000
    assert macro.min_iou == 0.0
    assert macro.max_iou == 1.0
    assert macro.mean_f1 == 0.5000

    # Global pixel counts: TP=20, FP=20, FN=20, TN=140, Total=200
    # Global Precision = 20 / (20 + 20) = 0.5000
    # Global Recall = 20 / (20 + 20) = 0.5000
    # Global IoU = 20 / (20 + 20 + 20) = 20/60 = 0.3333
    # Global F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5000
    assert global_micro.total_pixels == 200
    assert global_micro.total_tp == 20
    assert global_micro.total_fp == 20
    assert global_micro.total_fn == 20
    assert global_micro.total_tn == 140
    assert global_micro.global_iou == 0.3333
    assert global_micro.global_precision == 0.5000
    assert global_micro.global_recall == 0.5000
    assert global_micro.global_f1 == 0.5000


# =========================================================================
# 3. Manifest Loading, Saving, and Filtering
# =========================================================================

def test_manifest_loading_and_filtering():
    """Verify loading from docs/evaluation/levir_cd_subset.json and filtering."""
    p = Path("docs/evaluation/levir_cd_subset.json")
    assert p.exists(), f"LEVIR manifest missing at {p}"

    manifest = EvaluationManifest.load(p)
    assert manifest.sample_count == 20
    assert manifest.quantitative_count == 20
    assert manifest.qualitative_count == 0

    quant_samples = manifest.filter_quantitative()
    assert len(quant_samples) == 20

    urban_samples = manifest.filter_by_category(SceneCategory.URBAN)
    assert len(urban_samples) == 20


def test_diverse_manifest_structure():
    """Verify diverse remote-sensing manifest integrity across categories."""
    p = Path("docs/evaluation/diverse_rs_manifest.json")
    assert p.exists(), f"Diverse RS manifest missing at {p}"

    manifest = EvaluationManifest.load(p)
    assert manifest.sample_count >= 10
    assert manifest.quantitative_count >= 9
    assert manifest.qualitative_count >= 1

    # Check presence of key categories
    categories = {s.scene_category for s in manifest.samples}
    assert SceneCategory.URBAN in categories
    assert SceneCategory.FOREST in categories
    assert SceneCategory.AGRICULTURE in categories
    assert SceneCategory.WATER in categories
    assert SceneCategory.NUISANCE_VARIATION in categories

    # Verify qualitative sample has no fake GT mask
    qual_samples = [s for s in manifest.samples if not s.is_quantitative]
    assert len(qual_samples) >= 1
    assert qual_samples[0].gt_mask_path is None


# =========================================================================
# 4. Failure Taxonomy Diagnosis Rules
# =========================================================================

def test_failure_taxonomy_classification():
    """Verify deterministic assignment of failure categories without hallucination."""
    # 1. Reliable sample (F1 >= 0.85, low FPR) -> No diagnosis (None)
    sample_urban = EvaluationSample(
        sample_id="s1",
        filename="s1.png",
        dataset="test",
        dimensions=[100, 100],
        total_pixels=10000,
        t1_path="dummy_t1",
        t2_path="dummy_t2",
        scene_category=SceneCategory.URBAN,
        is_quantitative=True,
    )
    m_good = ChangeMetrics(
        iou=0.85,
        precision=0.92,
        recall=0.91,
        f1=0.915,
        false_positive_ratio=0.002,
        changed_pixel_ratio=0.1,
        confusion=ConfusionMatrix(tp=900, fp=20, fn=80, tn=9000, total_pixels=10000),
        is_empty=False,
        is_full_image=False,
    )
    assert FailureTaxonomyClassifier.classify(sample_urban, m_good) is None

    # 2. Registration jitter nuisance -> ALIGNMENT_RELATED_ERROR
    sample_reg = EvaluationSample(
        sample_id="s_reg",
        filename="s_reg.png",
        dataset="test",
        dimensions=[100, 100],
        total_pixels=10000,
        t1_path="dummy_t1",
        t2_path="dummy_t2",
        scene_category=SceneCategory.NUISANCE_VARIATION,
        nuisance_type="registration",
        is_quantitative=True,
    )
    m_reg = ChangeMetrics(
        iou=0.0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        false_positive_ratio=0.02,
        changed_pixel_ratio=0.02,
        confusion=ConfusionMatrix(tp=0, fp=200, fn=0, tn=9800, total_pixels=10000),
        is_empty=False,
        is_full_image=False,
    )
    d_reg = FailureTaxonomyClassifier.classify(sample_reg, m_reg)
    assert d_reg is not None
    assert d_reg.category == FailureCategory.ALIGNMENT_RELATED_ERROR

    # 3. High false alarms -> FALSE_POSITIVE
    m_fp = ChangeMetrics(
        iou=0.20,
        precision=0.22,
        recall=0.85,
        f1=0.35,
        false_positive_ratio=0.15,
        changed_pixel_ratio=0.25,
        confusion=ConfusionMatrix(tp=500, fp=1800, fn=90, tn=7610, total_pixels=10000),
        is_empty=False,
        is_full_image=False,
    )
    d_fp = FailureTaxonomyClassifier.classify(sample_urban, m_fp)
    assert d_fp is not None
    assert d_fp.category == FailureCategory.FALSE_POSITIVE

    # 4. Low recall -> MISSED_CHANGE
    m_miss = ChangeMetrics(
        iou=0.30,
        precision=0.85,
        recall=0.32,
        f1=0.46,
        false_positive_ratio=0.005,
        changed_pixel_ratio=0.08,
        confusion=ConfusionMatrix(tp=2000, fp=350, fn=4200, tn=3450, total_pixels=10000),
        is_empty=False,
        is_full_image=False,
    )
    d_miss = FailureTaxonomyClassifier.classify(sample_urban, m_miss)
    assert d_miss is not None
    assert d_miss.category == FailureCategory.MISSED_CHANGE


# =========================================================================
# 5. Benchmark Harness Execution on Test Fixture
# =========================================================================

def test_harness_cva_execution_on_infrastructure_sample(tmp_path):
    """Verify ChangeBenchmarkHarness runs CVA on projected infrastructure GeoTIFF sample."""
    manifest_data = {
        "manifest_id": "test_infra_manifest",
        "dataset_name": "Test Dataset",
        "version": "1.0.0",
        "description": "Test fixture manifest",
        "license": "BSD-3-Clause",
        "samples": [
            {
                "sample_id": "infra_test_001",
                "filename": "time1_pre_change.tif",
                "dataset": "Test",
                "dimensions": [256, 256],
                "total_pixels": 65536,
                "changed_pixels": 6400,
                "t1_path": "tests/fixtures/bitemporal/time1_pre_change.tif",
                "t2_path": "tests/fixtures/bitemporal/time2_post_change.tif",
                "gt_mask_path": "tests/fixtures/bitemporal/change_ground_truth.png",
                "scene_category": "infrastructure",
                "is_quantitative": True,
                "crs": "EPSG:32643",
            }
        ],
    }

    m_path = tmp_path / "test_manifest.json"
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    harness = ChangeBenchmarkHarness("cva")
    report = harness.run(m_path)

    assert report.total_samples == 1
    assert report.quantitative_samples == 1
    assert report.model_name == "CVA_Baseline"
    assert len(report.per_sample_results) == 1

    sample_res = report.per_sample_results[0]
    assert sample_res.sample_id == "infra_test_001"
    assert sample_res.metrics is not None
    assert sample_res.metrics.iou >= 0.80  # high contrast block detected by CVA
    # Verify physical area calculation on projected UTM linear-meter CRS
    assert sample_res.physical_area_m2 is not None
    assert sample_res.physical_area_ha is not None
    # 6400 px * (10m * 10m) = 640,000 m^2 = 64.0 ha
    assert sample_res.physical_area_m2 == 640000.0
    assert sample_res.physical_area_ha == 64.0


def test_harness_qualitative_sample_no_fake_metrics(tmp_path):
    """Verify qualitative sample executes inference but generates zero fake metrics/IoU."""
    manifest_data = {
        "manifest_id": "test_qualitative_manifest",
        "dataset_name": "Test Dataset",
        "version": "1.0.0",
        "description": "Test qualitative fixture manifest",
        "license": "BSD-3-Clause",
        "samples": [
            {
                "sample_id": "qual_test_001",
                "filename": "real_rs_sample.tif",
                "dataset": "Test",
                "dimensions": [791, 718],
                "total_pixels": 567938,
                "t1_path": "tests/fixtures/real_rs_sample.tif",
                "t2_path": "tests/fixtures/real_rs_sample.tif",
                "gt_mask_path": None,
                "scene_category": "forest",
                "is_quantitative": False,
                "crs": "EPSG:32618",
            }
        ],
    }

    m_path = tmp_path / "test_qual_manifest.json"
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    harness = ChangeBenchmarkHarness("cva")
    report = harness.run(m_path)

    assert report.total_samples == 1
    assert report.quantitative_samples == 0
    assert report.qualitative_samples == 1
    assert report.macro_metrics is None  # no aggregated quantitative metrics

    sample_res = report.per_sample_results[0]
    assert sample_res.is_quantitative is False
    assert sample_res.metrics is None  # strictly null, no fake IoU or F1!
    assert sample_res.predicted_changed_pixels == 0  # identical images
