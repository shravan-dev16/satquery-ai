"""Unified bi-temporal change detection benchmark harness.

Provides:
- Standardized execution for TinyCD, CVA, and swappable ModelRegistry specialists
- Strict isolation of quantitative vs qualitative-only sample tracks
- Geospatial CRS-aware physical area validation (m^2, hectares)
- Detailed per-sample records, failure taxonomy classification, and aggregate reports
- Latency and peak CUDA memory measurement
"""

import gc
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import torch

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from backend.evaluation.failure_taxonomy import FailureDiagnosis, FailureTaxonomyClassifier
from backend.evaluation.manifests import EvaluationManifest, EvaluationSample, SceneCategory
from backend.evaluation.metrics import (
    ChangeMetrics,
    MacroAggregateMetrics,
    PixelGlobalAggregateMetrics,
    aggregate_metrics,
    calculate_metrics,
)
from backend.models.change import (
    DeterministicCVASpecialist,
    TinyCDSpecialist,
)

logger = logging.getLogger(__name__)


class BenchmarkSampleResult(BaseModel):
    """Detailed benchmark evaluation result for a single bi-temporal sample."""

    sample_id: str
    filename: str
    scene_category: SceneCategory
    is_quantitative: bool
    nuisance_type: Optional[str] = None
    latency_ms: float
    predicted_changed_pixels: int
    predicted_change_ratio_pct: float
    ground_truth_changed_pixels: Optional[int] = None
    metrics: Optional[ChangeMetrics] = None
    failure_diagnosis: Optional[FailureDiagnosis] = None
    physical_area_m2: Optional[float] = None
    physical_area_ha: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


class ChangeBenchmarkReport(BaseModel):
    """Comprehensive machine-readable benchmark evaluation report."""

    benchmark_id: str
    model_name: str
    model_type: str
    manifest_id: str
    dataset_name: str
    total_samples: int
    quantitative_samples: int
    qualitative_samples: int
    hardware: Dict[str, Any]
    latency: Dict[str, float]
    peak_vram_mb: float
    macro_metrics: Optional[MacroAggregateMetrics] = None
    pixel_global_metrics: Optional[PixelGlobalAggregateMetrics] = None
    failure_distribution: Dict[str, int] = Field(default_factory=dict)
    per_sample_results: List[BenchmarkSampleResult]
    conclusions: Dict[str, Any]

    def save(self, path: Union[str, Path]) -> Path:
        """Saves report as formatted JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return p


class ChangeBenchmarkHarness:
    """Rigorous evaluation runner for change detection models."""

    def __init__(
        self,
        model: Union[str, TinyCDSpecialist, DeterministicCVASpecialist],
        device: Optional[str] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path or "models/checkpoints/levir_best.pth")

        if isinstance(model, str):
            model_key = model.lower().strip()
            if model_key in ("tinycd", "learned"):
                self.detector = TinyCDSpecialist(checkpoint_path=self.checkpoint_path, device=self.device)
                self.model_name = "TinyCD"
                self.model_type = "neural_siamese_attention"
            elif model_key in ("cva", "baseline"):
                self.detector = DeterministicCVASpecialist()
                self.model_name = "CVA_Baseline"
                self.model_type = "deterministic_cva"
            else:
                raise ValueError(f"Unknown model identifier: {model}. Choose 'tinycd' or 'cva'.")
        elif isinstance(model, TinyCDSpecialist):
            self.detector = model
            self.model_name = "TinyCD"
            self.model_type = "neural_siamese_attention"
        elif isinstance(model, DeterministicCVASpecialist):
            self.detector = model
            self.model_name = "CVA_Baseline"
            self.model_type = "deterministic_cva"
        else:
            self.detector = model
            self.model_name = getattr(model, "name", "CustomSpecialist")
            self.model_type = "custom"

    def _load_image_pair(self, sample: EvaluationSample) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Loads T1 and T2 images from GeoTIFF or standard formats, extracting geospatial metadata."""
        p1 = Path(sample.t1_path)
        p2 = Path(sample.t2_path)

        if not p1.exists():
            raise FileNotFoundError(f"Missing T1 image: {p1}")
        if not p2.exists():
            raise FileNotFoundError(f"Missing T2 image: {p2}")

        metadata: Dict[str, Any] = {"is_geotiff": False}

        if HAS_RASTERIO and p1.suffix.lower() in (".tif", ".tiff"):
            try:
                with rasterio.open(p1) as src1, rasterio.open(p2) as src2:
                    img1 = src1.read()
                    img2 = src2.read()
                    crs = src1.crs
                    transform = src1.transform
                    metadata["is_geotiff"] = True
                    metadata["crs"] = str(crs) if crs else None
                    metadata["is_projected"] = bool(crs and crs.is_projected)
                    metadata["pixel_res_x"] = abs(transform.a) if transform else None
                    metadata["pixel_res_y"] = abs(transform.e) if transform else None
            except Exception as e:
                logger.warning("Rasterio failed to open GeoTIFF %s: %s. Falling back to PIL.", p1, e)
                img1 = np.array(Image.open(p1).convert("RGB")).transpose(2, 0, 1)
                img2 = np.array(Image.open(p2).convert("RGB")).transpose(2, 0, 1)
        else:
            img1 = np.array(Image.open(p1).convert("RGB")).transpose(2, 0, 1)
            img2 = np.array(Image.open(p2).convert("RGB")).transpose(2, 0, 1)

        return img1, img2, metadata

    def _load_ground_truth(self, sample: EvaluationSample) -> Optional[np.ndarray]:
        """Loads ground truth binary mask if sample is quantitative."""
        if not sample.is_quantitative or not sample.gt_mask_path:
            return None

        p = Path(sample.gt_mask_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing GT mask for sample {sample.sample_id}: {p}")

        if HAS_RASTERIO and p.suffix.lower() in (".tif", ".tiff"):
            try:
                with rasterio.open(p) as src:
                    gt = src.read(1)
            except Exception:
                gt = np.array(Image.open(p).convert("L"))
        else:
            gt = np.array(Image.open(p).convert("L"))

        return gt

    def run(self, manifest_or_path: Union[str, Path, EvaluationManifest]) -> ChangeBenchmarkReport:
        """Executes full benchmark evaluation across manifest samples."""
        if isinstance(manifest_or_path, EvaluationManifest):
            manifest = manifest_or_path
        else:
            manifest = EvaluationManifest.load(manifest_or_path)

        logger.info(
            "Starting benchmark [%s] on manifest [%s] (%d samples)...",
            self.model_name,
            manifest.manifest_id,
            manifest.sample_count,
        )

        # Reset GPU peak telemetry
        is_cuda = (self.device == "cuda" and torch.cuda.is_available())
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Warmup pass on first sample
        if manifest.samples:
            s0 = manifest.samples[0]
            w1, w2, _ = self._load_image_pair(s0)
            _ = self.detector.predict(w1, w2)
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()

        sample_results: List[BenchmarkSampleResult] = []
        quantitative_metrics: List[ChangeMetrics] = []
        latencies: List[float] = []
        failure_counts: Dict[str, int] = {}

        for idx, sample in enumerate(manifest.samples):
            img1, img2, geo_meta = self._load_image_pair(sample)
            gt_mask = self._load_ground_truth(sample)

            # Measure inference latency
            t_start = time.perf_counter()
            pred_mask, prob_map, diag = self.detector.predict(img1, img2)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            latencies.append(latency_ms)

            # Pixel counts
            pred_binary = (pred_mask > 0).astype(np.uint8)
            changed_pixels = int(np.count_nonzero(pred_binary))
            total_pixels = int(pred_binary.size)
            change_ratio_pct = round((changed_pixels / float(total_pixels)) * 100.0, 4)

            # Physical Area calculation (Rule 9 / Work Package D)
            area_m2: Optional[float] = None
            area_ha: Optional[float] = None
            warnings: List[str] = []

            if geo_meta.get("is_geotiff"):
                if geo_meta.get("is_projected") and geo_meta.get("pixel_res_x") and geo_meta.get("pixel_res_y"):
                    pixel_area = geo_meta["pixel_res_x"] * geo_meta["pixel_res_y"]
                    area_m2 = round(changed_pixels * pixel_area, 2)
                    area_ha = round(area_m2 / 10000.0, 4)
                else:
                    warnings.append(
                        "CRS does not use linear meters or is unprojected; physical area cannot be deterministically computed."
                    )

            # Quantitative vs Qualitative separation (Work Package E)
            sample_metrics: Optional[ChangeMetrics] = None
            gt_changed: Optional[int] = None

            if sample.is_quantitative and gt_mask is not None:
                gt_changed = int(np.count_nonzero(gt_mask > 0))
                sample_metrics = calculate_metrics(gt_mask, pred_mask)
                quantitative_metrics.append(sample_metrics)

            # Structured failure diagnosis (Work Package F)
            diag_payload = {
                "predicted_changed_pixels": changed_pixels,
                "change_ratio_pct": change_ratio_pct,
            }
            diagnosis = FailureTaxonomyClassifier.classify(sample, sample_metrics, diag_payload)

            if diagnosis:
                cat_key = diagnosis.category.value
                failure_counts[cat_key] = failure_counts.get(cat_key, 0) + 1

            res = BenchmarkSampleResult(
                sample_id=sample.sample_id,
                filename=sample.filename,
                scene_category=sample.scene_category,
                is_quantitative=sample.is_quantitative,
                nuisance_type=sample.nuisance_type,
                latency_ms=round(latency_ms, 2),
                predicted_changed_pixels=changed_pixels,
                predicted_change_ratio_pct=change_ratio_pct,
                ground_truth_changed_pixels=gt_changed,
                metrics=sample_metrics,
                failure_diagnosis=diagnosis,
                physical_area_m2=area_m2,
                physical_area_ha=area_ha,
                warnings=warnings,
            )
            sample_results.append(res)
            logger.info(
                "[%d/%d] Sample %s (%s) — Latency: %.1f ms | Changed: %d px (%.2f%%)",
                idx + 1,
                manifest.sample_count,
                sample.sample_id,
                sample.scene_category.value,
                latency_ms,
                changed_pixels,
                change_ratio_pct,
            )

        peak_vram_mb = 0.0
        if is_cuda:
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)

        # Macro and Pixel-Global Aggregation
        macro_stats: Optional[MacroAggregateMetrics] = None
        global_stats: Optional[PixelGlobalAggregateMetrics] = None
        if quantitative_metrics:
            macro_stats, global_stats = aggregate_metrics(quantitative_metrics)

        latency_summary = {
            "mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "median_ms": round(float(np.median(latencies)), 2) if latencies else 0.0,
            "min_ms": round(float(np.min(latencies)), 2) if latencies else 0.0,
            "max_ms": round(float(np.max(latencies)), 2) if latencies else 0.0,
        }

        hardware_info = {
            "device": self.device,
            "gpu_name": torch.cuda.get_device_name(0) if is_cuda else "CPU",
            "cuda_version": torch.version.cuda if is_cuda else None,
        }

        conclusions = {
            "evaluated_samples": len(sample_results),
            "quantitative_evaluated": len(quantitative_metrics),
            "qualitative_evaluated": len(sample_results) - len(quantitative_metrics),
            "total_failure_diagnoses": sum(failure_counts.values()),
            "primary_failure_mode": max(failure_counts, key=failure_counts.get) if failure_counts else "none",
        }

        report = ChangeBenchmarkReport(
            benchmark_id=f"bm_{manifest.manifest_id}_{self.model_name.lower()}",
            model_name=self.model_name,
            model_type=self.model_type,
            manifest_id=manifest.manifest_id,
            dataset_name=manifest.dataset_name,
            total_samples=len(sample_results),
            quantitative_samples=len(quantitative_metrics),
            qualitative_samples=len(sample_results) - len(quantitative_metrics),
            hardware=hardware_info,
            latency=latency_summary,
            peak_vram_mb=peak_vram_mb,
            macro_metrics=macro_stats,
            pixel_global_metrics=global_stats,
            failure_distribution=failure_counts,
            per_sample_results=sample_results,
            conclusions=conclusions,
        )

        logger.info(
            "Benchmark completed: %s on %s (Mean Latency: %.1f ms | Peak VRAM: %.1f MB)",
            self.model_name,
            manifest.manifest_id,
            latency_summary["mean_ms"],
            peak_vram_mb,
        )
        return report
