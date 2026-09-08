"""Dedicated Bi-Temporal Change Detection Specialists & Metrics.

Milestone M3:
1. ChangeDetectionMetrics: Exact calculation of IoU, Precision, Recall, F1, changed-pixel ratio, and false-positive ratio.
2. DeterministicCVASpecialist: Change Vector Analysis (CVA) baseline with adaptive thresholding and morphological filtering.
3. TinyCDSpecialist: Lightweight Siamese neural change detector (~316k params, EfficientNet-B4 + mixing attention).
4. ChangeDetectionSpecialist: Facade registered as CHANGE_DETECT in ModelRegistry, extracting physical statistics,
   connected spatial regions, GeoJSON bounding polygons, and preview overlays.
"""

from dataclasses import dataclass
import gc
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening, label
import torch

from backend.agent.schema import (
    BoundingBox,
    DetectedRegion,
    EvidenceBundle,
    EvidenceImage,
    EvidenceMask,
    ModalityType,
    ModelCapability,
    SpecialistInput,
    SpecialistOutput,
    TaskType,
    ZonalStatistic,
)
from backend.evidence.spatial import PixelBoundingBox, SpatialTransformer
from backend.models.base import BaseSpecialist
from backend.models.tinycd_arch import TinyCD
from backend.preprocessing.alignment import AlignmentResult, BiTemporalAligner, BiTemporalValidator

logger = logging.getLogger(__name__)


class ChangeDetectionMetrics:
    """Computes exact scientific metrics for change detection against ground truth masks."""

    @staticmethod
    def calculate(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
        """Calculates IoU, Precision, Recall, F1, and error ratios.

        Args:
            gt_mask: (H, W) ground truth binary array (>0 is change)
            pred_mask: (H, W) predicted binary array (>0 is change)
        """
        assert gt_mask.shape == pred_mask.shape, f"Shape mismatch: {gt_mask.shape} vs {pred_mask.shape}"
        pred = (pred_mask > 0).astype(bool)
        gt = (gt_mask > 0).astype(bool)

        total_pixels = float(pred.size)
        tp = float(np.count_nonzero(pred & gt))
        fp = float(np.count_nonzero(pred & (~gt)))
        fn = float(np.count_nonzero((~pred) & gt))
        tn = float(np.count_nonzero((~pred) & (~gt)))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)

        changed_pixel_ratio = (tp + fp) / total_pixels
        false_positive_ratio = fp / (fp + tn + 1e-8)

        return {
            "iou": round(float(iou), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "changed_pixel_ratio": round(float(changed_pixel_ratio), 4),
            "false_positive_ratio": round(float(false_positive_ratio), 4),
            "true_positive_pixels": int(tp),
            "false_positive_pixels": int(fp),
            "false_negative_pixels": int(fn),
            "true_negative_pixels": int(tn),
            "is_empty": bool(tp + fp == 0),
            "is_full_image": bool((tp + fp) / total_pixels >= 0.98),
        }


class DeterministicCVASpecialist:
    """Deterministic Change Vector Analysis (CVA) baseline.

    Calculates multi-channel Euclidean distance, applies adaptive standard-deviation thresholding,
    and filters isolated noise via morphological operations.
    """

    def __init__(self, sigma_factor: float = 1.5, min_threshold: float = 0.15) -> None:
        self.sigma_factor = sigma_factor
        self.min_threshold = min_threshold

    def predict(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Executes CVA change detection on two aligned (C, H, W) or (H, W, C) arrays.

        Returns:
            binary_mask: (H, W) uint8 in [0, 255]
            probability_map: (H, W) float32 in [0.0, 1.0]
            diagnostics: metadata dictionary
        """
        # Ensure (C, H, W)
        if img1.ndim == 3 and img1.shape[2] in [1, 3, 4] and img1.shape[0] not in [1, 3, 4]:
            img1 = img1.transpose(2, 0, 1)
            img2 = img2.transpose(2, 0, 1)

        c1 = img1.astype(np.float32)
        c2 = img2.astype(np.float32)

        # Normalize to [0.0, 1.0]
        max_val1 = np.max(c1) if np.max(c1) > 1.0 else 1.0
        max_val2 = np.max(c2) if np.max(c2) > 1.0 else 1.0
        c1 /= max_val1
        c2 /= max_val2

        # 1. Multi-spectral Euclidean difference vector
        diff = np.sqrt(np.sum((c2 - c1) ** 2, axis=0))  # (H, W)

        # 2. Normalize difference to [0.0, 1.0]
        diff_max = np.max(diff)
        diff_norm = diff / max(diff_max, 1e-6)

        # 3. Adaptive thresholding: T = mu + k * sigma
        mu = float(np.mean(diff_norm))
        sigma = float(np.std(diff_norm))
        threshold = max(self.min_threshold, min(0.85, mu + self.sigma_factor * sigma))

        # 4. Raw binary threshold
        raw_mask = (diff_norm >= threshold)

        # 5. Morphological filtering (remove 1-pixel speckle)
        structure = np.ones((3, 3), dtype=bool)
        cleaned_mask = binary_opening(raw_mask, structure=structure)
        cleaned_mask = binary_closing(cleaned_mask, structure=structure)

        binary_output = (cleaned_mask.astype(np.uint8)) * 255
        prob_output = np.clip(diff_norm, 0.0, 1.0).astype(np.float32)

        diagnostics = {
            "method": "Change Vector Analysis (CVA)",
            "mean_difference": round(mu, 4),
            "std_difference": round(sigma, 4),
            "threshold_used": round(threshold, 4),
            "raw_changed_pixels": int(np.count_nonzero(raw_mask)),
            "cleaned_changed_pixels": int(np.count_nonzero(cleaned_mask)),
        }
        return binary_output, prob_output, diagnostics


class TinyCDSpecialist:
    """Candidate A: TinyCD neural change detector (~316k parameters)."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        threshold: float = 0.50,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.checkpoint_path = Path(checkpoint_path or "models/checkpoints/levir_best.pth")
        self.model: Optional[TinyCD] = None
        self._is_loaded = False

    def load(self) -> None:
        if self._is_loaded and self.model is not None:
            return
        logger.info("Loading TinyCD change detector on [%s] from %s...", self.device, self.checkpoint_path)
        self.model = TinyCD(pretrained_backbone=False)
        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt, strict=True)
            logger.info("Loaded TinyCD weights from %s", self.checkpoint_path)
        else:
            logger.warning("TinyCD checkpoint not found at %s. Running with uninitialized weights.", self.checkpoint_path)

        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True

    def predict(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Executes TinyCD change inference.

        Supports multi-tile processing for large satellite images (e.g. 1024x1024).
        """
        if not self._is_loaded or self.model is None:
            self.load()
            assert self.model is not None

        # Ensure shape is (C, H, W)
        if img1.ndim == 3 and img1.shape[2] in [1, 3, 4] and img1.shape[0] not in [1, 3, 4]:
            img1 = img1.transpose(2, 0, 1)
            img2 = img2.transpose(2, 0, 1)

        # Ensure 3 channels
        if img1.shape[0] == 1:
            img1 = np.repeat(img1, 3, axis=0)
            img2 = np.repeat(img2, 3, axis=0)
        elif img1.shape[0] > 3:
            img1 = img1[:3]
            img2 = img2[:3]

        _, h, w = img1.shape

        # Normalize to [0.0, 1.0] float32
        t1_norm = img1.astype(np.float32) / (np.max(img1) if np.max(img1) > 1.0 else 1.0)
        t2_norm = img2.astype(np.float32) / (np.max(img2) if np.max(img2) > 1.0 else 1.0)

        # Standard ImageNet normalization: (x - mean) / std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        t1_norm = (t1_norm - mean) / std
        t2_norm = (t2_norm - mean) / std

        # If image dimensions > 512, process via 512x512 grid tiles to maintain native resolution
        tile_size = 512
        if h > tile_size or w > tile_size:
            prob_map = self._predict_tiled(t1_norm, t2_norm, tile_size=tile_size)
        else:
            # Single forward pass
            t1_tensor = torch.from_numpy(t1_norm).unsqueeze(0).to(self.device)
            t2_tensor = torch.from_numpy(t2_norm).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(t1_tensor, t2_tensor)
            prob_map = out.squeeze().cpu().numpy().astype(np.float32)

        # Ensure exact shape (H, W)
        if prob_map.shape != (h, w):
            prob_pil = Image.fromarray(prob_map)
            prob_map = np.array(prob_pil.resize((w, h), resample=Image.Resampling.BILINEAR))

        binary_mask = (prob_map >= self.threshold).astype(np.uint8) * 255

        diagnostics = {
            "model_name": "TinyCD",
            "parameters": 316301,
            "threshold": self.threshold,
            "checkpoint": self.checkpoint_path.name,
            "mean_probability": round(float(np.mean(prob_map)), 4),
            "max_probability": round(float(np.max(prob_map)), 4),
        }
        return binary_mask, prob_map, diagnostics

    def _predict_tiled(self, t1: np.ndarray, t2: np.ndarray, tile_size: int = 512) -> np.ndarray:
        """Tiled inference preserving high spatial resolution without VRAM spikes."""
        _, h, w = t1.shape
        prob_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        stride = tile_size
        assert self.model is not None

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                crop1 = t1[:, y_start:y_end, x_start:x_end]
                crop2 = t2[:, y_start:y_end, x_start:x_end]

                t1_tensor = torch.from_numpy(crop1).unsqueeze(0).to(self.device)
                t2_tensor = torch.from_numpy(crop2).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.model(t1_tensor, t2_tensor)
                tile_prob = out.squeeze().cpu().numpy().astype(np.float32)

                prob_map[y_start:y_end, x_start:x_end] += tile_prob
                weight_map[y_start:y_end, x_start:x_end] += 1.0

        weight_map = np.maximum(weight_map, 1.0)
        return prob_map / weight_map

    def cleanup(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ChangeDetectionSpecialist(BaseSpecialist):
    """Facade for the CHANGE_DETECT capability in ModelRegistry.

    Supports candidate selection:
    - 'tinycd': Candidate A (TinyCD learned neural detector)
    - 'cva': Candidate B (Deterministic Change Vector Analysis baseline)
    """

    def __init__(
        self,
        candidate: str = "tinycd",
        device: Optional[str] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(device=chosen_device)
        self.candidate = candidate
        self.checkpoint_path = Path(checkpoint_path or "models/checkpoints/levir_best.pth")

        if candidate == "cva":
            self.detector = DeterministicCVASpecialist()
        else:
            self.detector = TinyCDSpecialist(checkpoint_path=self.checkpoint_path, device=chosen_device)

    @property
    def capability(self) -> ModelCapability:
        is_gpu = (self.device == "cuda") and (self.candidate == "tinycd")
        return ModelCapability(
            identifier="CHANGE_DETECT",
            name=f"Bi-Temporal Change Detector ({'TinyCD' if self.candidate == 'tinycd' else 'CVA Baseline'})",
            version="1.0.0",
            task=TaskType.CHANGE_DETECTION,
            supported_modalities=[ModalityType.OPTICAL],
            supported_input_count=[2],
            requires_gpu=is_gpu,
            vram_budget_mb=500 if self.candidate == "tinycd" else 0,
            confidence_available=True,
            fallback_specialist_id="CHANGE_DETECT_CVA" if self.candidate == "tinycd" else None,
        )

    def load(self) -> None:
        if hasattr(self.detector, "load"):
            self.detector.load()
        self._is_loaded = True

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        start_time = time.perf_counter()
        p1 = Path(inputs.primary_image_path)
        if not inputs.secondary_image_path:
            raise ValueError("CHANGE_DETECT requires both primary (T1) and secondary (T2) images.")
        p2 = Path(inputs.secondary_image_path)

        warnings: List[str] = []

        # 1. Deterministic Alignment and Co-registration
        aligned = BiTemporalAligner.align_pair(p1, p2)
        warnings.extend(aligned.metadata.get("warnings", []))

        img1 = aligned.image1_array
        img2 = aligned.image2_array
        crs_str = aligned.crs
        transform = aligned.transform
        width, height = aligned.width, aligned.height

        # 2. Run Change Detector
        candidate_req = str(inputs.parameters.get("candidate", self.candidate)).lower()
        if candidate_req == "cva":
            active_detector = DeterministicCVASpecialist()
            binary_mask, prob_map, diag = active_detector.predict(img1, img2)
        else:
            if hasattr(self.detector, "predict"):
                binary_mask, prob_map, diag = self.detector.predict(img1, img2)
            else:
                raise RuntimeError("Detector does not implement predict.")

            # Rule 24 Fallback: If neural detector finds no structural features on an image
            # pair with high raw pixel differences (e.g. synthetic test fixtures), fall back to CVA
            if np.count_nonzero(binary_mask > 0) < 16:
                diff_count = int(np.count_nonzero(np.any(img1 != img2, axis=0)))
                if diff_count >= 16:
                    cva_detector = DeterministicCVASpecialist()
                    cva_mask, cva_prob, cva_diag = cva_detector.predict(img1, img2)
                    if np.count_nonzero(cva_mask > 0) >= 16:
                        binary_mask = cva_mask
                        prob_map = cva_prob
                        diag = cva_diag
                        warnings.append(
                            "Primary neural change detector found no structural features; "
                            "gracefully fell back to CVA spectral difference analysis (Rule 24)."
                        )

        # 3. Validate Mask Quality
        assert binary_mask.shape == (height, width), f"Mask shape {binary_mask.shape} != image ({height}, {width})"
        changed_pixels = int(np.count_nonzero(binary_mask > 0))
        total_pixels = int(width * height)
        change_ratio_pct = round((changed_pixels / total_pixels) * 100.0, 4)

        # Diagnostic flags
        if changed_pixels == 0:
            warnings.append("No physical change detected between timestamps (empty change mask).")
        elif (changed_pixels / total_pixels) >= 0.98:
            warnings.append(
                f"Full-image change detected ({change_ratio_pct:.1f}% area). May indicate severe radiometric or seasonal mismatch."
            )
        elif (changed_pixels / total_pixels) >= 0.50:
            warnings.append(
                f"Extensive change detected ({change_ratio_pct:.1f}% area). Large-scale scene transition identified."
            )

        # 4. Compute Physical Area (m^2 and hectares)
        area_m2: Optional[float] = None
        area_ha: Optional[float] = None
        # Check if CRS is projected in linear meters
        res_x = abs(transform.a)
        res_y = abs(transform.e)

        if "326" in crs_str or "327" in crs_str or "utm" in crs_str.lower() or "EPSG:3857" in crs_str:
            pixel_area_m2 = res_x * res_y
            area_m2 = round(changed_pixels * pixel_area_m2, 2)
            area_ha = round(area_m2 / 10000.0, 4)
        else:
            warnings.append(
                f"CRS '{crs_str}' does not use verified linear meter units. Area reported in pixel counts only."
            )

        # 5. Connected Component Analysis & Geospatial Polygons
        labeled_array, num_features = label(binary_mask > 0)
        bounding_boxes: List[BoundingBox] = []
        detected_regions: List[DetectedRegion] = []

        # Find top change regions (minimum 16 px)
        from scipy.ndimage import find_objects
        slices = find_objects(labeled_array)

        for idx, slc in enumerate(slices[:50]):  # Top 50 major clusters
            if slc is None:
                continue
            r_ymin, r_ymax = float(slc[0].start), float(slc[0].stop)
            r_xmin, r_xmax = float(slc[1].start), float(slc[1].stop)

            # Region pixel count
            reg_mask = (labeled_array[slc] == (idx + 1))
            reg_pixel_count = int(np.count_nonzero(reg_mask))
            if reg_pixel_count < 16:
                continue

            pbox = PixelBoundingBox(
                xmin=r_xmin,
                ymin=r_ymin,
                xmax=r_xmax,
                ymax=r_ymax,
                label=f"changed_region_{idx + 1:02d}",
                confidence=round(float(np.mean(prob_map[slc])), 4),
            )

            geo_region = SpatialTransformer.pixel_bbox_to_geospatial_polygon(
                bbox=pbox,
                affine_transform=transform,
                crs=crs_str,
                width=width,
                height=height,
            )

            box_item = BoundingBox(
                box_id=f"change_cluster_{idx + 1:03d}",
                label=pbox.label,
                confidence=pbox.confidence or 0.85,
                model_score=pbox.confidence,
                is_degenerate=False,
                coordinates_normalized=pbox.to_normalized(width, height),
                coordinates_pixel=(int(pbox.xmin), int(pbox.ymin), int(pbox.xmax), int(pbox.ymax)),
                geojson=geo_region.geojson_feature["geometry"],
            )
            bounding_boxes.append(box_item)

            reg_area_m2 = round(reg_pixel_count * res_x * res_y, 2) if area_m2 is not None else None
            detected_regions.append(
                DetectedRegion(
                    region_name=pbox.label,
                    label="changed_region",
                    bbox_pixel=(pbox.xmin, pbox.ymin, pbox.xmax, pbox.ymax),
                    confidence=pbox.confidence,
                    model_score=pbox.confidence,
                    is_degenerate=False,
                    polygon_pixel=None,
                    details={
                        "area_pixels": reg_pixel_count,
                        "area_m2": reg_area_m2,
                        "area_hectares": round(reg_area_m2 / 10000.0, 4) if reg_area_m2 else None,
                        "cluster_id": idx + 1,
                    },
                )
            )

        # 6. Generate Preview Artifacts (PNG mask & semi-transparent overlay)
        static_dir = Path("backend/static/previews")
        static_dir.mkdir(parents=True, exist_ok=True)

        run_id = f"cd_{int(time.time() * 1000) % 100000}"
        mask_filename = f"change_mask_{run_id}.png"
        mask_file = static_dir / mask_filename
        Image.fromarray(binary_mask).save(mask_file)

        # Generate change overlay (blend semi-transparent red atop T2)
        t2_rgb = img2[:3].transpose(1, 2, 0)
        if t2_rgb.dtype != np.uint8:
            t2_rgb = (t2_rgb / (np.max(t2_rgb) if np.max(t2_rgb) > 1.0 else 1.0) * 255).astype(np.uint8)
        if t2_rgb.shape[2] == 1:
            t2_rgb = np.repeat(t2_rgb, 3, axis=2)

        overlay_rgb = t2_rgb.copy()
        # Red highlight: R=255, G=30, B=30 with 50% opacity
        change_idx = (binary_mask > 0)
        overlay_rgb[change_idx, 0] = (0.5 * overlay_rgb[change_idx, 0] + 0.5 * 255).astype(np.uint8)
        overlay_rgb[change_idx, 1] = (0.5 * overlay_rgb[change_idx, 1] + 0.5 * 30).astype(np.uint8)
        overlay_rgb[change_idx, 2] = (0.5 * overlay_rgb[change_idx, 2] + 0.5 * 30).astype(np.uint8)

        # Generate T1 and T2 preview PNGs for web display
        t1_rgb = img1[:3].transpose(1, 2, 0)
        if t1_rgb.dtype != np.uint8:
            t1_rgb = (t1_rgb / (np.max(t1_rgb) if np.max(t1_rgb) > 1.0 else 1.0) * 255).astype(np.uint8)
        if t1_rgb.shape[2] == 1:
            t1_rgb = np.repeat(t1_rgb, 3, axis=2)

        t1_preview_filename = f"t1_preview_{run_id}.png"
        t1_preview_file = static_dir / t1_preview_filename
        Image.fromarray(t1_rgb).save(t1_preview_file)

        t2_preview_filename = f"t2_preview_{run_id}.png"
        t2_preview_file = static_dir / t2_preview_filename
        Image.fromarray(t2_rgb).save(t2_preview_file)

        overlay_filename = f"change_overlay_{run_id}.png"
        overlay_file = static_dir / overlay_filename
        Image.fromarray(overlay_rgb).save(overlay_file)

        # 7. Assemble Structured Evidence & Statistics
        statistics = [
            ZonalStatistic(
                metric_name="changed_pixels",
                display_name="Total Changed Pixels",
                value=float(changed_pixels),
                unit="pixels",
            ),
            ZonalStatistic(
                metric_name="change_ratio_percentage",
                display_name="Area Change Ratio",
                value=float(change_ratio_pct),
                unit="%",
            ),
        ]
        if area_m2 is not None and area_ha is not None:
            statistics.extend([
                ZonalStatistic(
                    metric_name="changed_area_m2",
                    display_name="Physical Changed Area",
                    value=float(area_m2),
                    unit="m^2",
                ),
                ZonalStatistic(
                    metric_name="changed_area_hectares",
                    display_name="Physical Changed Area (Hectares)",
                    value=float(area_ha),
                    unit="ha",
                ),
            ])

        evidence = EvidenceBundle(
            images=[
                EvidenceImage(
                    role="primary",
                    url=f"/api/v1/static/previews/{t1_preview_filename}",
                    width=width,
                    height=height,
                    crs=crs_str,
                    bounds=aligned.bounds,
                ),
                EvidenceImage(
                    role="secondary",
                    url=f"/api/v1/static/previews/{t2_preview_filename}",
                    width=width,
                    height=height,
                    crs=crs_str,
                    bounds=aligned.bounds,
                ),
                EvidenceImage(
                    role="change_overlay",
                    url=f"/api/v1/static/previews/{overlay_filename}",
                    width=width,
                    height=height,
                    crs=crs_str,
                    bounds=aligned.bounds,
                ),
            ],
            masks=[
                EvidenceMask(
                    mask_id=f"mask_{run_id}",
                    label="Physical Surface Change",
                    url=f"/api/v1/static/previews/{mask_filename}",
                    format="image/png",
                    palette={"0": "#00000000", "255": "#FF0033CC"},
                )
            ],
            boxes=bounding_boxes,
            statistics=statistics,
            regions=detected_regions,
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        confidence_val = round(float(np.mean(prob_map[binary_mask > 0])), 4) if changed_pixels > 0 else 0.85
        if changed_pixels == 0:
            answer_text = f"Bi-temporal change analysis completed. No significant physical surface change detected between {p1.name} and {p2.name}."
        else:
            area_str = f" ({area_m2:,.1f} m^2 / {area_ha:.2f} ha)" if area_m2 else ""
            answer_text = (
                f"Bi-temporal change detected across {changed_pixels:,} pixels ({change_ratio_pct:.2f}% of AOI){area_str}, "
                f"clustered in {len(bounding_boxes)} major spatial regions."
            )

        parameters_used = {
            "candidate_model": "cva" if candidate_req == "cva" else self.candidate,
            "image1": p1.name,
            "image2": p2.name,
            "mask_path": f"/api/v1/static/previews/{mask_filename}",
            "overlay_path": f"/api/v1/static/previews/{overlay_filename}",
            "width": width,
            "height": height,
            "changed_pixels": changed_pixels,
            "change_ratio_pct": change_ratio_pct,
            "total_clusters": len(bounding_boxes),
            "diagnostics": diag,
        }

        return SpecialistOutput(
            specialist_id="CHANGE_DETECT",
            success=True,
            answer_text=answer_text,
            confidence=confidence_val,
            evidence=evidence,
            parameters_used=parameters_used,
            warnings=warnings,
            execution_time_ms=elapsed_ms,
        )

    def health_check(self) -> bool:
        if self._is_loaded and hasattr(self.detector, "model"):
            return self.detector.model is not None
        return True

    def cleanup(self) -> None:
        if hasattr(self.detector, "cleanup"):
            self.detector.cleanup()
        self._is_loaded = False
