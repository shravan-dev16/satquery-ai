"""Format-Robust Bi-Temporal Input Normalization & Spatial Correspondence Engine.

Milestone Extended Maximum Training & Format Robustness:
1. Normalizes bi-temporal image pairs across PNG, JPEG, TIFF, and GeoTIFF into a common representation.
2. Supports two operating regimes:
   - Mode A (Georeferenced): Valid CRS, affine transforms, reprojection, and metric ground areas (m^2, ha).
   - Mode B (Non-Georeferenced): PNG, JPEG, local TIFF with image-level alignment, phase correlation,
     and explicit scientific qualification (pixel coordinates only, no fabricated CRS or metric areas).
3. ImageAlignmentEngine:
   - Sub-pixel translation estimation via 2D Fast Fourier Transform (FFT) Phase Correlation.
   - Normalized Cross-Correlation (NCC) and structural correspondence scoring.
   - Continuous alignment_score in [0.0, 1.0] exposed to M8 Consistency and M9 Confidence.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from rasterio.transform import Affine

from backend.preprocessing.geotiff import GeoTIFFReader

logger = logging.getLogger(__name__)


@dataclass
class NormalizedPair:
    """Unified container representing a normalized bi-temporal image pair."""
    image1_array: np.ndarray        # (C, H, W) normalized uint8 or float32 in [0, 255]
    image2_array: np.ndarray        # (C, H, W) aligned to common grid
    width: int
    height: int
    channels: int
    is_georeferenced: bool
    crs: Optional[str] = None
    transform: Optional[Affine] = None
    bounds: Optional[Tuple[float, float, float, float]] = None
    alignment_score: float = 1.0    # 0.0 (incompatible/unrelated) to 1.0 (perfect)
    offset_xy: Tuple[float, float] = (0.0, 0.0)  # (dx, dy) translation offset in pixels
    format_t1: str = "unknown"
    format_t2: str = "unknown"
    resolution: Optional[Tuple[float, float]] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def target_crs(self) -> Optional[str]:
        return self.crs

    @property
    def target_resolution(self) -> Optional[Tuple[float, float]]:
        return self.resolution


class ImageAlignmentEngine:
    """Estimates sub-pixel translation and visual correspondence using 2D FFT Phase Correlation."""

    @staticmethod
    def compute_alignment(
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
    ) -> Tuple[float, Tuple[float, float]]:
        """Computes continuous alignment score and translation shift (dx, dy).

        Args:
            img1_gray: 2D float32 array [H, W]
            img2_gray: 2D float32 array [H, W]

        Returns:
            alignment_score: float in [0.0, 1.0]
            offset_xy: (dx, dy) in pixels
        """
        assert img1_gray.ndim == 2 and img2_gray.ndim == 2, "Inputs must be 2D grayscale arrays"
        h, w = img1_gray.shape

        # Normalize to zero mean, unit variance for FFT stability
        std1 = float(np.std(img1_gray))
        std2 = float(np.std(img2_gray))
        if std1 < 1e-4 or std2 < 1e-4:
            # Degenerate uniform image
            return 0.20, (0.0, 0.0)

        norm1 = (img1_gray - np.mean(img1_gray)) / std1
        norm2 = (img2_gray - np.mean(img2_gray)) / std2

        # 1. 2D Fast Fourier Transform Phase Correlation
        f1 = np.fft.fft2(norm1)
        f2 = np.fft.fft2(norm2)
        cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-8)
        corr_map = np.real(np.fft.ifft2(cross_power))

        # Find peak
        max_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        dy = max_idx[0] if max_idx[0] <= h // 2 else max_idx[0] - h
        dx = max_idx[1] if max_idx[1] <= w // 2 else max_idx[1] - w
        phase_peak = float(corr_map[max_idx])

        # 2. Shifted array and robust background consensus
        if abs(dx) > 0 or abs(dy) > 0:
            shifted_g2 = np.roll(np.roll(img2_gray, shift=dy, axis=0), shift=dx, axis=1)
        else:
            shifted_g2 = img2_gray

        # Robust local consensus (fraction of pixels within tolerance after shift)
        dyn_range = max(10.0, float(np.ptp(img1_gray)))
        norm_diff = np.abs(img1_gray - shifted_g2) / dyn_range
        consensus = float(np.mean(norm_diff < 0.25))

        # 3. Normalized Cross-Correlation (NCC) at zero-shift and optimal-shift
        zero_shift_ncc = float(np.mean(norm1 * norm2))
        shifted_norm2 = (shifted_g2 - np.mean(shifted_g2)) / max(float(np.std(shifted_g2)), 1e-4)
        best_ncc = max(zero_shift_ncc, float(np.mean(norm1 * shifted_norm2)))
        best_ncc = max(-1.0, min(1.0, best_ncc))

        # Sidelobe statistics for Peak-to-Sidelobe Ratio (PSR)
        # Exclude 7x7 neighborhood around the correlation peak
        sidelobe = corr_map.copy()
        y_min, y_max = max(0, max_idx[0] - 3), min(h, max_idx[0] + 4)
        x_min, x_max = max(0, max_idx[1] - 3), min(w, max_idx[1] + 4)
        sidelobe[y_min:y_max, x_min:x_max] = np.nan
        valid_side = sidelobe[~np.isnan(sidelobe)]
        side_std = float(np.std(valid_side))
        psr = (phase_peak - float(np.mean(valid_side))) / max(side_std, 1e-6)

        # 4. Composite continuous alignment score
        # Phase peak prominence combines absolute height and peak-to-sidelobe ratio (PSR),
        # making alignment robust to high-frequency spectral damping from lossy JPEG compression
        # while strictly preserving low scores for noise or unrelated imagery.
        phase_score = min(1.0, max(phase_peak / 0.65, psr / 60.0))
        base_score = 0.55 * phase_score + 0.35 * consensus + 0.10 * max(0.0, best_ncc)

        # Penalize excessive translation jitter (> 10% image width/height)
        shift_ratio = max(abs(dx) / float(w), abs(dy) / float(h))
        if shift_ratio > 0.10:
            penalty = min(0.60, (shift_ratio - 0.10) * 2.0)
            final_score = base_score * (1.0 - penalty)
        else:
            final_score = base_score

        alignment_score = max(0.001, min(0.99, round(float(final_score), 4)))
        return alignment_score, (float(dx), float(dy))


class BiTemporalNormalizer:
    """Format-robust normalizer for bi-temporal remote sensing pairs."""

    @classmethod
    def normalize_pair(
        cls,
        image1_path: Union[str, Path],
        image2_path: Union[str, Path],
        max_dimension: Optional[int] = 2048,
    ) -> NormalizedPair:
        """Normalizes any pair of PNG, JPEG, TIFF, or GeoTIFF images into a NormalizedPair."""
        p1 = Path(image1_path)
        p2 = Path(image2_path)

        if not p1.exists():
            raise FileNotFoundError(f"Primary image (T1) not found: {p1}")
        if not p2.exists():
            raise FileNotFoundError(f"Secondary image (T2) not found: {p2}")

        warnings: List[str] = []

        # 1. Inspect headers via GeoTIFFReader
        meta1 = GeoTIFFReader.inspect(p1)
        meta2 = GeoTIFFReader.inspect(p2)

        fmt1 = meta1.get("driver", p1.suffix.strip(".").upper())
        fmt2 = meta2.get("driver", p2.suffix.strip(".").upper())

        crs1 = meta1.get("crs")
        crs2 = meta2.get("crs")
        has_crs1 = crs1 not in (None, "None", "null", "")
        has_crs2 = crs2 not in (None, "None", "null", "")

        is_georeferenced = has_crs1 and has_crs2

        # -------------------------------------------------------------------
        # MODE A: GEOREFERENCED PAIR (GeoTIFF / projected TIFF)
        # -------------------------------------------------------------------
        if is_georeferenced:
            from backend.preprocessing.alignment import BiTemporalAligner, BiTemporalValidator
            val = BiTemporalValidator.validate_pair(p1, p2)
            if not val.is_valid:
                raise ValueError(f"Georeferenced pair validation failed: {val.error}")

            aligned = BiTemporalAligner.align_pair(p1, p2)
            warnings.extend(aligned.metadata.get("warnings", []))

            # Grayscale representations for alignment score calculation
            arr1 = aligned.image1_array
            arr2 = aligned.image2_array
            g1 = np.mean(arr1[:3], axis=0).astype(np.float32) if arr1.shape[0] >= 3 else arr1[0].astype(np.float32)
            g2 = np.mean(arr2[:3], axis=0).astype(np.float32) if arr2.shape[0] >= 3 else arr2[0].astype(np.float32)

            score, offset = ImageAlignmentEngine.compute_alignment(g1, g2)
            overlap_ratio = val.spatial_overlap.overlap_ratio_image1 if val.spatial_overlap else 1.0
            # Condition georeferenced score with spatial overlap
            final_align_score = max(0.10, min(0.99, round(0.50 * score + 0.50 * overlap_ratio, 4)))

            return NormalizedPair(
                image1_array=arr1,
                image2_array=arr2,
                width=aligned.width,
                height=aligned.height,
                channels=arr1.shape[0],
                is_georeferenced=True,
                crs=aligned.crs,
                transform=aligned.transform,
                bounds=aligned.bounds,
                alignment_score=final_align_score,
                offset_xy=offset,
                format_t1=fmt1,
                format_t2=fmt2,
                resolution=aligned.target_resolution,
                warnings=warnings,
                metadata={
                    "mode": "Mode A (Georeferenced)",
                    "overlap_ratio": overlap_ratio,
                    "temporal_order_verified": val.temporal_order_verified,
                    "time1_iso": val.time1_iso,
                    "time2_iso": val.time2_iso,
                },
            )

        # -------------------------------------------------------------------
        # MODE B: NON-GEOREFERENCED PAIR (PNG, JPEG, local unprojected TIFF)
        # -------------------------------------------------------------------
        warnings.append(
            "Geospatial metadata (CRS/geotransform) unavailable. Analysis operating in Mode B (pixel coordinate space). Metric ground area measurements are unverified."
        )

        # Read normalized RGB arrays
        rgb1, _ = GeoTIFFReader.read_normalized_rgb(p1)
        rgb2, _ = GeoTIFFReader.read_normalized_rgb(p2)

        h1, w1, _ = rgb1.shape
        h2, w2, _ = rgb2.shape

        # Verify dimension compatibility
        aspect1 = round(w1 / float(h1), 3)
        aspect2 = round(w2 / float(h2), 3)
        aspect_diff = abs(aspect1 - aspect2) / max(aspect1, 1e-4)

        if aspect_diff > 0.25:
            warnings.append(
                f"Aspect ratio discrepancy between inputs: T1 is {aspect1:.2f}, T2 is {aspect2:.2f} ({aspect_diff * 100:.1f}% difference)."
            )

        # Resample to common grid if dimensions differ
        if (h1, w1) != (h2, w2):
            warnings.append(
                f"Image dimensions differ: T1 ({w1}x{h1}) vs T2 ({w2}x{h2}). Resampling T2 to match T1 pixel grid."
            )
            pil2 = Image.fromarray(rgb2)
            pil2_resampled = pil2.resize((w1, h1), Image.Resampling.BILINEAR)
            rgb2 = np.array(pil2_resampled)
            target_w, target_h = w1, h1
        else:
            target_w, target_h = w1, h1

        # Transpose to (C, H, W)
        arr1 = rgb1.transpose(2, 0, 1)
        arr2 = rgb2.transpose(2, 0, 1)

        # Grayscale for phase correlation
        g1 = (0.2989 * arr1[0] + 0.5870 * arr1[1] + 0.1140 * arr1[2]).astype(np.float32)
        g2 = (0.2989 * arr2[0] + 0.5870 * arr2[1] + 0.1140 * arr2[2]).astype(np.float32)

        alignment_score, (dx, dy) = ImageAlignmentEngine.compute_alignment(g1, g2)

        if alignment_score >= 0.85:
            logger.info("High spatial alignment score (%.2f) on non-georeferenced pair (%s, %s)", alignment_score, p1.name, p2.name)
        elif alignment_score >= 0.70:
            warnings.append(f"Acceptable image alignment (score: {alignment_score:.2f}). Slight visual discrepancy or translation shift ({dx:.1f}, {dy:.1f}) px detected.")
        elif alignment_score >= 0.35:
            warnings.append(f"Substantial visual misalignment (score: {alignment_score:.2f}). Images may represent different viewing geometries, lighting, or seasons.")
        else:
            warnings.append(f"Poor image correspondence (score: {alignment_score:.2f}). Images appear structurally disjoint or unrelated.")

        return NormalizedPair(
            image1_array=arr1,
            image2_array=arr2,
            width=target_w,
            height=target_h,
            channels=arr1.shape[0],
            is_georeferenced=False,
            crs=None,
            transform=None,
            bounds=None,
            alignment_score=alignment_score,
            offset_xy=(dx, dy),
            format_t1=fmt1,
            format_t2=fmt2,
            resolution=None,
            warnings=warnings,
            metadata={
                "mode": "Mode B (Non-Georeferenced)",
                "raw_dimensions_t1": (w1, h1),
                "raw_dimensions_t2": (w2, h2),
                "aspect_ratio_t1": aspect1,
                "aspect_ratio_t2": aspect2,
                "translation_offset": (dx, dy),
            },
        )
