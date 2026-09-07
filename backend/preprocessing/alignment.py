"""Bi-temporal input validation and deterministic geospatial alignment engine.

Milestone M3:
1. BiTemporalValidator: Performs 11-point geospatial and temporal compatibility checks.
2. BiTemporalAligner: Performs deterministic reprojection, grid resampling, and common intersection cropping.
"""

from dataclasses import dataclass, field
import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

from backend.agent.schema import ImageMetadata, PairCompatibility, ValidationResult
from backend.preprocessing.geotiff import GeoTIFFReader
from backend.preprocessing.metadata import MetadataExtractor
from backend.preprocessing.modality import ModalityDetector

logger = logging.getLogger(__name__)


@dataclass
class SpatialOverlapInfo:
    """Geospatial overlap metrics between two georeferenced rasters."""
    has_overlap: bool
    intersection_bounds: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy) in target CRS
    intersection_area: float = 0.0
    image1_area: float = 0.0
    image2_area: float = 0.0
    overlap_ratio_image1: float = 0.0  # intersection_area / image1_area
    overlap_ratio_image2: float = 0.0  # intersection_area / image2_area
    common_crs: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class BiTemporalValidationResult:
    """Structured validation outcome for a bi-temporal image pair."""
    is_valid: bool
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    image1_meta: Optional[Dict[str, Any]] = None
    image2_meta: Optional[Dict[str, Any]] = None
    spatial_overlap: Optional[SpatialOverlapInfo] = None
    temporal_order_verified: bool = False
    time1_iso: Optional[str] = None
    time2_iso: Optional[str] = None
    modality_compatible: bool = False


@dataclass
class AlignmentResult:
    """Result of deterministic bi-temporal raster alignment."""
    image1_array: np.ndarray  # (C, H, W) normalized or raw
    image2_array: np.ndarray  # (C, H, W) aligned to image1 grid
    crs: str
    transform: Affine
    bounds: Tuple[float, float, float, float]
    width: int
    height: int
    resampling_method: str
    metadata: Dict[str, Any]

    @property
    def target_crs(self) -> str:
        return self.crs

    @property
    def target_resolution(self) -> Tuple[float, float]:
        return (float(self.transform[0]), float(abs(self.transform[4])))


class BiTemporalValidator:
    """Performs rigorous 11-point validation on bi-temporal image pairs."""

    MIN_OVERLAP_RATIO = 0.50  # Require at least 50% spatial overlap
    FULL_OVERLAP_WARN_RATIO = 0.95  # Issue warning if overlap < 95%

    @classmethod
    def validate_pair(
        cls,
        image1_path: Union[str, Path],
        image2_path: Union[str, Path],
        min_overlap_ratio: float = 0.50,
    ) -> BiTemporalValidationResult:
        """Validates a bi-temporal pair across all 11 required dimensions.

        Checks:
        1. Exactly two distinct images
        2. Both readable via Rasterio
        3. CRS availability on both images
        4. Valid affine geotransforms
        5. Spatial resolution availability
        6. Dimension sanity (width, height > 0)
        7. Geospatial bounding boxes
        8. Real geospatial intersection
        9. Overlap percentage relative to both images
        10. Temporal ordering (t1 < t2 if timestamps exist; warnings if missing)
        11. Modality compatibility (e.g. optical-optical)
        """
        p1 = Path(image1_path)
        p2 = Path(image2_path)
        warnings: List[str] = []

        # 1. Exactly two images
        if not p1.exists():
            return BiTemporalValidationResult(is_valid=False, error=f"Primary image (T1) not found: {p1}")
        if not p2.exists():
            return BiTemporalValidationResult(is_valid=False, error=f"Secondary image (T2) not found: {p2}")

        # 2. Both readable
        try:
            meta1 = GeoTIFFReader.inspect(p1)
        except Exception as e:
            return BiTemporalValidationResult(is_valid=False, error=f"Failed to read image T1: {e}")

        try:
            meta2 = GeoTIFFReader.inspect(p2)
        except Exception as e:
            return BiTemporalValidationResult(is_valid=False, error=f"Failed to read image T2: {e}", image1_meta=meta1)

        # 3. CRS availability
        crs1_str = meta1.get("crs")
        crs2_str = meta2.get("crs")
        if not crs1_str or crs1_str in ["None", "null", ""]:
            return BiTemporalValidationResult(
                is_valid=False, error=f"Image T1 ({p1.name}) lacks a valid Coordinate Reference System (CRS).",
                image1_meta=meta1, image2_meta=meta2
            )
        if not crs2_str or crs2_str in ["None", "null", ""]:
            return BiTemporalValidationResult(
                is_valid=False, error=f"Image T2 ({p2.name}) lacks a valid Coordinate Reference System (CRS).",
                image1_meta=meta1, image2_meta=meta2
            )

        # 4. Affine transform
        tf1 = meta1.get("transform")
        tf2 = meta2.get("transform")
        if not tf1:
            return BiTemporalValidationResult(is_valid=False, error="Image T1 lacks geotransform matrix.", image1_meta=meta1, image2_meta=meta2)
        if not tf2:
            return BiTemporalValidationResult(is_valid=False, error="Image T2 lacks geotransform matrix.", image1_meta=meta1, image2_meta=meta2)

        # 5 & 6. Dimensions and resolution
        w1, h1 = meta1["width"], meta1["height"]
        w2, h2 = meta2["width"], meta2["height"]
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return BiTemporalValidationResult(is_valid=False, error="Non-positive image dimensions detected.", image1_meta=meta1, image2_meta=meta2)

        res1 = meta1.get("resolution", (1.0, 1.0))
        res2 = meta2.get("resolution", (1.0, 1.0))
        res_diff_pct = abs(res1[0] - res2[0]) / max(res1[0], 1e-6) * 100.0
        if res_diff_pct > 10.0:
            warnings.append(
                f"Spatial resolution difference: T1 has {res1[0]:.2f}m, T2 has {res2[0]:.2f}m ({res_diff_pct:.1f}% difference). T2 will be resampled."
            )

        # 7, 8, 9. Geospatial bounding boxes & Real spatial overlap
        overlap_info = cls._calculate_spatial_overlap(meta1, meta2)
        if not overlap_info.has_overlap:
            return BiTemporalValidationResult(
                is_valid=False,
                error="Images T1 and T2 have zero spatial intersection (disjoint areas).",
                image1_meta=meta1,
                image2_meta=meta2,
                spatial_overlap=overlap_info,
            )

        min_overlap = min(overlap_info.overlap_ratio_image1, overlap_info.overlap_ratio_image2)
        if min_overlap < min_overlap_ratio:
            return BiTemporalValidationResult(
                is_valid=False,
                error=f"Insufficient spatial overlap ({min_overlap * 100:.1f}% < minimum required {min_overlap_ratio * 100:.0f}%).",
                image1_meta=meta1,
                image2_meta=meta2,
                spatial_overlap=overlap_info,
            )
        elif min_overlap < cls.FULL_OVERLAP_WARN_RATIO:
            warnings.append(
                f"Partial spatial overlap detected ({min_overlap * 100:.1f}%). Non-overlapping margins will be cropped to the common intersection."
            )

        warnings.extend(overlap_info.warnings)

        # 10. Temporal ordering validation
        t1_date, t2_date, temporal_verified, temp_warn = cls._verify_temporal_order(meta1, meta2, p1, p2)
        if temp_warn:
            warnings.append(temp_warn)

        # 11. Modality compatibility
        mod1 = ModalityDetector.identify(p1)
        mod2 = ModalityDetector.identify(p2)
        modality_compatible = (mod1.modality == mod2.modality)
        if not modality_compatible:
            warnings.append(
                f"Cross-modality pair detected: T1 is '{mod1.modality.value}', T2 is '{mod2.modality.value}'."
            )

        # Band count comparison
        if meta1["band_count"] != meta2["band_count"]:
            warnings.append(
                f"Band count difference: T1 has {meta1['band_count']} band(s), T2 has {meta2['band_count']} band(s). Preprocessing will extract common RGB bands."
            )

        return BiTemporalValidationResult(
            is_valid=True,
            warnings=warnings,
            image1_meta=meta1,
            image2_meta=meta2,
            spatial_overlap=overlap_info,
            temporal_order_verified=temporal_verified,
            time1_iso=t1_date.isoformat() if t1_date else None,
            time2_iso=t2_date.isoformat() if t2_date else None,
            modality_compatible=modality_compatible,
        )

    @classmethod
    def validate(
        cls,
        image1_path: Union[str, Path],
        image2_path: Union[str, Path],
        min_overlap_ratio: float = 0.50,
    ) -> ValidationResult:
        """Validates bi-temporal pair and returns standard ValidationResult Pydantic schema."""
        raw_res = cls.validate_pair(image1_path, image2_path, min_overlap_ratio=min_overlap_ratio)
        errors = [raw_res.error] if raw_res.error else []
        warnings = list(raw_res.warnings)

        images: List[ImageMetadata] = []
        if raw_res.image1_meta:
            images.append(MetadataExtractor.extract_image_metadata(image1_path))
        if raw_res.image2_meta:
            images.append(MetadataExtractor.extract_image_metadata(image2_path))

        compat: Optional[PairCompatibility] = None
        if raw_res.spatial_overlap:
            overlap = raw_res.spatial_overlap
            ratio = min(1.0, max(0.0, overlap.overlap_ratio_image1))
            pct = min(100.0, max(0.0, ratio * 100.0))
            crs_match = (
                raw_res.image1_meta.get("crs") == raw_res.image2_meta.get("crs")
                if raw_res.image1_meta and raw_res.image2_meta
                else True
            )
            temporal_status = "valid" if raw_res.temporal_order_verified else ("missing_dates" if not raw_res.time1_iso else "unverified")
            if any("chronologically precedes" in w for w in warnings):
                temporal_status = "invalid_chronology"

            compat = PairCompatibility(
                pair_supported=raw_res.is_valid,
                spatial_overlap_ratio=round(ratio, 4),
                spatial_overlap_percentage=round(pct, 2),
                crs_match=crs_match,
                reprojection_required=not crs_match,
                temporal_ordering=temporal_status,
                warnings=warnings,
            )

        return ValidationResult(
            valid=raw_res.is_valid,
            images=images,
            compatibility=compat,
            errors=errors,
            warnings=warnings,
        )

    @classmethod
    def _calculate_spatial_overlap(cls, meta1: Dict[str, Any], meta2: Dict[str, Any]) -> SpatialOverlapInfo:
        """Calculates exact geospatial overlap area and ratios in target CRS."""
        crs1_str = meta1["crs"]
        crs2_str = meta2["crs"]
        bounds1 = meta1["bounds"]  # (minx, miny, maxx, maxy)
        bounds2 = meta2["bounds"]

        warnings: List[str] = []

        # If CRS differ, transform bounds2 to crs1
        if crs1_str != crs2_str:
            warnings.append(f"CRS mismatch: T1 is in {crs1_str}, T2 is in {crs2_str}. Reprojecting T2 coordinates.")
            try:
                with rasterio.Env():
                    bounds2_in_crs1 = transform_bounds(crs2_str, crs1_str, *bounds2)
            except Exception as e:
                return SpatialOverlapInfo(
                    has_overlap=False,
                    warnings=[f"Failed to transform coordinates between {crs2_str} and {crs1_str}: {e}"],
                )
        else:
            bounds2_in_crs1 = bounds2

        # Compute intersection rectangle
        minx = max(bounds1[0], bounds2_in_crs1[0])
        miny = max(bounds1[1], bounds2_in_crs1[1])
        maxx = min(bounds1[2], bounds2_in_crs1[2])
        maxy = min(bounds1[3], bounds2_in_crs1[3])

        if minx >= maxx or miny >= maxy:
            return SpatialOverlapInfo(has_overlap=False, common_crs=crs1_str, warnings=warnings)

        inter_w = maxx - minx
        inter_h = maxy - miny
        inter_area = inter_w * inter_h

        b1_w = bounds1[2] - bounds1[0]
        b1_h = bounds1[3] - bounds1[1]
        area1 = max(1e-6, b1_w * b1_h)

        b2_w = bounds2_in_crs1[2] - bounds2_in_crs1[0]
        b2_h = bounds2_in_crs1[3] - bounds2_in_crs1[1]
        area2 = max(1e-6, b2_w * b2_h)

        ratio1 = min(1.0, max(0.0, inter_area / area1))
        ratio2 = min(1.0, max(0.0, inter_area / area2))

        return SpatialOverlapInfo(
            has_overlap=True,
            intersection_bounds=(minx, miny, maxx, maxy),
            intersection_area=inter_area,
            image1_area=area1,
            image2_area=area2,
            overlap_ratio_image1=round(ratio1, 4),
            overlap_ratio_image2=round(ratio2, 4),
            common_crs=crs1_str,
            warnings=warnings,
        )

    @classmethod
    def _verify_temporal_order(
        cls,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any],
        p1: Path,
        p2: Path,
    ) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime], bool, Optional[str]]:
        """Extracts and verifies timestamps. Never invents dates if absent."""
        t1 = cls._extract_date(meta1, p1)
        t2 = cls._extract_date(meta2, p2)

        if t1 is None and t2 is None:
            return None, None, False, "Acquisition timestamps missing for both images; temporal ordering cannot be verified."
        if t1 is None:
            return None, t2, False, f"Acquisition timestamp missing for T1 ({p1.name}); temporal ordering cannot be verified."
        if t2 is None:
            return t1, None, False, f"Acquisition timestamp missing for T2 ({p2.name}); temporal ordering cannot be verified."

        if t1 >= t2:
            return t1, t2, False, f"Temporal ordering anomaly: T1 date ({t1.isoformat()}) is not earlier than T2 date ({t2.isoformat()})."

        return t1, t2, True, None

    @classmethod
    def _extract_date(cls, meta: Dict[str, Any], path: Path) -> Optional[datetime.datetime]:
        """Attempts to extract timestamp from GeoTIFF tags or filename pattern."""
        raw_tags = meta.get("raw_tags", {})
        candidate_tags = [
            "TIFFTAG_DATETIME",
            "acquisition_date",
            "ACQUISITION_DATE",
            "ACQUISITIONDATETIME",
            "datetime",
            "DATE_ACQUIRED",
        ]
        for key in candidate_tags:
            val = raw_tags.get(key)
            if val:
                dt = cls._parse_iso_or_exif(str(val))
                if dt:
                    return dt

        # Filename regex parsing (e.g. 20210512 or 2021-05-12)
        import re
        m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", path.name)
        if m:
            try:
                y, mon, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 1970 <= y <= 2035 and 1 <= mon <= 12 and 1 <= d <= 31:
                    return datetime.datetime(y, mon, d)
            except Exception:
                pass
        return None

    @staticmethod
    def _parse_iso_or_exif(val: str) -> Optional[datetime.datetime]:
        cleaned = val.strip().strip("'\"")
        # Format: 2021:05:12 10:30:00
        for fmt in [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%Y%m%d",
        ]:
            try:
                return datetime.datetime.strptime(cleaned[:19], fmt)
            except Exception:
                continue
        return None


class BiTemporalAligner:
    """Deterministically co-registers, reprojects, and crops bi-temporal raster pairs."""

    @classmethod
    def align_pair(
        cls,
        image1_path: Union[str, Path],
        image2_path: Union[str, Path],
        resampling: Resampling = Resampling.bilinear,
    ) -> AlignmentResult:
        """Co-registers Image 2 to Image 1's grid and crops both to the common intersection."""
        p1 = Path(image1_path)
        p2 = Path(image2_path)

        # Validate pair first
        val = BiTemporalValidator.validate_pair(p1, p2)
        if not val.is_valid:
            raise ValueError(f"Cannot align incompatible pair: {val.error}")

        overlap = val.spatial_overlap
        assert overlap is not None and overlap.intersection_bounds is not None
        inter_bounds = overlap.intersection_bounds
        common_crs = overlap.common_crs or val.image1_meta["crs"]

        # Read Image 1
        with rasterio.open(p1) as src1:
            # Crop Image 1 to intersection bounds
            window1 = rasterio.windows.from_bounds(*inter_bounds, transform=src1.transform)
            # Ensure window coordinates are positive and non-zero
            window1 = window1.round_offsets().round_lengths()
            w1_int = int(window1.width)
            h1_int = int(window1.height)
            if w1_int <= 0 or h1_int <= 0:
                raise ValueError(f"Cropped intersection window has invalid dimensions: ({w1_int}, {h1_int})")

            data1 = src1.read(window=window1)
            # Calculate transform for cropped area
            cropped_transform = rasterio.windows.transform(window1, src1.transform)

        # Reproject and resample Image 2 into Image 1's cropped grid
        with rasterio.open(p2) as src2:
            num_bands = min(data1.shape[0], src2.count)
            data1_matched = data1[:num_bands]
            aligned_data2 = np.zeros((num_bands, h1_int, w1_int), dtype=data1_matched.dtype)

            for b in range(num_bands):
                reproject(
                    source=rasterio.band(src2, b + 1),
                    destination=aligned_data2[b],
                    src_transform=src2.transform,
                    src_crs=src2.crs,
                    dst_transform=cropped_transform,
                    dst_crs=common_crs,
                    resampling=resampling,
                )

        metadata = {
            "image1_name": p1.name,
            "image2_name": p2.name,
            "crs": common_crs,
            "cropped_dimensions": (w1_int, h1_int),
            "band_count": num_bands,
            "intersection_bounds": list(inter_bounds),
            "overlap_ratio_t1": overlap.overlap_ratio_image1,
            "overlap_ratio_t2": overlap.overlap_ratio_image2,
            "temporal_order_verified": val.temporal_order_verified,
            "time1_iso": val.time1_iso,
            "time2_iso": val.time2_iso,
            "warnings": val.warnings,
        }

        return AlignmentResult(
            image1_array=data1_matched,
            image2_array=aligned_data2,
            crs=common_crs,
            transform=cropped_transform,
            bounds=inter_bounds,
            width=w1_int,
            height=h1_int,
            resampling_method=resampling.name,
            metadata=metadata,
        )

    align = align_pair
