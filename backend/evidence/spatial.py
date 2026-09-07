"""Spatial evidence transformation utilities.

Converts pixel-space bounding boxes and contours to georeferenced geospatial geometries
(GeoJSON Polygons) using the raster's affine transform and Coordinate Reference System (CRS).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from affine import Affine
from pydantic import BaseModel, Field


class PixelBoundingBox(BaseModel):
    """Bounding box in raster pixel coordinates.

    Coordinate Convention:
    - Origin (0, 0) is at the TOP-LEFT corner of the raster.
    - x increases horizontally to the right: 0 <= x <= width
    - y increases vertically downwards: 0 <= y <= height
    - Ordering: [xmin, ymin, xmax, ymax]
    """
    xmin: float = Field(description="Left horizontal column index")
    ymin: float = Field(description="Top vertical row index")
    xmax: float = Field(description="Right horizontal column index")
    ymax: float = Field(description="Bottom vertical row index")
    label: str = Field(default="detected_region")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_degenerate: bool = Field(default=False)
    degenerate_reason: Optional[str] = None

    @property
    def area(self) -> float:
        """Returns the bounding box area in pixels (0.0 if inverted or non-positive)."""
        import math
        if any(math.isnan(c) or math.isinf(c) for c in (self.xmin, self.ymin, self.xmax, self.ymax)):
            return 0.0
        w = max(0.0, self.xmax - self.xmin)
        h = max(0.0, self.ymax - self.ymin)
        return w * h

    def validate_bounds(self, width: int, height: int) -> bool:
        """Verifies coordinates are logically consistent and within image bounds."""
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            return False
        if self.xmin < 0 or self.ymin < 0 or self.xmax > width or self.ymax > height:
            return False
        return True

    def check_degeneracy(
        self, width: int, height: int, threshold_pct: float = 0.98
    ) -> Tuple[bool, Optional[str]]:
        """Identifies degenerate bounding boxes per M2.1 rules (e.g. >= 98% coverage or zero-area)."""
        import math
        if any(math.isnan(c) or math.isinf(c) for c in (self.xmin, self.ymin, self.xmax, self.ymax)):
            return True, "non_finite_coordinates"
        if self.xmin > self.xmax or self.ymin > self.ymax:
            return True, "inverted_coordinates"
        box_w = self.xmax - self.xmin
        box_h = self.ymax - self.ymin
        if box_w <= 0 or box_h <= 0:
            return True, "zero_area"
        area = box_w * box_h
        img_area = float(width * height)
        if img_area > 0 and (area / img_area) >= threshold_pct:
            return True, f"full_image_coverage_{round((area / img_area) * 100, 1)}pct"
        if box_w >= width * 0.99 and box_h >= height * 0.99:
            return True, "full_image_fallback"
        return False, None

    def clamp(self, width: int, height: int) -> "PixelBoundingBox":
        """Clamps bounding box coordinates strictly within image dimensions."""
        clamped_xmin = max(0.0, min(float(width), self.xmin))
        clamped_ymin = max(0.0, min(float(height), self.ymin))
        clamped_xmax = max(clamped_xmin + 1.0, min(float(width), self.xmax))
        clamped_ymax = max(clamped_ymin + 1.0, min(float(height), self.ymax))
        is_degen, reason = self.check_degeneracy(width, height)
        return PixelBoundingBox(
            xmin=round(clamped_xmin, 2),
            ymin=round(clamped_ymin, 2),
            xmax=round(clamped_xmax, 2),
            ymax=round(clamped_ymax, 2),
            label=self.label,
            confidence=self.confidence,
            model_score=self.model_score,
            is_degenerate=self.is_degenerate or is_degen,
            degenerate_reason=self.degenerate_reason or reason,
        )

    def to_normalized(self, width: int, height: int) -> Tuple[float, float, float, float]:
        """Returns normalized coordinates [xmin, ymin, xmax, ymax] in [0.0, 1.0]."""
        return (
            round(self.xmin / width, 4),
            round(self.ymin / height, 4),
            round(self.xmax / width, 4),
            round(self.ymax / height, 4),
        )


class GeospatialRegion(BaseModel):
    """Georeferenced spatial evidence derived from a pixel bounding box."""
    pixel_bbox: Tuple[float, float, float, float] = Field(
        description="[xmin, ymin, xmax, ymax] in pixel space"
    )
    normalized_bbox: Tuple[float, float, float, float] = Field(
        description="[xmin, ymin, xmax, ymax] normalized to [0, 1]"
    )
    label: str
    confidence: Optional[float] = None
    crs: Optional[str] = None
    geojson_feature: Dict[str, Any] = Field(
        description="GeoJSON Feature with Polygon geometry in projected CRS coordinates"
    )
    is_axis_aligned_polygon: bool = Field(
        default=True,
        description="True indicates geometry is an axis-aligned polygon projected from a bbox",
    )


class SpatialTransformer:
    """Transforms pixel coordinates to projected geospatial geometries."""

    @staticmethod
    def pixel_bbox_to_geospatial_polygon(
        bbox: PixelBoundingBox,
        affine_transform: Union[Affine, Tuple[float, ...]],
        crs: Optional[str],
        width: int,
        height: int,
    ) -> GeospatialRegion:
        """Converts a pixel bounding box [xmin, ymin, xmax, ymax] to a projected GeoJSON Polygon.

        Uses the exact raster affine transform matrix:
            X = a * x + b * y + c
            Y = d * x + e * y + f

        Returns a 5-point closed polygon: [TL, TR, BR, BL, TL].
        """
        # Ensure valid clamp
        c_bbox = bbox.clamp(width, height)

        if isinstance(affine_transform, (list, tuple)):
            fwd_transform = Affine(*affine_transform[:6])
        else:
            fwd_transform = affine_transform

        # 4 corners in pixel space (x, y)
        # Top-Left:     (xmin, ymin)
        # Top-Right:    (xmax, ymin)
        # Bottom-Right: (xmax, ymax)
        # Bottom-Left:  (xmin, ymax)
        pixel_corners = [
            (c_bbox.xmin, c_bbox.ymin),
            (c_bbox.xmax, c_bbox.ymin),
            (c_bbox.xmax, c_bbox.ymax),
            (c_bbox.xmin, c_bbox.ymax),
            (c_bbox.xmin, c_bbox.ymin),  # Closed ring
        ]

        # Project corners into CRS coordinates
        geo_coords: List[List[float]] = []
        for px, py in pixel_corners:
            gx, gy = fwd_transform * (px, py)
            geo_coords.append([round(gx, 4), round(gy, 4)])

        geojson_feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [geo_coords],
            },
            "properties": {
                "label": c_bbox.label,
                "confidence": c_bbox.confidence,
                "crs": crs or "pixel",
                "derivation": "axis_aligned_bounding_box",
                "pixel_bbox": [c_bbox.xmin, c_bbox.ymin, c_bbox.xmax, c_bbox.ymax],
            },
        }

        norm_bbox = c_bbox.to_normalized(width, height)

        return GeospatialRegion(
            pixel_bbox=(c_bbox.xmin, c_bbox.ymin, c_bbox.xmax, c_bbox.ymax),
            normalized_bbox=norm_bbox,
            label=c_bbox.label,
            confidence=c_bbox.confidence,
            crs=crs,
            geojson_feature=geojson_feature,
            is_axis_aligned_polygon=True,
        )

    @staticmethod
    def batch_transform_bboxes(
        bboxes: List[PixelBoundingBox],
        affine_transform: Union[Affine, Tuple[float, ...]],
        crs: Optional[str],
        width: int,
        height: int,
    ) -> List[GeospatialRegion]:
        """Transforms a collection of pixel bounding boxes to geospatial regions."""
        regions: List[GeospatialRegion] = []
        for bbox in bboxes:
            if bbox.validate_bounds(width, height):
                reg = SpatialTransformer.pixel_bbox_to_geospatial_polygon(
                    bbox, affine_transform, crs, width, height
                )
                regions.append(reg)
        return regions
