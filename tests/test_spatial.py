"""Unit tests for spatial coordinate transformations and GeoJSON generation."""

import pytest
from affine import Affine

from backend.evidence.spatial import PixelBoundingBox, SpatialTransformer


def test_pixel_bounding_box_convention():
    """Verify coordinate convention is strictly [xmin, ymin, xmax, ymax]."""
    bbox = PixelBoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=60.0, label="runway")
    assert bbox.xmin == 10.0
    assert bbox.ymin == 20.0
    assert bbox.xmax == 50.0
    assert bbox.ymax == 60.0

    # Validation within dimensions
    assert bbox.validate_bounds(width=100, height=100) is True
    assert bbox.validate_bounds(width=40, height=100) is False  # xmax out of bounds


def test_pixel_bounding_box_clamping():
    """Verify clamping to image boundaries."""
    bbox = PixelBoundingBox(xmin=-10.0, ymin=5.0, xmax=300.0, ymax=260.0)
    clamped = bbox.clamp(width=256, height=256)
    assert clamped.xmin == 0.0
    assert clamped.ymin == 5.0
    assert clamped.xmax == 256.0
    assert clamped.ymax == 256.0


def test_normalized_coordinates():
    """Verify normalization to [0.0, 1.0]."""
    bbox = PixelBoundingBox(xmin=64.0, ymin=128.0, xmax=192.0, ymax=256.0)
    norm = bbox.to_normalized(width=256, height=256)
    assert norm == (0.25, 0.50, 0.75, 1.0)


def test_pixel_to_geospatial_polygon_transformation():
    """Verify affine transformation maps pixel coordinates to exact projected CRS coordinates."""
    # Affine: origin at (725000.0, 3130000.0), 10m pixel resolution
    # X = 725000 + 10 * x
    # Y = 3130000 - 10 * y
    transform = Affine(10.0, 0.0, 725000.0, 0.0, -10.0, 3130000.0)
    crs = "EPSG:32643"

    bbox = PixelBoundingBox(xmin=10.0, ymin=20.0, xmax=50.0, ymax=80.0, label="water body", confidence=0.88)
    region = SpatialTransformer.pixel_bbox_to_geospatial_polygon(
        bbox=bbox,
        affine_transform=transform,
        crs=crs,
        width=256,
        height=256,
    )

    assert region.crs == "EPSG:32643"
    assert region.label == "water body"
    assert region.confidence == 0.88
    assert region.is_axis_aligned_polygon is True

    # Check GeoJSON Feature structure
    geojson = region.geojson_feature
    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "Polygon"

    coords = geojson["geometry"]["coordinates"][0]
    assert len(coords) == 5  # Closed loop: 4 corners + repeat of first

    # Expected corners:
    # TL: (725000 + 10*10, 3130000 - 10*20) = (725100.0, 3129800.0)
    # TR: (725000 + 10*50, 3130000 - 10*20) = (725500.0, 3129800.0)
    # BR: (725000 + 10*50, 3130000 - 10*80) = (725500.0, 3129200.0)
    # BL: (725000 + 10*10, 3130000 - 10*80) = (725100.0, 3129200.0)
    assert coords[0] == [725100.0, 3129800.0]
    assert coords[1] == [725500.0, 3129800.0]
    assert coords[2] == [725500.0, 3129200.0]
    assert coords[3] == [725100.0, 3129200.0]
    assert coords[4] == [725100.0, 3129800.0]  # Closed


def test_batch_transformation_with_invalid_boxes():
    """Verify batch transformer filters out malformed bounding boxes."""
    transform = Affine(10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0)
    valid_box = PixelBoundingBox(xmin=10, ymin=10, xmax=50, ymax=50)
    invalid_box = PixelBoundingBox(xmin=50, ymin=50, xmax=10, ymax=10)  # Inverted

    regions = SpatialTransformer.batch_transform_bboxes(
        bboxes=[valid_box, invalid_box],
        affine_transform=transform,
        crs="EPSG:32643",
        width=100,
        height=100,
    )
    assert len(regions) == 1
    assert regions[0].pixel_bbox == (10.0, 10.0, 50.0, 50.0)
