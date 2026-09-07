"""Unit tests for bi-temporal input validation and geospatial alignment."""

import datetime
from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from backend.preprocessing.alignment import (
    BiTemporalAligner,
    BiTemporalValidator,
    SpatialOverlapInfo,
)


@pytest.fixture
def bitemporal_synthetic_paths(tmp_path: Path):
    """Creates synthetic paired GeoTIFFs with controllable bounds, CRS, and dates."""
    t1_path = tmp_path / "test_20210101_t1.tif"
    t2_path = tmp_path / "test_20220101_t2.tif"

    # Base grid: 100x100 pixels, EPSG:32643, bounds: (725000, 3129000, 726000, 3130000)
    transform = from_bounds(725000, 3129000, 726000, 3130000, 100, 100)
    data1 = np.full((3, 100, 100), 50, dtype=np.uint8)
    data2 = np.full((3, 100, 100), 70, dtype=np.uint8)

    for path, data in [(t1_path, data1), (t2_path, data2)]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=3,
            dtype=np.uint8,
            crs="EPSG:32643",
            transform=transform,
        ) as dst:
            dst.write(data)

    return t1_path, t2_path


def test_bitemporal_validation_success(bitemporal_synthetic_paths):
    """Verify validation passes on aligned pair and detects temporal order from filename."""
    t1_path, t2_path = bitemporal_synthetic_paths
    result = BiTemporalValidator.validate_pair(t1_path, t2_path)

    assert result.is_valid is True
    assert result.error is None
    assert result.temporal_order_verified is True
    assert result.time1_iso == "2021-01-01T00:00:00"
    assert result.time2_iso == "2022-01-01T00:00:00"
    assert result.spatial_overlap.overlap_ratio_image1 == 1.0
    assert result.spatial_overlap.overlap_ratio_image2 == 1.0


def test_missing_timestamps_emits_warning(tmp_path: Path):
    """Verify missing acquisition dates produce explicit warnings without inventing timestamps."""
    p1 = tmp_path / "scene_alpha.tif"
    p2 = tmp_path / "scene_beta.tif"

    transform = from_bounds(100, 100, 200, 200, 50, 50)
    data = np.zeros((3, 50, 50), dtype=np.uint8)
    for p in [p1, p2]:
        with rasterio.open(
            p, "w", driver="GTiff", height=50, width=50, count=3, dtype=np.uint8, crs="EPSG:32632", transform=transform
        ) as dst:
            dst.write(data)

    res = BiTemporalValidator.validate_pair(p1, p2)
    assert res.is_valid is True
    assert res.temporal_order_verified is False
    assert any("temporal ordering cannot be verified" in w for w in res.warnings)


def test_reversed_temporal_order_flagged(tmp_path: Path):
    """Verify reversed dates (t1 > t2) are flagged with an explicit temporal warning."""
    p1 = tmp_path / "img_20230501.tif"
    p2 = tmp_path / "img_20200101.tif"

    transform = from_bounds(100, 100, 200, 200, 50, 50)
    data = np.zeros((3, 50, 50), dtype=np.uint8)
    for p in [p1, p2]:
        with rasterio.open(
            p, "w", driver="GTiff", height=50, width=50, count=3, dtype=np.uint8, crs="EPSG:32632", transform=transform
        ) as dst:
            dst.write(data)

    res = BiTemporalValidator.validate_pair(p1, p2)
    assert res.is_valid is True
    assert res.temporal_order_verified is False
    assert any("Temporal ordering anomaly" in w for w in res.warnings)


def test_disjoint_images_fail_validation(tmp_path: Path):
    """Verify images with zero spatial intersection fail with explicit disjoint error."""
    p1 = tmp_path / "area1.tif"
    p2 = tmp_path / "area2.tif"

    # Box 1: (0, 0, 100, 100)
    tf1 = from_bounds(0, 0, 100, 100, 50, 50)
    # Box 2: (500, 500, 600, 600) -> completely disjoint
    tf2 = from_bounds(500, 500, 600, 600, 50, 50)

    data = np.zeros((3, 50, 50), dtype=np.uint8)
    with rasterio.open(p1, "w", driver="GTiff", height=50, width=50, count=3, dtype=np.uint8, crs="EPSG:32632", transform=tf1) as dst:
        dst.write(data)
    with rasterio.open(p2, "w", driver="GTiff", height=50, width=50, count=3, dtype=np.uint8, crs="EPSG:32632", transform=tf2) as dst:
        dst.write(data)

    res = BiTemporalValidator.validate_pair(p1, p2)
    assert res.is_valid is False
    assert "zero spatial intersection" in res.error


def test_insufficient_overlap_fails_validation(tmp_path: Path):
    """Verify pairs with less than 50% spatial overlap fail validation."""
    p1 = tmp_path / "box_main.tif"
    p2 = tmp_path / "box_sliver.tif"

    # Box 1: [0, 0, 100, 100] (area 10,000)
    tf1 = from_bounds(0, 0, 100, 100, 100, 100)
    # Box 2: [80, 0, 180, 100] -> overlap is [80, 0, 100, 100] = 20x100 = 2000 (20% overlap)
    tf2 = from_bounds(80, 0, 180, 100, 100, 100)

    data = np.zeros((3, 100, 100), dtype=np.uint8)
    with rasterio.open(p1, "w", driver="GTiff", height=100, width=100, count=3, dtype=np.uint8, crs="EPSG:32632", transform=tf1) as dst:
        dst.write(data)
    with rasterio.open(p2, "w", driver="GTiff", height=100, width=100, count=3, dtype=np.uint8, crs="EPSG:32632", transform=tf2) as dst:
        dst.write(data)

    res = BiTemporalValidator.validate_pair(p1, p2, min_overlap_ratio=0.50)
    assert res.is_valid is False
    assert "Insufficient spatial overlap" in res.error


def test_bitemporal_alignment_intersection_crop(tmp_path: Path):
    """Verify alignment crops both rasters to their exact common bounding intersection."""
    p1 = tmp_path / "crop_t1.tif"
    p2 = tmp_path / "crop_t2.tif"

    # Image 1: [0, 0, 100, 100] (100x100 pixels, resolution 1.0)
    tf1 = from_bounds(0, 0, 100, 100, 100, 100)
    # Image 2: [20, 20, 120, 120] (100x100 pixels, resolution 1.0)
    # Intersection: [20, 20, 100, 100] -> width=80, height=80
    tf2 = from_bounds(20, 20, 120, 120, 100, 100)

    data1 = np.full((3, 100, 100), 10, dtype=np.uint8)
    data2 = np.full((3, 100, 100), 20, dtype=np.uint8)

    with rasterio.open(p1, "w", driver="GTiff", height=100, width=100, count=3, dtype=np.uint8, crs="EPSG:32632", transform=tf1) as dst:
        dst.write(data1)
    with rasterio.open(p2, "w", driver="GTiff", height=100, width=100, count=3, dtype=np.uint8, crs="EPSG:32632", transform=tf2) as dst:
        dst.write(data2)

    aligned = BiTemporalAligner.align_pair(p1, p2)
    assert aligned.width == 80
    assert aligned.height == 80
    assert aligned.image1_array.shape == (3, 80, 80)
    assert aligned.image2_array.shape == (3, 80, 80)
    assert aligned.bounds == (20.0, 20.0, 100.0, 100.0)
    assert aligned.crs == "EPSG:32632"
