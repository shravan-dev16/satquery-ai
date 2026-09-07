"""Unit tests for raster validation logic.

Verifies valid GeoTIFF, corrupted files, missing CRS, and structured error responses.
"""

from pathlib import Path
from backend.preprocessing.validation import RasterValidator


def test_validate_valid_optical_geotiff(optical_sample_path: Path):
    """Verify valid GeoTIFF passes validation with extracted metadata."""
    res = RasterValidator.validate_single_raster(optical_sample_path)
    assert res.valid is True
    assert len(res.errors) == 0
    assert len(res.images) == 1
    img = res.images[0]
    assert img.width == 256
    assert img.height == 256
    assert img.band_count == 3
    assert img.crs == "EPSG:32643"
    assert img.resolution == (10.0, 10.0)


def test_validate_corrupted_geotiff(fixtures_dir: Path):
    """Verify corrupted file produces structured validation error."""
    corrupt_path = fixtures_dir / "invalid" / "corrupted_header.tif"
    res = RasterValidator.validate_single_raster(corrupt_path)
    assert res.valid is False
    assert len(res.errors) >= 1
    assert any("corrupted" in err.lower() or "invalid" in err.lower() for err in res.errors)


def test_validate_unprojected_image(fixtures_dir: Path):
    """Verify unprojected image is valid but raises a missing CRS warning."""
    unprojected_path = fixtures_dir / "invalid" / "unprojected_image.png"
    res = RasterValidator.validate_single_raster(unprojected_path)
    assert res.valid is True  # Format is readable
    assert any("crs" in w.lower() for w in res.warnings)


def test_validate_nonexistent_file(tmp_path: Path):
    """Verify non-existent file produces unreadable error."""
    missing = tmp_path / "does_not_exist.tif"
    res = RasterValidator.validate_single_raster(missing)
    assert res.valid is False
    assert any("not found" in err.lower() or "unreadable" in err.lower() for err in res.errors)
