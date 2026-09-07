"""Unit tests for metadata extraction."""

from pathlib import Path
from backend.preprocessing.metadata import MetadataExtractor


def test_metadata_extraction_optical(optical_sample_path: Path):
    """Verify deterministic metadata extraction on optical sample."""
    meta = MetadataExtractor.extract_image_metadata(optical_sample_path)
    assert meta.filename == "optical_s2_sample.tif"
    assert meta.width == 256
    assert meta.height == 256
    assert meta.band_count == 3
    assert meta.crs == "EPSG:32643"
    assert meta.resolution == (10.0, 10.0)
    assert meta.bounds is not None
    assert meta.dtype == "uint16"


def test_metadata_extraction_real_rs(fixtures_dir: Path):
    """Verify metadata extraction on Rasterio repository GeoTIFF test image used for smoke testing."""
    real_sample = fixtures_dir / "real_rs_sample.tif"
    if not real_sample.exists():
        return

    meta = MetadataExtractor.extract_image_metadata(real_sample)
    assert meta.filename == "real_rs_sample.tif"
    assert meta.width == 791
    assert meta.height == 718
    assert meta.band_count == 3
    assert meta.crs == "EPSG:32618"


def test_spatial_summary_formatting(optical_sample_path: Path):
    """Verify spatial summary dictionary formatting."""
    meta = MetadataExtractor.extract_image_metadata(optical_sample_path)
    summary = MetadataExtractor.format_spatial_summary(meta)
    assert summary["dimensions"] == "256 x 256"
    assert summary["band_count"] == 3
    assert summary["crs"] == "EPSG:32643"
    assert summary["ground_sample_distance_m"] == 10.0
