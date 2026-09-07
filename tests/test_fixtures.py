"""Unit tests verifying synthetic GeoTIFF fixtures and geospatial metadata parsing.

Ensures fixtures adhere to size limits, valid CRS, valid transforms,
and correct band configurations.
"""

from pathlib import Path
import pytest
import numpy as np
import rasterio


def test_fixture_directory_size(fixtures_dir: Path):
    """Verify total fixture footprint is strictly under 5 MB."""
    total_bytes = sum(f.stat().st_size for f in fixtures_dir.rglob("*") if f.is_file())
    total_mb = total_bytes / (1024 * 1024)
    assert total_mb < 5.0, f"Fixture size {total_mb:.2f} MB exceeds 5.0 MB limit"


def test_optical_geotiff_fixture(optical_sample_path: Path):
    """Verify optical GeoTIFF has 3 bands, valid CRS, and non-zero dimensions."""
    assert optical_sample_path.exists()
    with rasterio.open(optical_sample_path) as src:
        assert src.count == 3
        assert src.width == 256
        assert src.height == 256
        assert src.crs is not None
        assert src.crs.to_string() == "EPSG:32643"
        assert src.res == (10.0, 10.0)

        # Read bands
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)
        assert red.shape == (256, 256)
        assert red.dtype == np.uint16


def test_sar_geotiff_fixture(sar_sample_path: Path):
    """Verify SAR GeoTIFF has 2 bands (VV, VH) and float32 dtype."""
    assert sar_sample_path.exists()
    with rasterio.open(sar_sample_path) as src:
        assert src.count == 2
        assert src.width == 256
        assert src.height == 256
        assert src.crs.to_string() == "EPSG:32643"
        assert src.dtypes[0] == "float32"
        assert src.dtypes[1] == "float32"

        vv = src.read(1)
        vh = src.read(2)
        # Check backscatter ranges (water is < -20 dB, urban is > -5 dB)
        assert np.min(vv) < -20.0
        assert np.max(vv) > -5.0


def test_bitemporal_fixtures(bitemporal_paths: tuple[Path, Path, Path]):
    """Verify bi-temporal pairs have identical geometry and matching change ground truth."""
    t1_path, t2_path, gt_path = bitemporal_paths
    assert t1_path.exists()
    assert t2_path.exists()
    assert gt_path.exists()

    with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
        assert src1.crs == src2.crs
        assert src1.bounds == src2.bounds
        assert src1.res == src2.res
        assert src1.shape == src2.shape

        t1 = src1.read()
        t2 = src2.read()
        # Verify there is a detected difference
        diff = np.abs(t2.astype(int) - t1.astype(int))
        assert np.sum(diff) > 0


def test_invalid_fixtures(fixtures_dir: Path):
    """Verify invalid fixtures are properly detected as corrupted or unprojected."""
    corrupt = fixtures_dir / "invalid" / "corrupted_header.tif"
    assert corrupt.exists()
    with pytest.raises(Exception):  # rasterio.errors.RasterioIOError
        rasterio.open(corrupt)

    unprojected = fixtures_dir / "invalid" / "unprojected_image.png"
    assert unprojected.exists()
    # PNG should open but will not have an EPSG CRS
    with rasterio.open(unprojected) as src:
        assert src.crs is None
