"""Pytest fixtures and test environment configuration."""

import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Returns the path to test fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def optical_sample_path(fixtures_dir: Path) -> Path:
    """Returns path to sample optical GeoTIFF."""
    return fixtures_dir / "optical_s2_sample.tif"


@pytest.fixture
def sar_sample_path(fixtures_dir: Path) -> Path:
    """Returns path to sample SAR GeoTIFF."""
    return fixtures_dir / "sar_s1_sample.tif"


@pytest.fixture
def bitemporal_paths(fixtures_dir: Path) -> tuple[Path, Path, Path]:
    """Returns paths to (time1, time2, ground_truth_mask)."""
    t1 = fixtures_dir / "bitemporal" / "time1_pre_change.tif"
    t2 = fixtures_dir / "bitemporal" / "time2_post_change.tif"
    gt = fixtures_dir / "bitemporal" / "change_ground_truth.png"
    return t1, t2, gt
