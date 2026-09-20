"""Dedicated Test Suite: Format-Robust Bi-Temporal Change Detection & Alignment.

Milestone Extended Maximum Training & Format Robustness (Part 19):
Verifies the 16 mandatory scenarios across PNG, JPEG, TIFF, and GeoTIFF:
1.  PNG + PNG obvious building change
2.  PNG + PNG no change
3.  PNG + PNG vegetation change
4.  PNG + PNG water change
5.  PNG + PNG road change
6.  JPEG + JPEG obvious change
7.  JPEG + JPEG no change
8.  TIFF + TIFF
9.  GeoTIFF + GeoTIFF
10. PNG + TIFF compatibility handling
11. Different resolutions handling (aspect/dimension discrepancy)
12. Misaligned PNG (phase correlation detects shift, avoids false warp)
13. Unrelated PNG (rejects or triggers LOW confidence)
14. Heavily compressed JPEG (compression artifacts != physical change)
15. Brightness variation (illumination invariance)
16. Seasonal variation (phenology hue change != physical change)
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import rasterio
from rasterio.transform import from_bounds

from backend.agent.schema import TaskType
from backend.evidence.confidence import ConfidenceEngine
from backend.models.change import ChangeDetectionSpecialist
from backend.preprocessing.alignment import BiTemporalAligner, BiTemporalValidator
from backend.preprocessing.bi_temporal_normalizer import BiTemporalNormalizer, ImageAlignmentEngine

BENCHMARK_DIR = Path("datasets/change_robustness")


# 1. PNG + PNG obvious building change
def test_01_png_png_obvious_building_change():
    t1 = BENCHMARK_DIR / "01_building" / "t1.png"
    t2 = BENCHMARK_DIR / "01_building" / "t2.png"
    norm = BiTemporalNormalizer.normalize_pair(t1, t2)
    assert norm.is_georeferenced is False
    assert norm.alignment_score >= 0.75

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="What changed between these two dates?")
    assert out.success is True
    assert out.parameters_used["changed_pixels"] > 500
    assert "Bi-temporal change detected" in out.answer_text


# 2. PNG + PNG no change
def test_02_png_png_no_change():
    t1 = BENCHMARK_DIR / "06_no_change" / "t1.png"
    t2 = BENCHMARK_DIR / "06_no_change" / "t2.png"
    norm = BiTemporalNormalizer.normalize_pair(t1, t2)
    assert norm.alignment_score >= 0.90

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="What changed between these two dates?")
    assert out.success is True
    assert out.parameters_used["changed_pixels"] == 0
    assert "No significant physical surface change detected" in out.answer_text


# 3. PNG + PNG vegetation change
def test_03_png_png_vegetation_change():
    t1 = BENCHMARK_DIR / "02_vegetation" / "t1.png"
    t2 = BENCHMARK_DIR / "02_vegetation" / "t2.png"
    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="Detect vegetation clearance.")
    assert out.success is True
    assert out.parameters_used["changed_pixels"] > 1000
    assert out.parameters_used["change_ratio_pct"] >= 10.0


# 4. PNG + PNG water change
def test_04_png_png_water_change():
    t1 = BENCHMARK_DIR / "03_water" / "t1.png"
    t2 = BENCHMARK_DIR / "03_water" / "t2.png"
    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="Has water surface area expanded?")
    assert out.success is True
    assert out.parameters_used["changed_pixels"] > 2000
    assert out.parameters_used["change_ratio_pct"] >= 5.0


# 5. PNG + PNG road change
def test_05_png_png_road_change():
    t1 = BENCHMARK_DIR / "04_roads" / "t1.png"
    t2 = BENCHMARK_DIR / "04_roads" / "t2.png"
    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="Find road construction.")
    assert out.success is True
    assert out.parameters_used["changed_pixels"] > 200
    assert out.parameters_used["total_clusters"] >= 1


# 6. JPEG + JPEG obvious change
def test_06_jpeg_jpeg_obvious_change():
    t1 = BENCHMARK_DIR / "05_construction" / "t1.jpg"
    t2 = BENCHMARK_DIR / "05_construction" / "t2.jpg"
    norm = BiTemporalNormalizer.normalize_pair(t1, t2)
    assert norm.format_t1 == "JPEG" and norm.format_t2 == "JPEG"
    assert norm.alignment_score >= 0.65

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="What changed?")
    assert out.success is True
    assert out.parameters_used["changed_pixels"] > 1000


# 7. JPEG + JPEG no change
def test_07_jpeg_jpeg_no_change(tmp_path: Path):
    p1 = tmp_path / "scene_j1.jpg"
    p2 = tmp_path / "scene_j2.jpg"
    img = np.random.RandomState(42).randint(50, 200, (128, 128, 3), dtype=np.uint8)
    Image.fromarray(img).save(p1, quality=95)
    Image.fromarray(img).save(p2, quality=95)

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(p1, p2, query="What changed?")
    assert out.success is True
    assert out.parameters_used["changed_pixels"] == 0
    assert "No significant physical surface change detected" in out.answer_text


# 8. TIFF + TIFF (unprojected)
def test_08_tiff_tiff(tmp_path: Path):
    p1 = tmp_path / "t1.tif"
    p2 = tmp_path / "t2.tif"
    data1 = np.full((128, 128, 3), 100, dtype=np.uint8)
    data2 = data1.copy()
    data2[30:70, 30:70] = 220
    Image.fromarray(data1).save(p1, format="TIFF")
    Image.fromarray(data2).save(p2, format="TIFF")

    norm = BiTemporalNormalizer.normalize_pair(p1, p2)
    assert norm.is_georeferenced is False
    assert norm.format_t1.upper() in ("TIFF", "GTIFF")

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(p1, p2, query="What changed?")
    assert out.success is True
    assert out.parameters_used["changed_pixels"] >= 1600


# 9. GeoTIFF + GeoTIFF
def test_09_geotiff_geotiff():
    t1 = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
    t2 = Path("tests/fixtures/bitemporal/time2_post_change.tif")
    norm = BiTemporalNormalizer.normalize_pair(t1, t2)
    assert norm.is_georeferenced is True
    assert norm.crs == "EPSG:32643"

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="What changed?")
    assert out.success is True
    # Mode A must provide metric area
    area_ha = out.parameters_used.get("diagnostics", {}).get("threshold_used")
    stats = {s.metric_name: s.value for s in out.evidence.statistics}
    assert "changed_area_hectares" in stats
    assert stats["changed_area_hectares"] > 0.0


# 10. PNG + TIFF compatibility handling
def test_10_png_tiff_cross_format():
    t1 = BENCHMARK_DIR / "10_cross_format" / "t1.png"
    t2 = BENCHMARK_DIR / "10_cross_format" / "t2.tif"
    norm = BiTemporalNormalizer.normalize_pair(t1, t2)
    assert norm.format_t1 == "PNG"
    assert norm.format_t2.upper() in ("TIFF", "GTIFF")

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="What changed?")
    assert out.success is True
    assert out.parameters_used["changed_pixels"] > 500


# 11. Different resolutions handling
def test_11_different_resolutions(tmp_path: Path):
    p1 = tmp_path / "img_256.png"
    p2 = tmp_path / "img_512.png"
    # Same scene content, different raster sampling
    arr = np.random.RandomState(42).randint(60, 180, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(arr).save(p1)
    arr_big = np.array(Image.fromarray(arr).resize((512, 512), Image.Resampling.BILINEAR))
    Image.fromarray(arr_big).save(p2)

    norm = BiTemporalNormalizer.normalize_pair(p1, p2)
    assert norm.width == 256 and norm.height == 256
    assert norm.alignment_score >= 0.85
    assert any("dimensions differ" in w.lower() for w in norm.warnings)


# 12. Misaligned PNG
def test_12_misaligned_png():
    t1 = BENCHMARK_DIR / "09_misalignment" / "t1.png"
    t2 = BENCHMARK_DIR / "09_misalignment" / "t2.png"
    norm = BiTemporalNormalizer.normalize_pair(t1, t2)
    # Alignment engine detects large offset
    assert abs(norm.offset_xy[0]) > 20 or abs(norm.offset_xy[1]) > 20
    # Must emit explicit alignment warning
    assert any("shift" in w.lower() or "misalignment" in w.lower() or "offset" in w.lower() or "discrepancy" in w.lower() for w in norm.warnings)


# 13. Unrelated PNG
def test_13_unrelated_png(tmp_path: Path):
    p1 = tmp_path / "airport.png"
    p2 = tmp_path / "ocean.png"
    # Two completely distinct, unrelated patterns
    airport = np.zeros((128, 128, 3), dtype=np.uint8)
    airport[60:70, :] = 255  # runway
    ocean = np.full((128, 128, 3), (20, 60, 140), dtype=np.uint8)
    Image.fromarray(airport).save(p1)
    Image.fromarray(ocean).save(p2)

    # Validation should reject or flag as very low alignment score
    res = BiTemporalValidator.validate_pair(p1, p2)
    if res.is_valid:
        assert res.spatial_overlap.overlap_ratio_image1 < 0.40
    else:
        assert "zero visual correspondence" in res.error.lower() or "unrelated" in res.error.lower()


# 14. Heavily compressed JPEG
def test_14_heavily_compressed_jpeg():
    t1 = BENCHMARK_DIR / "08_compression" / "t1.jpg"
    t2 = BENCHMARK_DIR / "08_compression" / "t2.jpg"
    norm = BiTemporalNormalizer.normalize_pair(t1, t2)
    assert norm.alignment_score >= 0.60

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="What changed?")
    assert out.success is True
    # The change ratio must not be massive despite severe Q=35 compression
    assert out.parameters_used["change_ratio_pct"] <= 10.0


# 15. Brightness variation
def test_15_brightness_variation(tmp_path: Path):
    p1 = tmp_path / "bright_t1.png"
    p2 = tmp_path / "dim_t2.png"
    arr = np.full((128, 128, 3), 120, dtype=np.uint8)
    arr[30:70, 30:70] = 180
    Image.fromarray(arr).save(p1)
    # 20% dimmer due to sun angle
    arr_dim = (arr.astype(np.float32) * 0.80).astype(np.uint8)
    Image.fromarray(arr_dim).save(p2)

    norm = BiTemporalNormalizer.normalize_pair(p1, p2)
    assert norm.alignment_score >= 0.85

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(p1, p2, query="What changed?")
    assert out.success is True
    # Does not trigger massive false change
    assert out.parameters_used["change_ratio_pct"] <= 5.0


# 16. Seasonal variation
def test_16_seasonal_variation():
    t1 = BENCHMARK_DIR / "07_seasonal" / "t1.png"
    t2 = BENCHMARK_DIR / "07_seasonal" / "t2.png"
    norm = BiTemporalNormalizer.normalize_pair(t1, t2)
    assert norm.alignment_score >= 0.70

    spec = ChangeDetectionSpecialist(candidate="cva")
    out = spec.execute_change_detection(t1, t2, query="What changed?")
    assert out.success is True
    # Seasonal phenology does not produce full-image false positive
    assert out.parameters_used["change_ratio_pct"] <= 10.0
