"""Generates the Dedicated Change Robustness Benchmark Dataset.

Milestone Extended Maximum Training & Format Robustness (Part 20):
Creates datasets/change_robustness/ with 10 structured test scenarios:
01_building/        - Obvious building construction (PNG + PNG)
02_vegetation/      - Deforestation / canopy loss (PNG + PNG)
03_water/           - Reservoir inundation / water body expansion (PNG + PNG)
04_roads/           - Highway / transportation corridor addition (PNG + PNG)
05_construction/    - Earth excavation / bare ground transition (JPEG + JPEG)
06_no_change/       - Identical urban scene (zero physical change) (PNG + PNG)
07_seasonal/        - Seasonal vegetation hue shift without physical change (PNG + PNG)
08_compression/     - Severe JPEG compression artifacts (Q=30 vs Q=95) without physical change (JPEG + JPEG)
09_misalignment/    - Significant translation offset (>45px jitter) testing alignment detection (PNG + PNG)
10_cross_format/    - Real physical change across formats (PNG T1 + TIFF T2)

Each folder contains:
- T1 (primary image: t1.png, t1.jpg, or t1.tif)
- T2 (secondary image: t2.png, t2.jpg, or t2.tif)
- metadata.json (sensor, dimensions, format, acquisition dates)
- expected_behavior.json (change_detected, direction, confidence_tier, alignment_status)
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "datasets" / "change_robustness"


def create_base_texture(h=256, w=256, base_color=(120, 110, 95)):
    """Creates a realistic earth background texture."""
    np.random.seed(42)
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.float32)
    base = np.full((h, w, 3), base_color, dtype=np.float32) + noise
    return np.clip(base, 0, 255).astype(np.uint8)


def generate_robustness_benchmark():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating change robustness benchmark in {BASE_DIR}...")

    # -------------------------------------------------------------
    # 01_building: Obvious building construction
    # -------------------------------------------------------------
    d1 = BASE_DIR / "01_building"
    d1.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (140, 130, 115))  # bare soil
    t2 = t1.copy()
    # Add 4 distinct rectangular buildings with shadows
    for (y, x) in [(40, 40), (40, 140), (140, 40), (140, 140)]:
        t2[y:y+50, x:x+60] = [210, 215, 220]  # bright concrete roof
        t2[y+50:y+60, x:x+60] = [40, 40, 45]   # shadow
    Image.fromarray(t1).save(d1 / "t1.png")
    Image.fromarray(t2).save(d1 / "t2.png")
    (d1 / "metadata.json").write_text(json.dumps({
        "scenario_id": "01_building",
        "category": "building_change",
        "format_t1": "PNG", "format_t2": "PNG",
        "dimensions": [256, 256],
        "date_t1": "2021-02-15", "date_t2": "2023-08-20"
    }, indent=2))
    (d1 / "expected_behavior.json").write_text(json.dumps({
        "change_detected": True,
        "temporal_direction": "addition",
        "predominant_from_class": "bare_ground_or_soil",
        "predominant_to_class": "built_structure",
        "expected_confidence_level": "HIGH",
        "expected_alignment_quality": "HIGH"
    }, indent=2))

    # -------------------------------------------------------------
    # 02_vegetation: Forest canopy clearance / Deforestation
    # -------------------------------------------------------------
    d2 = BASE_DIR / "02_vegetation"
    d2.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (35, 95, 40))  # dense forest green
    t2 = t1.copy()
    # Clear a large circular/irregular swath in center
    yy, xx = np.ogrid[:256, :256]
    mask = ((xx - 128)**2 + (yy - 128)**2) <= 65**2
    t2[mask] = [175, 140, 95]  # cleared bare ground
    Image.fromarray(t1).save(d2 / "t1.png")
    Image.fromarray(t2).save(d2 / "t2.png")
    (d2 / "metadata.json").write_text(json.dumps({
        "scenario_id": "02_vegetation",
        "category": "vegetation_loss",
        "format_t1": "PNG", "format_t2": "PNG",
        "dimensions": [256, 256],
        "date_t1": "2020-05-10", "date_t2": "2022-09-14"
    }, indent=2))
    (d2 / "expected_behavior.json").write_text(json.dumps({
        "change_detected": True,
        "temporal_direction": "removal",
        "predominant_from_class": "forest_or_trees",
        "predominant_to_class": "bare_ground_or_soil",
        "expected_confidence_level": "HIGH",
        "expected_alignment_quality": "HIGH"
    }, indent=2))

    # -------------------------------------------------------------
    # 03_water: Water body inundation / expansion
    # -------------------------------------------------------------
    d3 = BASE_DIR / "03_water"
    d3.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (160, 145, 120))  # floodplain bare ground
    t2 = t1.copy()
    # Water inundation across lower half
    t2[110:, :] = [25, 65, 125]  # deep water blue
    Image.fromarray(t1).save(d3 / "t1.png")
    Image.fromarray(t2).save(d3 / "t2.png")
    (d3 / "metadata.json").write_text(json.dumps({
        "scenario_id": "03_water",
        "category": "water_expansion",
        "format_t1": "PNG", "format_t2": "PNG",
        "dimensions": [256, 256],
        "date_t1": "2021-03-01", "date_t2": "2021-08-10"
    }, indent=2))
    (d3 / "expected_behavior.json").write_text(json.dumps({
        "change_detected": True,
        "temporal_direction": "addition",
        "predominant_from_class": "bare_ground_or_soil",
        "predominant_to_class": "water_body",
        "expected_confidence_level": "HIGH",
        "expected_alignment_quality": "HIGH"
    }, indent=2))

    # -------------------------------------------------------------
    # 04_roads: Highway / Road corridor addition
    # -------------------------------------------------------------
    d4 = BASE_DIR / "04_roads"
    d4.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (110, 130, 90))  # rural grassland
    t2 = t1.copy()
    # Paved asphalt highway diagonal
    for i in range(256):
        t2[max(0, i-6):min(256, i+6), i] = [55, 55, 60]  # asphalt
    Image.fromarray(t1).save(d4 / "t1.png")
    Image.fromarray(t2).save(d4 / "t2.png")
    (d4 / "metadata.json").write_text(json.dumps({
        "scenario_id": "04_roads",
        "category": "infrastructure_road",
        "format_t1": "PNG", "format_t2": "PNG",
        "dimensions": [256, 256],
        "date_t1": "2019-11-05", "date_t2": "2022-04-18"
    }, indent=2))
    (d4 / "expected_behavior.json").write_text(json.dumps({
        "change_detected": True,
        "temporal_direction": "addition",
        "predominant_from_class": "vegetation_or_cropland",
        "predominant_to_class": "built_structure",
        "expected_confidence_level": "HIGH",
        "expected_alignment_quality": "HIGH"
    }, indent=2))

    # -------------------------------------------------------------
    # 05_construction: Excavation / bare ground transition (JPEG + JPEG)
    # -------------------------------------------------------------
    d5 = BASE_DIR / "05_construction"
    d5.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (90, 120, 80))   # vegetated land
    t2 = t1.copy()
    t2[60:200, 50:210] = [185, 150, 100]  # bulldozed construction site
    Image.fromarray(t1).save(d5 / "t1.jpg", quality=90)
    Image.fromarray(t2).save(d5 / "t2.jpg", quality=90)
    (d5 / "metadata.json").write_text(json.dumps({
        "scenario_id": "05_construction",
        "category": "construction_excavation",
        "format_t1": "JPEG", "format_t2": "JPEG",
        "dimensions": [256, 256],
        "date_t1": "2021-01-20", "date_t2": "2022-06-15"
    }, indent=2))
    (d5 / "expected_behavior.json").write_text(json.dumps({
        "change_detected": True,
        "temporal_direction": "addition",
        "predominant_from_class": "vegetation_or_cropland",
        "predominant_to_class": "bare_ground_or_soil",
        "expected_confidence_level": "HIGH",
        "expected_alignment_quality": "HIGH"
    }, indent=2))

    # -------------------------------------------------------------
    # 06_no_change: Identical urban scene (Zero Physical Change)
    # -------------------------------------------------------------
    d6 = BASE_DIR / "06_no_change"
    d6.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (120, 120, 120))
    # Add static buildings
    for (y, x) in [(30, 30), (30, 150), (150, 30), (150, 150)]:
        t1[y:y+45, x:x+55] = [200, 200, 210]
        t1[y+45:y+52, x:x+55] = [40, 40, 45]
    t2 = t1.copy()  # strictly identical scene
    Image.fromarray(t1).save(d6 / "t1.png")
    Image.fromarray(t2).save(d6 / "t2.png")
    (d6 / "metadata.json").write_text(json.dumps({
        "scenario_id": "06_no_change",
        "category": "zero_physical_change",
        "format_t1": "PNG", "format_t2": "PNG",
        "dimensions": [256, 256],
        "date_t1": "2022-03-01", "date_t2": "2022-03-15"
    }, indent=2))
    (d6 / "expected_behavior.json").write_text(json.dumps({
        "change_detected": False,
        "temporal_direction": "no_change",
        "predominant_from_class": "built_structure",
        "predominant_to_class": "built_structure",
        "expected_confidence_level": "HIGH",
        "expected_alignment_quality": "HIGH"
    }, indent=2))

    # -------------------------------------------------------------
    # 07_seasonal: Seasonal vegetation hue variation (Nuisance)
    # -------------------------------------------------------------
    d7 = BASE_DIR / "07_seasonal"
    d7.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (40, 110, 45))  # wet season lush green
    # Dry season: vegetation becomes yellow/brown, but physical canopy remains
    t2 = t1.astype(np.float32).copy()
    t2[:, :, 0] += 35.0  # warmer red/yellow
    t2[:, :, 1] += 10.0
    t2[:, :, 2] -= 15.0
    t2 = np.clip(t2, 0, 255).astype(np.uint8)
    Image.fromarray(t1).save(d7 / "t1.png")
    Image.fromarray(t2).save(d7 / "t2.png")
    (d7 / "metadata.json").write_text(json.dumps({
        "scenario_id": "07_seasonal",
        "category": "seasonal_vegetation_variation",
        "format_t1": "PNG", "format_t2": "PNG",
        "dimensions": [256, 256],
        "date_t1": "2022-07-15", "date_t2": "2022-12-10"
    }, indent=2))
    (d7 / "expected_behavior.json").write_text(json.dumps({
        "change_detected": False,
        "temporal_direction": "no_change",
        "predominant_from_class": "vegetation_or_cropland",
        "predominant_to_class": "vegetation_or_cropland",
        "expected_confidence_level": "MEDIUM",
        "expected_alignment_quality": "HIGH"
    }, indent=2))

    # -------------------------------------------------------------
    # 08_compression: Heavy JPEG compression artifacts (Q=30 vs Q=95)
    # -------------------------------------------------------------
    d8 = BASE_DIR / "08_compression"
    d8.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (130, 125, 110))
    # Add strong high-contrast infrastructure features that survive JPEG compression
    t1[50:110, 50:110] = [210, 215, 220]
    t1[150:200, 140:220] = [45, 45, 50]
    for i in range(256):
        t1[i, max(0, i-4):min(256, i+4)] = [60, 60, 65]
    # Identical scene, but T2 saved with harsh JPEG block compression
    Image.fromarray(t1).save(d8 / "t1.jpg", quality=95)
    Image.fromarray(t1).save(d8 / "t2.jpg", quality=35)
    (d8 / "metadata.json").write_text(json.dumps({
        "scenario_id": "08_compression",
        "category": "compression_artifacts",
        "format_t1": "JPEG", "format_t2": "JPEG",
        "dimensions": [256, 256],
        "compression_t1_quality": 95, "compression_t2_quality": 35,
        "date_t1": "2023-01-10", "date_t2": "2023-01-10"
    }, indent=2))
    (d8 / "expected_behavior.json").write_text(json.dumps({
        "change_detected": False,
        "temporal_direction": "no_change",
        "expected_confidence_level": "MEDIUM",
        "expected_alignment_quality": "HIGH"
    }, indent=2))

    # -------------------------------------------------------------
    # 09_misalignment: Large translation misalignment (>40px shift)
    # -------------------------------------------------------------
    d9 = BASE_DIR / "09_misalignment"
    d9.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (120, 120, 120))
    # Add recognizable pattern
    t1[40:90, 40:90] = [220, 220, 230]
    t1[140:190, 140:190] = [30, 30, 35]
    # Shift T2 by (55, 45) pixels
    t2 = np.roll(np.roll(t1, shift=45, axis=0), shift=55, axis=1)
    Image.fromarray(t1).save(d9 / "t1.png")
    Image.fromarray(t2).save(d9 / "t2.png")
    (d9 / "metadata.json").write_text(json.dumps({
        "scenario_id": "09_misalignment",
        "category": "spatial_misalignment",
        "format_t1": "PNG", "format_t2": "PNG",
        "dimensions": [256, 256],
        "true_pixel_shift": [55, 45]
    }, indent=2))
    (d9 / "expected_behavior.json").write_text(json.dumps({
        "expected_confidence_level": "LOW",
        "expected_alignment_quality": "LOW",
        "alignment_warning_expected": True
    }, indent=2))

    # -------------------------------------------------------------
    # 10_cross_format: Real physical change across formats (PNG + TIFF)
    # -------------------------------------------------------------
    d10 = BASE_DIR / "10_cross_format"
    d10.mkdir(exist_ok=True)
    t1 = create_base_texture(256, 256, (140, 130, 110))
    t2 = t1.copy()
    t2[70:180, 70:180] = [215, 220, 225]  # New industrial complex
    Image.fromarray(t1).save(d10 / "t1.png")
    # Save t2 as uncompressed TIFF
    Image.fromarray(t2).save(d10 / "t2.tif", format="TIFF")
    (d10 / "metadata.json").write_text(json.dumps({
        "scenario_id": "10_cross_format",
        "category": "cross_format_change",
        "format_t1": "PNG", "format_t2": "TIFF",
        "dimensions": [256, 256],
        "date_t1": "2020-04-12", "date_t2": "2023-05-18"
    }, indent=2))
    (d10 / "expected_behavior.json").write_text(json.dumps({
        "change_detected": True,
        "temporal_direction": "addition",
        "expected_confidence_level": "HIGH",
        "expected_alignment_quality": "HIGH"
    }, indent=2))

    print("Successfully generated all 10 change robustness benchmark scenarios!")


if __name__ == "__main__":
    generate_robustness_benchmark()
