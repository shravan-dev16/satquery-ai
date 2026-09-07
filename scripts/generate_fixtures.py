"""Generates lightweight synthetic GeoTIFF and benchmark fixtures for unit testing.

Keeps total size < 5 MB, ensuring offline testing in CI/CD environments.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def create_fixtures():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "bitemporal").mkdir(parents=True, exist_ok=True)
    (FIXTURES_DIR / "invalid").mkdir(parents=True, exist_ok=True)

    # Common geospatial parameters
    crs = "EPSG:32643"  # UTM Zone 43N
    res = 10.0  # 10m Ground Sample Distance
    width, height = 256, 256
    origin_x, origin_y = 725000.0, 3130000.0
    transform = from_origin(origin_x, origin_y, res, res)

    # -------------------------------------------------------------
    # 1. Optical Sentinel-2 Style RGB GeoTIFF
    # -------------------------------------------------------------
    optical_path = FIXTURES_DIR / "optical_s2_sample.tif"
    # Band 1: Red, Band 2: Green, Band 3: Blue (uint16 reflectance e.g. 0-10000)
    b_red = np.full((height, width), 1200, dtype=np.uint16)
    b_green = np.full((height, width), 1800, dtype=np.uint16)
    b_blue = np.full((height, width), 900, dtype=np.uint16)

    # Add a water body in the top-left quadrant (low reflectance)
    b_red[:80, :80] = 300
    b_green[:80, :80] = 500
    b_blue[:80, :80] = 800

    # Add an urban cluster in the bottom-right (high reflectance)
    b_red[160:, 160:] = 3500
    b_green[160:, 160:] = 3200
    b_blue[160:, 160:] = 3000

    with rasterio.open(
        optical_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.uint16,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(b_red, 1)
        dst.write(b_green, 2)
        dst.write(b_blue, 3)
    print(f"Created {optical_path} ({optical_path.stat().st_size} bytes)")

    # -------------------------------------------------------------
    # 2. SAR Sentinel-1 Style Dual-Pol (VV, VH) GeoTIFF
    # -------------------------------------------------------------
    sar_path = FIXTURES_DIR / "sar_s1_sample.tif"
    # Band 1: VV (float32 backscatter in linear or dB)
    # Band 2: VH (float32 cross-pol)
    vv = np.random.normal(-12.0, 2.0, (height, width)).astype(np.float32)
    vh = np.random.normal(-19.0, 2.0, (height, width)).astype(np.float32)

    # Water has very low backscatter (specular reflection away from sensor: -25 dB)
    vv[:80, :80] = -26.0
    vh[:80, :80] = -32.0

    # Urban has high backscatter (double-bounce corner reflectors: -3 dB)
    vv[160:, 160:] = -2.5
    vh[160:, 160:] = -8.0

    with rasterio.open(
        sar_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=2,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(vv, 1)
        dst.write(vh, 2)
    print(f"Created {sar_path} ({sar_path.stat().st_size} bytes)")

    # -------------------------------------------------------------
    # 3. Bi-temporal Pair (Time 1 & Time 2) & Change Ground Truth
    # -------------------------------------------------------------
    t1_path = FIXTURES_DIR / "bitemporal" / "time1_pre_change.tif"
    t2_path = FIXTURES_DIR / "bitemporal" / "time2_post_change.tif"
    gt_mask_path = FIXTURES_DIR / "bitemporal" / "change_ground_truth.png"

    # Time 1: Agricultural/vegetation field
    t1_img = np.zeros((3, height, width), dtype=np.uint8)
    t1_img[0] = 70   # R
    t1_img[1] = 140  # G
    t1_img[2] = 60   # B

    # Time 2: Same scene, but with a new industrial warehouse in center (80x80 pixels)
    t2_img = t1_img.copy()
    change_mask = np.zeros((height, width), dtype=np.uint8)

    # New building coordinates: [90:170, 90:170]
    t2_img[0, 90:170, 90:170] = 210  # Concrete bright
    t2_img[1, 90:170, 90:170] = 215
    t2_img[2, 90:170, 90:170] = 220
    change_mask[90:170, 90:170] = 255  # Ground truth changed pixels

    for path, data in [(t1_path, t1_img), (t2_path, t2_img)]:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data[0], 1)
            dst.write(data[1], 2)
            dst.write(data[2], 3)
        print(f"Created {path} ({path.stat().st_size} bytes)")

    # Save PNG mask
    Image.fromarray(change_mask).save(gt_mask_path)
    print(f"Created {gt_mask_path} ({gt_mask_path.stat().st_size} bytes)")

    # -------------------------------------------------------------
    # 4. Invalid Fixtures for Negative Testing
    # -------------------------------------------------------------
    # Corrupted header
    corrupt_path = FIXTURES_DIR / "invalid" / "corrupted_header.tif"
    corrupt_path.write_bytes(b"NOT_A_VALID_TIFF_HEADER_1234567890")
    print(f"Created {corrupt_path}")

    # Unprojected plain PNG
    unproj_path = FIXTURES_DIR / "invalid" / "unprojected_image.png"
    Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)).save(unproj_path)
    print(f"Created {unproj_path}")

    # Mismatched CRS (EPSG:4326 vs UTM)
    mismatch_path = FIXTURES_DIR / "invalid" / "mismatched_crs.tif"
    transform_deg = from_origin(77.5, 12.9, 0.0001, 0.0001)
    with rasterio.open(
        mismatch_path,
        "w",
        driver="GTiff",
        height=128,
        width=128,
        count=1,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform_deg,
    ) as dst:
        dst.write(np.zeros((128, 128), dtype=np.uint8), 1)
    print(f"Created {mismatch_path}")

    # -------------------------------------------------------------
    # 5. Mock Query Corpus
    # -------------------------------------------------------------
    queries_path = FIXTURES_DIR / "mock_queries.json"
    queries = [
        {"id": "q1", "task": "vqa", "query": "Is there a water body in the top-left section?", "expected": "Yes"},
        {"id": "q2", "task": "grounding", "query": "Ground the urban structures in the bottom-right", "expected_boxes": 1},
        {"id": "q3", "task": "change_detection", "query": "What changed between time 1 and time 2?", "expected_change": True},
        {"id": "q4", "task": "optical_sar_analysis", "query": "Identify cloud-penetrated structures using optical and SAR", "expected_fusion": True}
    ]
    queries_path.write_text(json.dumps(queries, indent=2))
    print(f"Created {queries_path}")

    # Compute total size of fixtures
    total_bytes = sum(f.stat().st_size for f in FIXTURES_DIR.rglob("*") if f.is_file())
    print(f"\nTOTAL FIXTURE SIZE: {total_bytes / (1024 * 1024):.2f} MB (Must be < 5.0 MB)")


if __name__ == "__main__":
    create_fixtures()
