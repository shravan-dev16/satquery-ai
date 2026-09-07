"""Deterministic semantic validation fixture generator and registry.

Generates small, high-contrast, mathematically verifiable 256x256 GeoTIFF pairs (EPSG:32643)
with authoritative expected transitions for:
1. urban: bare_ground_or_soil -> built_structure
2. forest: forest_or_trees -> bare_ground_or_soil (canopy clearance)
3. water: bare_ground_or_soil -> water_body (flood / reservoir inundation)
4. agriculture: vegetation_or_cropland -> bare_ground_or_soil (crop harvest / plowing)
5. zero_change: identical observations (anti-hallucination control)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin


@dataclass
class SemanticFixtureCase:
    scenario_id: str
    category: str
    description: str
    t1_path: Path
    t2_path: Path
    expected_from_class: str
    expected_to_class: str
    expected_direction: str
    query: str
    candidate_detector: str = "cva"  # CVA provides robust multi-category detection


def generate_semantic_fixtures(fixture_dir: Optional[Path] = None) -> List[SemanticFixtureCase]:
    """Generates deterministic projected GeoTIFFs for semantic validation."""
    if fixture_dir is None:
        fixture_dir = Path("tests/fixtures/semantic")
    fixture_dir.mkdir(parents=True, exist_ok=True)

    w, h = 256, 256
    transform = from_origin(500000.0, 3000000.0, 10.0, 10.0)
    crs = "EPSG:32643"

    def write_geotiff(path: Path, rgb_arr: np.ndarray) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(rgb_arr.transpose(2, 0, 1))

    cases: List[SemanticFixtureCase] = []

    # -------------------------------------------------------------
    # 1. URBAN: Bare Soil -> Built Structures
    # -------------------------------------------------------------
    np.random.seed(101)
    u_t1 = np.full((h, w, 3), [190, 175, 145], dtype=np.uint8)
    u_t1 += np.random.normal(0, 5, (h, w, 3)).clip(-15, 15).astype(np.uint8)

    u_t2 = u_t1.copy()
    u_t2[60:130, 60:130] = [185, 55, 45]       # Building 1 (red roof)
    u_t2[60:130, 145:215] = [95, 115, 135]    # Building 2 (blue-grey roof)
    u_t2[145:205, 80:190] = [215, 215, 220]    # Building 3 (white-grey warehouse)

    p1 = fixture_dir / "t1_urban.tif"
    p2 = fixture_dir / "t2_urban.tif"
    write_geotiff(p1, u_t1)
    write_geotiff(p2, u_t2)
    cases.append(
        SemanticFixtureCase(
            scenario_id="semantic_urban_01",
            category="URBAN",
            description="Open bare ground developed into rectangular building structures",
            t1_path=p1,
            t2_path=p2,
            expected_from_class="bare_ground_or_soil",
            expected_to_class="built_structure",
            expected_direction="increased",
            query="What changed between these two images? Describe the type of construction or development.",
            candidate_detector="cva",
        )
    )

    # -------------------------------------------------------------
    # 2. FOREST: Dense Forest -> Cleared Soil (Deforestation)
    # -------------------------------------------------------------
    np.random.seed(102)
    f_t1 = np.zeros((h, w, 3), dtype=np.uint8)
    f_t1[:, :, 0] = np.random.normal(30, 8, (h, w)).clip(15, 50).astype(np.uint8)
    f_t1[:, :, 1] = np.random.normal(105, 12, (h, w)).clip(75, 145).astype(np.uint8)
    f_t1[:, :, 2] = np.random.normal(30, 8, (h, w)).clip(15, 50).astype(np.uint8)

    f_t2 = f_t1.copy()
    f_t2[64:192, 64:192] = [165, 135, 95]
    f_t2[64:192, 64:192] += np.random.normal(0, 8, (128, 128, 3)).clip(-20, 20).astype(np.uint8)

    p1 = fixture_dir / "t1_forest.tif"
    p2 = fixture_dir / "t2_forest.tif"
    write_geotiff(p1, f_t1)
    write_geotiff(p2, f_t2)
    cases.append(
        SemanticFixtureCase(
            scenario_id="semantic_forest_01",
            category="FOREST_VEGETATION",
            description="Dense green forest canopy cleared to exposed bare soil",
            t1_path=p1,
            t2_path=p2,
            expected_from_class="forest_or_trees",
            expected_to_class="bare_ground_or_soil",
            expected_direction="decreased",
            query="Describe the land-cover change. What vegetation change occurred?",
            candidate_detector="cva",
        )
    )

    # -------------------------------------------------------------
    # 3. WATER: Dry Ground -> Water Body Inundation
    # -------------------------------------------------------------
    np.random.seed(103)
    w_t1 = np.full((h, w, 3), [195, 180, 150], dtype=np.uint8)
    w_t1 += np.random.normal(0, 6, (h, w, 3)).clip(-15, 15).astype(np.uint8)

    w_t2 = w_t1.copy()
    w_t2[64:192, 64:192] = [25, 65, 135]
    w_t2[64:192, 64:192] += np.random.normal(0, 4, (128, 128, 3)).clip(-10, 10).astype(np.uint8)

    p1 = fixture_dir / "t1_water.tif"
    p2 = fixture_dir / "t2_water.tif"
    write_geotiff(p1, w_t1)
    write_geotiff(p2, w_t2)
    cases.append(
        SemanticFixtureCase(
            scenario_id="semantic_water_01",
            category="WATER_AQUATIC",
            description="Dry terrain inundated by surface water / reservoir expansion",
            t1_path=p1,
            t2_path=p2,
            expected_from_class="bare_ground_or_soil",
            expected_to_class="water_body",
            expected_direction="increased",
            query="What water or hydrological surface change occurred between these dates?",
            candidate_detector="cva",
        )
    )

    # -------------------------------------------------------------
    # 4. AGRICULTURE: Green Cropland -> Plowed Soil
    # -------------------------------------------------------------
    np.random.seed(104)
    a_t1 = np.full((h, w, 3), [65, 155, 55], dtype=np.uint8)
    a_t1 += np.random.normal(0, 6, (h, w, 3)).clip(-15, 15).astype(np.uint8)

    a_t2 = a_t1.copy()
    a_t2[50:200, 50:200] = [150, 115, 75]
    a_t2[50:200, 50:200] += np.random.normal(0, 7, (150, 150, 3)).clip(-18, 18).astype(np.uint8)

    p1 = fixture_dir / "t1_agri.tif"
    p2 = fixture_dir / "t2_agri.tif"
    write_geotiff(p1, a_t1)
    write_geotiff(p2, a_t2)
    cases.append(
        SemanticFixtureCase(
            scenario_id="semantic_agri_01",
            category="AGRICULTURE",
            description="Vigorous crop canopy harvested and plowed to exposed bare ground",
            t1_path=p1,
            t2_path=p2,
            expected_from_class="vegetation_or_cropland",
            expected_to_class="bare_ground_or_soil",
            expected_direction="modified",
            query="What agricultural change occurred? Describe the field and crop transition.",
            candidate_detector="cva",
        )
    )

    # -------------------------------------------------------------
    # 5. ZERO-CHANGE CONTROL: Identical Imagery
    # -------------------------------------------------------------
    p1_z = fixture_dir / "t1_zero.tif"
    p2_z = fixture_dir / "t2_zero.tif"
    write_geotiff(p1_z, u_t1)
    write_geotiff(p2_z, u_t1.copy())
    cases.append(
        SemanticFixtureCase(
            scenario_id="semantic_zero_01",
            category="ZERO_CHANGE",
            description="Identical pre-change imagery (true negative test for anti-hallucination)",
            t1_path=p1_z,
            t2_path=p2_z,
            expected_from_class="none",
            expected_to_class="none",
            expected_direction="no_change",
            query="Did anything change between these two observations?",
            candidate_detector="cva",
        )
    )

    return cases
