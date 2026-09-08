"""Deterministic Optical + SAR validation fixture generator and registry (Milestone M6).

Generates small, mathematically verifiable 256x256 GeoTIFF pairs (EPSG:32643)
with authoritative expected classifications for:
1. water_body: Optical dark blue + SAR specular non-reflection (near-zero backscatter)
2. built_structure: Optical geometric roofs + SAR double-bounce corner reflection (high backscatter & roughness)
3. vegetation_or_cropland: Optical high greenness + SAR moderate volume scattering
4. bare_ground_or_soil: Optical warm tan/brown + SAR low-to-moderate surface backscatter
5. Cross-modal sensitivity pairs:
   - Identical optical, varying SAR backscatter (verifies SAR dependence)
   - Identical SAR, varying optical spectral signature (verifies optical dependence)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import rasterio
from rasterio.transform import from_origin


@dataclass
class OpticalSARFixtureCase:
    scenario_id: str
    description: str
    optical_path: Path
    sar_path: Path
    expected_classes: List[str]
    query: str


def generate_optical_sar_fixtures(fixture_dir: Optional[Path] = None) -> Dict[str, OpticalSARFixtureCase]:
    """Generates deterministic projected Optical and SAR GeoTIFFs."""
    if fixture_dir is None:
        fixture_dir = Path("tests/fixtures/optical_sar")
    fixture_dir.mkdir(parents=True, exist_ok=True)

    w, h = 256, 256
    transform = from_origin(500000.0, 3000000.0, 10.0, 10.0)
    crs = "EPSG:32643"

    def write_optical(path: Path, rgb_arr: np.ndarray, custom_transform=transform, custom_crs=crs) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=3,
            dtype=np.uint8,
            crs=custom_crs,
            transform=custom_transform,
            nodata=0,
            compress="deflate",
        ) as dst:
            dst.write(rgb_arr.transpose(2, 0, 1))

    def write_sar(path: Path, sar_arr: np.ndarray, custom_transform=transform, custom_crs=crs) -> None:
        # Write 2-band float32 SAR (VV, VH)
        if sar_arr.ndim == 2:
            data = np.stack([sar_arr, sar_arr * 0.5], axis=0).astype(np.float32)
        else:
            data = sar_arr.astype(np.float32)

        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=data.shape[0],
            dtype=np.float32,
            crs=custom_crs,
            transform=custom_transform,
            nodata=-9999.0,
            compress="deflate",
        ) as dst:
            dst.write(data)

    cases: Dict[str, OpticalSARFixtureCase] = {}

    # -----------------------------------------------------------------------
    # Case 1: Standard Multi-Class Optical + SAR Pair
    # -----------------------------------------------------------------------
    np.random.seed(42)
    # Optical background: neutral transition soil
    opt_main = np.full((h, w, 3), [160, 150, 130], dtype=np.uint8)

    # Zone 1: Water (upper-left) - Dark Blue/Black
    opt_main[20:90, 20:110] = [20, 45, 95]
    # Zone 2: Built-Up (upper-right) - High contrast roofs
    opt_main[25:105, 145:225] = [200, 50, 40]   # Red tile roof
    opt_main[65:105, 145:185] = [100, 120, 145] # Grey-blue flat roof
    # Zone 3: Vegetation (lower-left) - Vibrant Green
    opt_main[145:225, 20:115] = [30, 175, 45]
    # Zone 4: Bare Soil (lower-right) - Warm Tan/Brown
    opt_main[145:225, 145:225] = [205, 155, 95]

    # SAR corresponding channels
    sar_main = np.full((h, w), 0.28, dtype=np.float32)
    # Water: Specular non-reflection -> near-zero backscatter & smooth
    sar_main[20:90, 20:110] = 0.03
    # Built-up: Double-bounce corner reflection -> strong backscatter & texture
    sar_built_patch = np.full((80, 80), 0.85, dtype=np.float32)
    sar_built_patch[::3, :] = 0.40  # Add building edges / structural variance
    sar_main[25:105, 145:225] = sar_built_patch
    # Vegetation: Moderate volume scattering
    sar_main[145:225, 20:115] = 0.42
    # Bare Soil: Low-to-moderate surface roughness backscatter
    sar_main[145:225, 145:225] = 0.26

    opt_p1 = fixture_dir / "optical_multimodal_01.tif"
    sar_p1 = fixture_dir / "sar_multimodal_01.tif"
    write_optical(opt_p1, opt_main)
    write_sar(sar_p1, sar_main)

    cases["standard_multimodal"] = OpticalSARFixtureCase(
        scenario_id="optsar_standard_01",
        description="Co-registered optical and SAR imagery featuring water, built structures, vegetation, and bare soil.",
        optical_path=opt_p1,
        sar_path=sar_p1,
        expected_classes=["water_body", "built_structure", "vegetation_or_cropland", "bare_ground_or_soil"],
        query="Use the optical and SAR images together to identify built-up and water-covered regions.",
    )

    # -----------------------------------------------------------------------
    # Case 2: SAR Dependency Sensitivity Test (Same Optical, Different SAR)
    # -----------------------------------------------------------------------
    # In optical, we have a greyish rectangular feature (ambiguous: flat pavement or 3D building).
    opt_ambig = np.full((h, w, 3), [170, 165, 150], dtype=np.uint8)
    opt_ambig[60:180, 60:180] = [180, 180, 185] # Light grey feature

    # SAR Condition A: Strong double-bounce & high roughness -> resolves as built_structure
    sar_cond_a = np.full((h, w), 0.25, dtype=np.float32)
    patch_rough = np.full((120, 120), 0.82, dtype=np.float32)
    patch_rough[::2, :] = 0.35
    sar_cond_a[60:180, 60:180] = patch_rough

    # SAR Condition B: Flat low backscatter & smooth -> resolves as road_or_infrastructure
    sar_cond_b = np.full((h, w), 0.25, dtype=np.float32)
    sar_cond_b[60:180, 60:180] = 0.10

    opt_sens_p = fixture_dir / "opt_sensitivity_ambig.tif"
    sar_sens_a_p = fixture_dir / "sar_sensitivity_built.tif"
    sar_sens_b_p = fixture_dir / "sar_sensitivity_smooth.tif"

    write_optical(opt_sens_p, opt_ambig)
    write_sar(sar_sens_a_p, sar_cond_a)
    write_sar(sar_sens_b_p, sar_cond_b)

    cases["sar_sensitivity_built"] = OpticalSARFixtureCase(
        scenario_id="optsar_sar_sens_built",
        description="Ambiguous optical feature resolved as built_structure by high SAR double-bounce.",
        optical_path=opt_sens_p,
        sar_path=sar_sens_a_p,
        expected_classes=["built_structure"],
        query="Identify the built-up areas using both optical and SAR information.",
    )

    cases["sar_sensitivity_smooth"] = OpticalSARFixtureCase(
        scenario_id="optsar_sar_sens_smooth",
        description="Identical optical feature resolved as road/smooth infrastructure by low SAR backscatter.",
        optical_path=opt_sens_p,
        sar_path=sar_sens_b_p,
        expected_classes=["road_or_infrastructure"],
        query="What land-cover classes can be identified using both sensors?",
    )

    # -----------------------------------------------------------------------
    # Case 3: Optical Dependency Sensitivity Test (Same SAR, Different Optical)
    # -----------------------------------------------------------------------
    # Constant SAR backscatter (low-to-moderate backscatter ~ 0.18)
    sar_const = np.full((h, w), 0.18, dtype=np.float32)
    sar_const_p = fixture_dir / "sar_sensitivity_const.tif"
    write_sar(sar_const_p, sar_const)

    # Optical A: Dark blue feature -> classified as water_body
    opt_blue = np.full((h, w, 3), [150, 140, 120], dtype=np.uint8)
    opt_blue[60:180, 60:180] = [20, 35, 95]
    opt_blue_p = fixture_dir / "opt_sensitivity_blue.tif"
    write_optical(opt_blue_p, opt_blue)

    # Optical B: Vibrant green feature -> classified as vegetation_or_cropland
    opt_green = np.full((h, w, 3), [150, 140, 120], dtype=np.uint8)
    opt_green[60:180, 60:180] = [30, 185, 40]
    opt_green_p = fixture_dir / "opt_sensitivity_green.tif"
    write_optical(opt_green_p, opt_green)

    cases["optical_sensitivity_water"] = OpticalSARFixtureCase(
        scenario_id="optsar_opt_sens_water",
        description="Identical SAR resolved as water_body due to dark blue optical response.",
        optical_path=opt_blue_p,
        sar_path=sar_const_p,
        expected_classes=["water_body"],
        query="Identify water-covered regions using the optical and SAR images.",
    )

    cases["optical_sensitivity_veg"] = OpticalSARFixtureCase(
        scenario_id="optsar_opt_sens_veg",
        description="Identical SAR resolved as vegetation due to bright green optical response.",
        optical_path=opt_green_p,
        sar_path=sar_const_p,
        expected_classes=["vegetation_or_cropland"],
        query="What land-cover classes can be identified using both sensors?",
    )

    # -----------------------------------------------------------------------
    # Case 4: Disjoint Image Pair (Negative Control for Spatial Incompatibility)
    # -----------------------------------------------------------------------
    disjoint_transform = from_origin(700000.0, 4000000.0, 10.0, 10.0) # >1000km away
    opt_disjoint_p = fixture_dir / "optical_disjoint.tif"
    write_optical(opt_disjoint_p, opt_main, custom_transform=disjoint_transform)

    cases["disjoint_pair"] = OpticalSARFixtureCase(
        scenario_id="optsar_disjoint_01",
        description="Optical and SAR rasters with zero spatial intersection.",
        optical_path=opt_disjoint_p,
        sar_path=sar_p1,
        expected_classes=[],
        query="Use the optical and SAR images together.",
    )

    # -----------------------------------------------------------------------
    # Case 5: Missing CRS (Negative Control for Unprojected Imagery)
    # -----------------------------------------------------------------------
    opt_no_crs_p = fixture_dir / "optical_no_crs.tif"
    write_optical(opt_no_crs_p, opt_main, custom_crs=None)

    cases["missing_crs"] = OpticalSARFixtureCase(
        scenario_id="optsar_no_crs_01",
        description="Optical image lacking a valid Coordinate Reference System.",
        optical_path=opt_no_crs_p,
        sar_path=sar_p1,
        expected_classes=[],
        query="Use both images.",
    )

    return cases
