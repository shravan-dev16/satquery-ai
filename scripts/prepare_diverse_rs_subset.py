"""Prepares deterministic diverse remote-sensing evaluation subset across 8 categories.

Covers:
1. Urban (building additions)
2. Infrastructure (commercial/runway structure on projected UTM GeoTIFF)
3. Forest (canopy clearing / deforestation)
4. Agriculture (crop conversion / plowing)
5. Water (reservoir expansion / flooding)
6. Coastal (tidal flat / coastline accretion)
7. Mining (quarry excavation / bare earth)
8. Nuisance Variations (Zero true physical change):
   - Solar illumination shift
   - Sun-azimuth shadow shift
   - Seasonal vegetation hue variance
   - Coregistration sub-pixel jitter
9. Qualitative-only Natural Satellite Terrain (GeoTIFF, no GT mask, strictly qualitative)

Generates:
- datasets/diverse_rs_eval_subset/A/
- datasets/diverse_rs_eval_subset/B/
- datasets/diverse_rs_eval_subset/label/
- docs/evaluation/diverse_rs_manifest.json
"""

import json
from pathlib import Path
import shutil
import sys

# Ensure repository root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin

from backend.evaluation.manifests import EvaluationManifest, EvaluationSample, SceneCategory


def prepare_diverse_subset() -> Path:
    base_dir = Path("datasets/diverse_rs_eval_subset")
    dir_a = base_dir / "A"
    dir_b = base_dir / "B"
    dir_l = base_dir / "label"

    for d in [dir_a, dir_b, dir_l]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("docs/evaluation/diverse_rs_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    samples: list[EvaluationSample] = []
    w, h = 512, 512

    # Helper to save RGB images and binary labels
    def save_sample(
        sample_id: str,
        name: str,
        img_a: np.ndarray,
        img_b: np.ndarray,
        gt_mask: np.ndarray | None,
        category: SceneCategory,
        nuisance: str | None = None,
        is_quant: bool = True,
        crs: str | None = None,
    ) -> None:
        p_a = dir_a / f"{name}.png"
        p_b = dir_b / f"{name}.png"
        p_l = (dir_l / f"{name}.png") if gt_mask is not None else None

        Image.fromarray(img_a).save(p_a)
        Image.fromarray(img_b).save(p_b)
        if p_l is not None and gt_mask is not None:
            Image.fromarray(gt_mask).save(p_l)

        ch_px = int(np.count_nonzero(gt_mask > 0)) if gt_mask is not None else None
        tot_px = int(img_a.shape[0] * img_a.shape[1])
        ch_pct = round((ch_px / tot_px) * 100.0, 4) if ch_px is not None else None

        record = EvaluationSample(
            sample_id=sample_id,
            filename=f"{name}.png",
            dataset="DiverseRS_Benchmark",
            split="val",
            license="CC-BY-4.0",
            dimensions=[img_a.shape[1], img_a.shape[0]],
            total_pixels=tot_px,
            changed_pixels=ch_px,
            change_ratio_pct=ch_pct,
            t1_path=str(p_a).replace("\\", "/"),
            t2_path=str(p_b).replace("\\", "/"),
            gt_mask_path=str(p_l).replace("\\", "/") if p_l else None,
            scene_category=category,
            is_quantitative=is_quant,
            nuisance_type=nuisance,
            crs=crs,
        )
        samples.append(record)

    # -------------------------------------------------------------
    # 1. Urban (Building additions from LEVIR-CD val_1)
    # -------------------------------------------------------------
    levir_a = Path("datasets/levir_cd_eval_subset/A/val_1.png")
    levir_b = Path("datasets/levir_cd_eval_subset/B/val_1.png")
    levir_l = Path("datasets/levir_cd_eval_subset/label/val_1.png")

    if levir_a.exists():
        # Crop 512x512 tile
        im_a = np.array(Image.open(levir_a).convert("RGB"))[:512, :512]
        im_b = np.array(Image.open(levir_b).convert("RGB"))[:512, :512]
        im_l = np.array(Image.open(levir_l).convert("L"))[:512, :512]
        save_sample("diverse_urban_001", "urban_01", im_a, im_b, im_l, SceneCategory.URBAN)

    # -------------------------------------------------------------
    # 2. Infrastructure (Projected UTM GeoTIFF)
    # -------------------------------------------------------------
    # Copy from tests/fixtures/bitemporal
    inf_t1 = Path("tests/fixtures/bitemporal/time1_pre_change.tif")
    inf_t2 = Path("tests/fixtures/bitemporal/time2_post_change.tif")
    inf_gt = Path("tests/fixtures/bitemporal/change_ground_truth.png")
    if inf_t1.exists() and inf_t2.exists() and inf_gt.exists():
        p_a = dir_a / "infrastructure_01.tif"
        p_b = dir_b / "infrastructure_01.tif"
        p_l = dir_l / "infrastructure_01.png"
        shutil.copyfile(inf_t1, p_a)
        shutil.copyfile(inf_t2, p_b)
        shutil.copyfile(inf_gt, p_l)

        with rasterio.open(p_a) as src:
            w_i, h_i = src.width, src.height
        gt_arr = np.array(Image.open(p_l).convert("L"))
        ch_px = int(np.count_nonzero(gt_arr > 0))
        tot_px = w_i * h_i

        samples.append(
            EvaluationSample(
                sample_id="diverse_infra_001",
                filename="infrastructure_01.tif",
                dataset="DiverseRS_Benchmark",
                split="val",
                license="CC-BY-4.0",
                dimensions=[w_i, h_i],
                total_pixels=tot_px,
                changed_pixels=ch_px,
                change_ratio_pct=round((ch_px / tot_px) * 100.0, 4),
                t1_path=str(p_a).replace("\\", "/"),
                t2_path=str(p_b).replace("\\", "/"),
                gt_mask_path=str(p_l).replace("\\", "/"),
                scene_category=SceneCategory.INFRASTRUCTURE,
                is_quantitative=True,
                crs="EPSG:32643",
            )
        )

    # -------------------------------------------------------------
    # 3. Forest (Canopy Clearing / Deforestation)
    # -------------------------------------------------------------
    np.random.seed(42)
    # T1: Deep green forest texture
    forest_t1 = np.zeros((h, w, 3), dtype=np.uint8)
    forest_t1[:, :, 0] = np.random.normal(35, 10, (h, w)).clip(20, 60).astype(np.uint8)
    forest_t1[:, :, 1] = np.random.normal(90, 15, (h, w)).clip(60, 140).astype(np.uint8)
    forest_t1[:, :, 2] = np.random.normal(30, 8, (h, w)).clip(15, 55).astype(np.uint8)

    # T2: Forest clearing in central section (soil/stump brown-tan)
    forest_t2 = forest_t1.copy()
    gt_forest = np.zeros((h, w), dtype=np.uint8)
    # Clearing patch 150:350, 150:350
    clearing_y1, clearing_y2 = 160, 340
    clearing_x1, clearing_x2 = 160, 360
    gt_forest[clearing_y1:clearing_y2, clearing_x1:clearing_x2] = 255
    forest_t2[clearing_y1:clearing_y2, clearing_x1:clearing_x2, 0] = np.random.normal(160, 12, (180, 200)).clip(130, 200).astype(np.uint8)
    forest_t2[clearing_y1:clearing_y2, clearing_x1:clearing_x2, 1] = np.random.normal(130, 10, (180, 200)).clip(100, 160).astype(np.uint8)
    forest_t2[clearing_y1:clearing_y2, clearing_x1:clearing_x2, 2] = np.random.normal(85, 10, (180, 200)).clip(60, 120).astype(np.uint8)
    save_sample("diverse_forest_001", "forest_01", forest_t1, forest_t2, gt_forest, SceneCategory.FOREST)

    # -------------------------------------------------------------
    # 4. Agriculture (Crop Harvesting / Field Conversion)
    # -------------------------------------------------------------
    # T1: Green vigorous crop fields
    agri_t1 = np.full((h, w, 3), [60, 150, 50], dtype=np.uint8)
    agri_t1 += np.random.normal(0, 8, (h, w, 3)).clip(-20, 20).astype(np.uint8)

    # T2: Field plowed to bare brown earth in northern half
    agri_t2 = agri_t1.copy()
    gt_agri = np.zeros((h, w), dtype=np.uint8)
    gt_agri[50:230, 80:430] = 255
    agri_t2[50:230, 80:430] = [175, 140, 95]
    agri_t2[50:230, 80:430] += np.random.normal(0, 10, (180, 350, 3)).clip(-25, 25).astype(np.uint8)
    save_sample("diverse_agri_001", "agriculture_01", agri_t1, agri_t2, gt_agri, SceneCategory.AGRICULTURE)

    # -------------------------------------------------------------
    # 5. Water (Reservoir Expansion / Flood Inundation)
    # -------------------------------------------------------------
    # T1: Dry floodplain with narrow river channel
    water_t1 = np.full((h, w, 3), [160, 145, 120], dtype=np.uint8)
    water_t1[:, 240:270] = [30, 60, 110]  # river channel
    water_t1 += np.random.normal(0, 6, (h, w, 3)).clip(-15, 15).astype(np.uint8)

    # T2: Lake inundation expansion
    water_t2 = water_t1.copy()
    gt_water = np.zeros((h, w), dtype=np.uint8)
    gt_water[120:380, 120:400] = 255
    # Exclude original river from change GT
    gt_water[:, 240:270] = 0
    water_t2[120:380, 120:400] = [25, 55, 105]
    water_t2 += np.random.normal(0, 5, (h, w, 3)).clip(-12, 12).astype(np.uint8)
    save_sample("diverse_water_001", "water_01", water_t1, water_t2, gt_water, SceneCategory.WATER)

    # -------------------------------------------------------------
    # 6. Coastal (Mudflat Accretion / Sediment Shift)
    # -------------------------------------------------------------
    # T1: Coastal shoreline
    coast_t1 = np.full((h, w, 3), [40, 80, 140], dtype=np.uint8)  # ocean water
    coast_t1[:200, :] = [200, 190, 150]  # beach/land
    coast_t1 += np.random.normal(0, 5, (h, w, 3)).clip(-12, 12).astype(np.uint8)

    # T2: Sand bar deposit / intertidal accretion
    coast_t2 = coast_t1.copy()
    gt_coast = np.zeros((h, w), dtype=np.uint8)
    gt_coast[200:290, 150:380] = 255
    coast_t2[200:290, 150:380] = [185, 175, 135]
    coast_t2 += np.random.normal(0, 5, (h, w, 3)).clip(-12, 12).astype(np.uint8)
    save_sample("diverse_coastal_001", "coastal_01", coast_t1, coast_t2, gt_coast, SceneCategory.COASTAL)

    # -------------------------------------------------------------
    # 7. Mining (Open-Pit Excavation)
    # -------------------------------------------------------------
    # T1: Natural scrubland
    mine_t1 = np.full((h, w, 3), [130, 125, 95], dtype=np.uint8)
    mine_t1 += np.random.normal(0, 8, (h, w, 3)).clip(-20, 20).astype(np.uint8)

    # T2: Excavation pit with exposed bright chalk/gravel
    mine_t2 = mine_t1.copy()
    gt_mine = np.zeros((h, w), dtype=np.uint8)
    gt_mine[180:360, 160:370] = 255
    mine_t2[180:360, 160:370] = [225, 220, 210]
    mine_t2 += np.random.normal(0, 8, (h, w, 3)).clip(-20, 20).astype(np.uint8)
    save_sample("diverse_mining_001", "mining_01", mine_t1, mine_t2, gt_mine, SceneCategory.MINING)

    # -------------------------------------------------------------
    # 8. Nuisance Variations (Zero True Physical Change -> Empty GT)
    # -------------------------------------------------------------
    empty_gt = np.zeros((h, w), dtype=np.uint8)

    # 8a. Illumination Shift (Solar angle / atmospheric brightness delta)
    base_illum_t1 = np.full((h, w, 3), [140, 130, 110], dtype=np.uint8)
    base_illum_t1[150:250, 150:250] = [180, 50, 50]  # building
    base_illum_t1 += np.random.normal(0, 5, (h, w, 3)).clip(-15, 15).astype(np.uint8)
    # T2 has uniform +25 brightness shift, 0 actual structural change
    illum_t2 = np.clip(base_illum_t1.astype(np.int16) + 28, 0, 255).astype(np.uint8)
    save_sample(
        "diverse_nuisance_illum_001",
        "nuisance_illumination_01",
        base_illum_t1,
        illum_t2,
        empty_gt,
        SceneCategory.NUISANCE_VARIATION,
        nuisance="illumination",
    )

    # 8b. Shadow Shift (Building cast shadow moved due to solar azimuth)
    base_shadow_t1 = np.full((h, w, 3), [150, 140, 120], dtype=np.uint8)
    base_shadow_t1[200:300, 200:300] = [210, 210, 210]  # bright warehouse
    base_shadow_t1[200:300, 300:350] = [60, 55, 45]     # shadow cast east
    shadow_t2 = base_shadow_t1.copy()
    shadow_t2[200:300, 300:350] = [150, 140, 120]       # remove east shadow
    shadow_t2[300:350, 200:300] = [60, 55, 45]         # cast south shadow
    # Zero change in building itself
    save_sample(
        "diverse_nuisance_shadow_001",
        "nuisance_shadow_01",
        base_shadow_t1,
        shadow_t2,
        empty_gt,
        SceneCategory.NUISANCE_VARIATION,
        nuisance="shadow",
    )

    # 8c. Seasonal Vegetation Phenology (Summer green -> Autumn yellow-green)
    base_veg_t1 = np.full((h, w, 3), [50, 145, 45], dtype=np.uint8)
    base_veg_t1 += np.random.normal(0, 6, (h, w, 3)).clip(-15, 15).astype(np.uint8)
    # T2: Autumn phenological shift (more yellow, less green)
    veg_t2 = np.full((h, w, 3), [135, 130, 40], dtype=np.uint8)
    veg_t2 += np.random.normal(0, 6, (h, w, 3)).clip(-15, 15).astype(np.uint8)
    save_sample(
        "diverse_nuisance_veg_001",
        "nuisance_seasonal_01",
        base_veg_t1,
        veg_t2,
        empty_gt,
        SceneCategory.NUISANCE_VARIATION,
        nuisance="seasonal_color",
    )

    # 8d. Coregistration / Alignment Jitter (Sub-pixel shift)
    reg_t1 = np.full((h, w, 3), [120, 120, 120], dtype=np.uint8)
    # Distinct sharp linear features (roads/runway)
    reg_t1[250:265, :] = [240, 240, 240]
    reg_t1[:, 250:265] = [240, 240, 240]
    # T2 is rolled by 2 pixels (misalignment artifact)
    reg_t2 = np.roll(reg_t1, shift=(2, 2), axis=(0, 1))
    save_sample(
        "diverse_nuisance_reg_001",
        "nuisance_registration_01",
        reg_t1,
        reg_t2,
        empty_gt,
        SceneCategory.NUISANCE_VARIATION,
        nuisance="registration",
    )

    # -------------------------------------------------------------
    # 9. Qualitative-Only Natural Multi-Spectral Satellite Terrain
    # (Real GeoTIFF, no GT mask, strictly is_quantitative=False)
    # -------------------------------------------------------------
    real_sample_p = Path("tests/fixtures/real_rs_sample.tif")
    if real_sample_p.exists():
        p_a = dir_a / "natural_terrain_01.tif"
        p_b = dir_b / "natural_terrain_01.tif"
        shutil.copyfile(real_sample_p, p_a)
        shutil.copyfile(real_sample_p, p_b)

        with rasterio.open(p_a) as src:
            w_r, h_r = src.width, src.height
            crs_r = str(src.crs)

        samples.append(
            EvaluationSample(
                sample_id="diverse_natural_terrain_001",
                filename="natural_terrain_01.tif",
                dataset="DiverseRS_Benchmark",
                split="val",
                license="BSD-3-Clause",
                dimensions=[w_r, h_r],
                total_pixels=w_r * h_r,
                changed_pixels=None,
                change_ratio_pct=None,
                t1_path=str(p_a).replace("\\", "/"),
                t2_path=str(p_b).replace("\\", "/"),
                gt_mask_path=None,
                scene_category=SceneCategory.FOREST,
                is_quantitative=False,
                crs=crs_r,
                metadata={"note": "Qualitative verification of natural multispectral terrain without fake GT"},
            )
        )

    # Save manifest
    manifest = EvaluationManifest(
        manifest_id="diverse_rs_manifest_v1",
        dataset_name="Diverse Remote Sensing & Nuisance Benchmark",
        version="1.0.0",
        description=(
            "Deterministic 12-sample evaluation split covering 7 land-cover categories, "
            "4 nuisance variations (zero physical change), and 1 qualitative natural satellite GeoTIFF."
        ),
        license="CC-BY-4.0 / Mixed Open",
        samples=samples,
    )
    manifest.save(manifest_path)
    print(
        f"Generated diverse evaluation manifest with {manifest.sample_count} samples "
        f"({manifest.quantitative_count} quantitative, {manifest.qualitative_count} qualitative) in {manifest_path}",
        flush=True,
    )
    return manifest_path


if __name__ == "__main__":
    prepare_diverse_subset()
