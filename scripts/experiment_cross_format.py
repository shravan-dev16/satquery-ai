"""Controlled Cross-Format Alignment Experiment.

Compares 7 controlled pairs:
1. PNG -> PNG
2. PNG -> JPEG Q85
3. PNG -> JPEG Q50
4. JPEG Q85 -> TIFF
5. TIFF -> PNG
6. Unrelated image pair
7. Genuinely shifted image pair

Measures:
- alignment score
- detected offset
- phase peak
- consensus
- best NCC
- change detector result (changed px, ratio %)
- confidence
"""

from pathlib import Path
from PIL import Image
import numpy as np
import sys
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.preprocessing.bi_temporal_normalizer import BiTemporalNormalizer, ImageAlignmentEngine
from backend.models.change import ChangeDetectionSpecialist

spec = ChangeDetectionSpecialist()
BENCH_DIR = Path("datasets/change_robustness")
d_bld = BENCH_DIR / "01_building"
p_t1_png = d_bld / "t1.png"
p_t2_png = d_bld / "t2.png"

tmp_dir = Path("datasets/change_robustness/exp_tmp")
tmp_dir.mkdir(exist_ok=True)

# 2. PNG -> JPEG Q85
p_t2_jpg85 = tmp_dir / "t2_q85.jpg"
Image.open(p_t2_png).save(p_t2_jpg85, quality=85)

# 3. PNG -> JPEG Q50
p_t2_jpg50 = tmp_dir / "t2_q50.jpg"
Image.open(p_t2_png).save(p_t2_jpg50, quality=50)

# 4. JPEG Q85 -> TIFF
p_t1_jpg85 = tmp_dir / "t1_q85.jpg"
Image.open(p_t1_png).save(p_t1_jpg85, quality=85)
p_t2_tif = tmp_dir / "t2.tif"
Image.open(p_t2_png).save(p_t2_tif, format="TIFF")

# 5. TIFF -> PNG
p_t1_tif = tmp_dir / "t1.tif"
Image.open(p_t1_png).save(p_t1_tif, format="TIFF")

# 6. Unrelated image pair
p_unrelated = tmp_dir / "unrelated.png"
arr_unrel = np.full((256, 256, 3), (20, 60, 140), dtype=np.uint8)
arr_unrel[100:150, 50:200] = (200, 200, 200)
Image.fromarray(arr_unrel).save(p_unrelated)

# 7. Genuinely shifted image pair (e.g. shifted by dy=40, dx=30)
p_t2_shifted = tmp_dir / "t2_shifted.png"
arr2 = np.array(Image.open(p_t2_png))
arr2_shifted = np.roll(np.roll(arr2, shift=40, axis=0), shift=30, axis=1)
Image.fromarray(arr2_shifted).save(p_t2_shifted)

test_pairs = [
    ("1. PNG -> PNG", p_t1_png, p_t2_png),
    ("2. PNG -> JPEG Q85", p_t1_png, p_t2_jpg85),
    ("3. PNG -> JPEG Q50", p_t1_png, p_t2_jpg50),
    ("4. JPEG Q85 -> TIFF", p_t1_jpg85, p_t2_tif),
    ("5. TIFF -> PNG", p_t1_tif, p_t2_png),
    ("6. Unrelated pair", p_t1_png, p_unrelated),
    ("7. Genuinely shifted pair", p_t1_png, p_t2_shifted),
]

print("\n| Experiment | Alignment Score | Offset (dx, dy) | Phase Peak | Consensus | Best NCC | Changed Px | Ratio % | Confidence |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for label, p1, p2 in test_pairs:
    norm = BiTemporalNormalizer.normalize_pair(p1, p2)
    arr1, arr2 = norm.image1_array, norm.image2_array
    g1 = (0.2989 * arr1[0] + 0.5870 * arr1[1] + 0.1140 * arr1[2]).astype(np.float32)
    g2 = (0.2989 * arr2[0] + 0.5870 * arr2[1] + 0.1140 * arr2[2]).astype(np.float32)

    std1, std2 = float(np.std(g1)), float(np.std(g2))
    norm1 = (g1 - np.mean(g1)) / std1
    norm2 = (g2 - np.mean(g2)) / std2
    f1, f2 = np.fft.fft2(norm1), np.fft.fft2(norm2)
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-8)
    corr_map = np.real(np.fft.ifft2(cross_power))
    max_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    phase_peak = float(corr_map[max_idx])

    h, w = g1.shape
    dy = max_idx[0] if max_idx[0] <= h // 2 else max_idx[0] - h
    dx = max_idx[1] if max_idx[1] <= w // 2 else max_idx[1] - w
    shifted_g2 = np.roll(np.roll(g2, shift=dy, axis=0), shift=dx, axis=1) if (dx or dy) else g2
    dyn_range = max(10.0, float(np.ptp(g1)))
    consensus = float(np.mean(np.abs(g1 - shifted_g2) / dyn_range < 0.25))
    best_ncc = float(np.mean(norm1 * (shifted_g2 - np.mean(shifted_g2)) / np.std(shifted_g2)))

    try:
        out = spec.execute_change_detection(p1, p2)
        ch_px = str(out.parameters_used.get("changed_pixels", 0))
        pct = f"{out.parameters_used.get('change_ratio_pct', 0.0):.2f}%"
        conf = f"{out.confidence:.4f}"
    except Exception as e:
        ch_px = "REJECTED"
        pct = "N/A"
        conf = "0.0000 (REJECTED)"

    print(f"| {label} | {norm.alignment_score:.4f} | ({norm.offset_xy[0]:.1f}, {norm.offset_xy[1]:.1f}) | {phase_peak:.4f} | {consensus:.4f} | {best_ncc:.4f} | {ch_px} | {pct} | {conf} |")

shutil.rmtree(tmp_dir, ignore_errors=True)
