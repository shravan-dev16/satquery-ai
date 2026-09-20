"""Format Robustness Audit Script.

Computes exact evaluation metrics across formats, clearly distinguishing:
1. Operational Benchmark Scenarios (10 distinct scenarios in datasets/change_robustness/)
2. Pair Format Evaluations (12 evaluated pair records across format combinations)
"""

from collections import defaultdict
import json
from pathlib import Path
import sys
from PIL import Image
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.models.change import ChangeDetectionSpecialist
from backend.preprocessing.bi_temporal_normalizer import BiTemporalNormalizer, ImageAlignmentEngine

spec = ChangeDetectionSpecialist()
spec_cva = ChangeDetectionSpecialist(candidate="cva")
BENCH_DIR = Path("datasets/change_robustness")
OUTPUT_JSON = Path("docs/evaluation/format_matrix_audit.json")

# =====================================================================
# PART 1: 10 Operational Benchmark Scenarios
# =====================================================================
scenarios = [
    ("01_building", "t1.png", "t2.png", True, "PNG -> PNG", "Obvious building change"),
    ("02_vegetation", "t1.png", "t2.png", True, "PNG -> PNG", "Vegetation canopy loss"),
    ("03_water", "t1.png", "t2.png", True, "PNG -> PNG", "Water surface expansion"),
    ("04_roads", "t1.png", "t2.png", True, "PNG -> PNG", "Road network addition"),
    ("05_construction", "t1.jpg", "t2.jpg", True, "JPEG -> JPEG", "Bare ground excavation"),
    ("06_no_change", "t1.png", "t2.png", False, "PNG -> PNG", "Zero physical change"),
    ("07_seasonal", "t1.png", "t2.png", False, "PNG -> PNG", "Seasonal phenology shift"),
    ("08_compression", "t1.jpg", "t2.jpg", False, "JPEG -> JPEG", "JPEG Q=30 artifact rejection"),
    ("09_misalignment", "t1.png", "t2.png", None, "PNG -> PNG", "Spatial translation jitter (-55,-45)px"),
    ("10_cross_format", "t1.png", "t2.tif", True, "PNG -> TIFF", "Cross-format building change"),
]

scenario_records = []
for sc, f1, f2, exp_change, fmt, desc in scenarios:
    d = BENCH_DIR / sc
    p1, p2 = d / f1, d / f2
    norm = BiTemporalNormalizer.normalize_pair(p1, p2)
    out = spec.execute_change_detection(p1, p2)
    ch_px = out.parameters_used.get("changed_pixels", 0)
    pct = out.parameters_used.get("change_ratio_pct", 0.0)
    pred_ch = (pct >= 1.0) and (ch_px > 100)

    if sc == "09_misalignment":
        shift_detected = abs(norm.offset_xy[0]) > 20 or abs(norm.offset_xy[1]) > 20
        warn_emitted = any("shift" in w.lower() or "misalignment" in w.lower() or "offset" in w.lower() for w in norm.warnings)
        passed = shift_detected and warn_emitted
        eval_notes = f"Shift detected {norm.offset_xy}, warning emitted"
    else:
        passed = (pred_ch == exp_change)
        eval_notes = f"Changed {ch_px} px ({pct:.2f}% AOI)"

    scenario_records.append({
        "scenario_id": sc,
        "format": fmt,
        "description": desc,
        "expected_change": exp_change,
        "predicted_change": pred_ch,
        "changed_pixels": ch_px,
        "ratio_pct": pct,
        "alignment_score": norm.alignment_score,
        "passed": passed,
        "notes": eval_notes,
    })

# =====================================================================
# PART 2: 12 Pair Format Evaluations
# =====================================================================
pair_tests = [
    ("01_building", BENCH_DIR / "01_building" / "t1.png", BENCH_DIR / "01_building" / "t2.png", True, "Same format", "PNG -> PNG"),
    ("02_vegetation", BENCH_DIR / "02_vegetation" / "t1.png", BENCH_DIR / "02_vegetation" / "t2.png", True, "Same format", "PNG -> PNG"),
    ("03_water", BENCH_DIR / "03_water" / "t1.png", BENCH_DIR / "03_water" / "t2.png", True, "Same format", "PNG -> PNG"),
    ("04_roads", BENCH_DIR / "04_roads" / "t1.png", BENCH_DIR / "04_roads" / "t2.png", True, "Same format", "PNG -> PNG"),
    ("06_no_change", BENCH_DIR / "06_no_change" / "t1.png", BENCH_DIR / "06_no_change" / "t2.png", False, "Same format", "PNG -> PNG"),
    ("07_seasonal", BENCH_DIR / "07_seasonal" / "t1.png", BENCH_DIR / "07_seasonal" / "t2.png", False, "Same format", "PNG -> PNG"),
    ("05_construction", BENCH_DIR / "05_construction" / "t1.jpg", BENCH_DIR / "05_construction" / "t2.jpg", True, "Same format", "JPEG -> JPEG"),
    ("08_compression", BENCH_DIR / "08_compression" / "t1.jpg", BENCH_DIR / "08_compression" / "t2.jpg", False, "Same format", "JPEG -> JPEG"),
    ("10_cross_format", BENCH_DIR / "10_cross_format" / "t1.png", BENCH_DIR / "10_cross_format" / "t2.tif", True, "Mixed format", "PNG -> TIFF"),
    ("geotiff_pair", Path("tests/fixtures/bitemporal/time1_pre_change.tif"), Path("tests/fixtures/bitemporal/time2_post_change.tif"), True, "Same format", "GeoTIFF -> GeoTIFF"),
]

# Synthesize Mixed Format Tests: PNG -> JPEG, JPEG -> TIFF
d_bld = BENCH_DIR / "01_building"
t2_jpg = BENCH_DIR / "scratch_t2.jpg"
t1_jpg = BENCH_DIR / "scratch_t1.jpg"
t2_tif = BENCH_DIR / "scratch_t2.tif"
Image.open(d_bld / "t2.png").save(t2_jpg, quality=85)
Image.open(d_bld / "t1.png").save(t1_jpg, quality=85)
Image.open(d_bld / "t2.png").save(t2_tif, format="TIFF")

pair_tests.append(("mixed_png_jpeg", d_bld / "t1.png", t2_jpg, True, "Mixed format", "PNG -> JPEG"))
pair_tests.append(("mixed_jpeg_tiff", t1_jpg, t2_tif, True, "Mixed format", "JPEG -> TIFF"))

pair_records = []
for name, p1, p2, exp_ch, ptype, fmt in pair_tests:
    norm = BiTemporalNormalizer.normalize_pair(p1, p2)
    align_failed = norm.alignment_score < 0.60
    out = spec.execute_change_detection(p1, p2)
    ch_px = out.parameters_used.get("changed_pixels", 0)
    pct = out.parameters_used.get("change_ratio_pct", 0.0)
    pred_ch = (pct >= 1.0) and (ch_px > 100)
    cor = (pred_ch == exp_ch)
    fp = pred_ch and not exp_ch
    pair_records.append({
        "pair_name": name,
        "format": fmt,
        "pair_type": ptype,
        "expected_change": exp_ch,
        "predicted_change": pred_ch,
        "changed_pixels": ch_px,
        "ratio_pct": pct,
        "alignment_score": norm.alignment_score,
        "align_failed": align_failed,
        "correct": cor,
        "false_positive": fp,
    })

# Clean scratch files
if t2_jpg.exists():
    t2_jpg.unlink()
if t1_jpg.exists():
    t1_jpg.unlink()
if t2_tif.exists():
    t2_tif.unlink()

# Per-Format Aggregation
matrix = defaultdict(lambda: {
    "pair_type": "",
    "total": 0,
    "correct": 0,
    "align_failures": 0,
    "false_positives": 0,
})

for r in pair_records:
    fmt = r["format"]
    matrix[fmt]["total"] += 1
    matrix[fmt]["pair_type"] = r["pair_type"]
    if r["correct"]:
        matrix[fmt]["correct"] += 1
    if r["align_failed"]:
        matrix[fmt]["align_failures"] += 1
    if r["false_positive"]:
        matrix[fmt]["false_positives"] += 1

# Summary Metrics
total_scenarios = len(scenario_records)
scenarios_passed = sum(1 for s in scenario_records if s["passed"])
scenario_accuracy = round((scenarios_passed / total_scenarios) * 100.0, 2)

total_pairs = len(pair_records)
pairs_correct = sum(1 for p in pair_records if p["correct"])
pair_overall_accuracy = round((pairs_correct / total_pairs) * 100.0, 2)
total_false_positives = sum(1 for p in pair_records if p["false_positive"])
total_align_failures = sum(1 for p in pair_records if p["align_failed"])

audit_summary = {
    "total_scenario_evaluations": total_scenarios,
    "scenarios_passed": scenarios_passed,
    "scenario_accuracy_pct": scenario_accuracy,
    "total_pair_evaluations": total_pairs,
    "pairs_correct": pairs_correct,
    "pair_overall_accuracy_pct": pair_overall_accuracy,
    "total_false_positives": total_false_positives,
    "total_alignment_failures": total_align_failures,
    "per_format_breakdown": {
        fmt: {
            "pair_type": s["pair_type"],
            "sample_count": s["total"],
            "correct_count": s["correct"],
            "accuracy_pct": round((s["correct"] / s["total"]) * 100.0, 2),
            "alignment_failures": s["align_failures"],
            "false_positives": s["false_positives"],
        }
        for fmt, s in sorted(matrix.items())
    },
    "scenarios": scenario_records,
    "pair_records": pair_records,
}

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(audit_summary, indent=2))

# Markdown Output
print("=" * 80)
print("SATQUERY AI — CANONICAL FORMAT ROBUSTNESS AUDIT RESULTS")
print("=" * 80)
print(f"Total Scenario Evaluations: {total_scenarios}")
print(f"Scenarios Passed:           {scenarios_passed}/{total_scenarios} ({scenario_accuracy}%)")
print(f"Total Pair Evaluations:     {total_pairs}")
print(f"Pairs Correct:              {pairs_correct}/{total_pairs} ({pair_overall_accuracy}%)")
print(f"Total False Positives:      {total_false_positives}")
print(f"Total Alignment Failures:   {total_align_failures}")
print("-" * 80)

print("\n### PER-FORMAT PAIR EVALUATION MATRIX (12 Pairs)")
print("| Format | Pair type | Samples | Correct | Accuracy | Alignment failures | False positives |")
print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
for fmt, s in sorted(matrix.items()):
    acc = (s["correct"] / s["total"]) * 100.0
    pt = s["pair_type"]
    tot = s["total"]
    cor = s["correct"]
    af = s["align_failures"]
    fp = s["false_positives"]
    print(f"| {fmt} | {pt} | {tot} | {cor} | {acc:.1f}% | {af} | {fp} |")

print("\n### 10 OPERATIONAL BENCHMARK SCENARIOS")
print("| ID | Format | Description | Changed Px | Ratio % | Alignment | Passed? |")
print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")
for s in scenario_records:
    sc = s["scenario_id"]
    fmt = s["format"]
    desc = s["description"]
    ch = s["changed_pixels"]
    pct = s["ratio_pct"]
    align = f"{s['alignment_score']:.3f}"
    pas = "PASS" if s["passed"] else "FAIL"
    print(f"| {sc} | {fmt} | {desc} | {ch} | {pct:.2f}% | {align} | {pas} |")
