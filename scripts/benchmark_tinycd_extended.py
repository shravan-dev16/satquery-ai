"""Extended TinyCD Generalization & Fairness Evaluation.

Evaluates 3 TinyCD checkpoints:
1. Baseline Finetuned (models/checkpoints/tinycd_finetuned.pth)
2. TinyCD Exp A (models/checkpoints/experiments/tinycd_exp_a/tinycd.pth)
3. TinyCD Exp B (models/checkpoints/experiments/tinycd_exp_b/tinycd.pth)

Across 4 Distinct Evaluation Benchmarks:
1. LEVIR-CD Benchmark (20 pairs, high-resolution aerial building change)
2. Diverse RS Benchmark (12 pairs, cross-domain multi-sensor optical change)
3. OSCD Multi-Temporal Change Set (20 pairs, Sentinel-2 10m real surface change)
4. Zero-Change & Hard-Negative Nuisance Set (20 pairs, illumination/seasonal invariance)

Reports:
- Mean IoU
- Mean F1 Score
- Precision
- Recall
- False Positive Rate / False Alarm Pixels
- Zero-Change Anti-Hallucination Accuracy
- Mean Latency (ms)
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple
import numpy as np
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tinycd_extended")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.models.tinycd_arch import TinyCD
from backend.preprocessing.geotiff import GeoTIFFReader


def load_model(checkpoint_path: Path, device: str = "cpu") -> torch.nn.Module:
    model = TinyCD()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def evaluate_pairs(
    model: torch.nn.Module,
    pair_list: List[Dict[str, Any]],
    benchmark_name: str,
    device: str = "cpu",
    is_zero_change_benchmark: bool = False,
) -> Dict[str, Any]:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    ious, f1s, precisions, recalls = [], [], [], []
    fp_pixel_counts = []
    zero_change_correct = 0
    latencies = []

    for item in pair_list:
        t1_p = REPO_ROOT / item["t1_path"]
        t2_p = REPO_ROOT / item["t2_path"]
        lbl_p = (REPO_ROOT / item["label_path"]) if item.get("label_path") else None

        if not (t1_p.exists() and t2_p.exists()):
            continue

        try:
            im1_arr, _ = GeoTIFFReader.read_normalized_rgb(t1_p)
            im1 = Image.fromarray(im1_arr)
        except Exception:
            im1 = Image.open(t1_p).convert("RGB")

        try:
            im2_arr, _ = GeoTIFFReader.read_normalized_rgb(t2_p)
            im2 = Image.fromarray(im2_arr)
        except Exception:
            im2 = Image.open(t2_p).convert("RGB")

        w, h = 256, 256
        im1 = im1.resize((w, h), Image.Resampling.BILINEAR)
        im2 = im2.resize((w, h), Image.Resampling.BILINEAR)

        arr1 = (np.array(im1, dtype=np.float32) / 255.0 - mean) / std
        arr2 = (np.array(im2, dtype=np.float32) / 255.0 - mean) / std

        t1_t = torch.from_numpy(arr1.transpose(2, 0, 1)).unsqueeze(0).to(device)
        t2_t = torch.from_numpy(arr2.transpose(2, 0, 1)).unsqueeze(0).to(device)

        if lbl_p and lbl_p.is_file():
            lbl_arr = np.array(Image.open(lbl_p).convert("L").resize((w, h), Image.Resampling.NEAREST))
            gt_mask = (lbl_arr > 128).astype(np.float32)
        else:
            gt_mask = np.zeros((h, w), dtype=np.float32)

        start_t = time.perf_counter()
        with torch.no_grad():
            pred = model(t1_t, t2_t)
        lat_ms = (time.perf_counter() - start_t) * 1000.0
        latencies.append(lat_ms)

        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float32)

        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection
        tp = intersection
        fp = pred_mask.sum() - intersection
        fn = gt_mask.sum() - intersection

        fp_pixel_counts.append(int(fp))

        if gt_mask.sum() == 0:
            # Zero-change evaluation
            if pred_mask.sum() < 25:  # Tolerance of 25 false pixels (~0.03% noise)
                zero_change_correct += 1
            iou = 1.0 if pred_mask.sum() == 0 else 0.0
            prec = 1.0 if pred_mask.sum() == 0 else 0.0
            rec = 1.0
            f1 = 1.0 if pred_mask.sum() == 0 else 0.0
        else:
            iou = (intersection + 1e-6) / (union + 1e-6) if union > 0 else 0.0
            prec = (tp + 1e-6) / (tp + fp + 1e-6) if (tp + fp) > 0 else 1.0
            rec = (tp + 1e-6) / (tp + fn + 1e-6) if (tp + fn) > 0 else 1.0
            f1 = (2 * prec * rec) / (prec + rec + 1e-6) if (prec + rec) > 0 else 0.0

        ious.append(float(iou))
        precisions.append(float(prec))
        recalls.append(float(rec))
        f1s.append(float(f1))

    total = len(ious)
    return {
        "benchmark": benchmark_name,
        "total_pairs": total,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "mean_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "mean_precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "mean_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "mean_false_positive_pixels": round(float(np.mean(fp_pixel_counts)), 1) if fp_pixel_counts else 0.0,
        "zero_change_accuracy_pct": round((zero_change_correct / max(1, total)) * 100.0, 2) if is_zero_change_benchmark else None,
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
    }


def run_tinycd_evaluation(device: str = "cpu") -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(f"RUNNING EXTENDED TINYCD MULTI-BENCHMARK EVALUATION (Device: {device})")
    logger.info("=" * 80)

    # 1. Load Datasets
    # LEVIR-CD
    with open(REPO_ROOT / "docs" / "evaluation" / "levir_cd_subset.json", "r") as f:
        levir_data = json.load(f)
    levir_pairs = [
        {"t1_path": s["t1_path"], "t2_path": s["t2_path"], "label_path": s["gt_mask_path"]}
        for s in levir_data
    ]

    # Diverse RS
    with open(REPO_ROOT / "docs" / "evaluation" / "diverse_rs_manifest.json", "r") as f:
        diverse_data = json.load(f)["samples"]
    diverse_pairs = [
        {"t1_path": s["t1_path"], "t2_path": s["t2_path"], "label_path": s["gt_mask_path"]}
        for s in diverse_data
    ]

    # OSCD Change Set & Zero-Change Set
    with open(REPO_ROOT / "datasets" / "external" / "oscd" / "oscd_samples.json", "r") as f:
        oscd_all = json.load(f)

    oscd_change_pairs = [
        {"t1_path": s["t1_path"], "t2_path": s["t2_path"], "label_path": s["label_path"]}
        for s in oscd_all if s.get("is_change")
    ][:20]

    oscd_zero_pairs = [
        {"t1_path": s["t1_path"], "t2_path": s["t2_path"], "label_path": s["label_path"]}
        for s in oscd_all if not s.get("is_change")
    ][:20]

    checkpoints = {
        "Baseline_Finetuned": REPO_ROOT / "models" / "checkpoints" / "tinycd_finetuned.pth",
        "TinyCD_Exp_A": REPO_ROOT / "models" / "checkpoints" / "experiments" / "tinycd_exp_a" / "tinycd.pth",
        "TinyCD_Exp_B": REPO_ROOT / "models" / "checkpoints" / "experiments" / "tinycd_exp_b" / "tinycd.pth",
    }

    full_results = {}

    for ckpt_name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found: {ckpt_path}")
            continue

        logger.info(f"Evaluating checkpoint: {ckpt_name}...")
        model = load_model(ckpt_path, device=device)

        res_levir = evaluate_pairs(model, levir_pairs, "LEVIR-CD", device=device)
        res_diverse = evaluate_pairs(model, diverse_pairs, "Diverse_RS", device=device)
        res_oscd = evaluate_pairs(model, oscd_change_pairs, "OSCD_Change", device=device)
        res_zero = evaluate_pairs(model, oscd_zero_pairs, "OSCD_Zero_Change_Negatives", device=device, is_zero_change_benchmark=True)

        full_results[ckpt_name] = {
            "checkpoint_path": str(ckpt_path.relative_to(REPO_ROOT)),
            "levir_cd": res_levir,
            "diverse_rs": res_diverse,
            "oscd_change": res_oscd,
            "zero_change_negatives": res_zero,
        }

    out_file = REPO_ROOT / "docs" / "evaluation" / "tinycd_extended_comparison.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"Saved extended TinyCD comparison to {out_file}")
    return full_results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # To run concurrently without disturbing VLM evaluation, run on CPU
    run_tinycd_evaluation(device="cpu")
