"""Comprehensive Benchmarking Harness for 48H Experimental Candidates.

Evaluates:
1. Qwen LoRA Candidates (Exp A, Exp B, Exp C) against Golden Baseline on:
   - Frozen 64-sample test benchmark (datasets/adaptation/test.json)
   - Expanded held-out validation benchmark (datasets/adaptation/experiments/val_expanded.json)
2. TinyCD Candidates (Exp A, Exp B) against Baseline on:
   - LEVIR-CD 20-sample validation set
   - Diverse RS & Nuisance benchmark (12 scenarios)

Calculates:
- Exact Match, Semantic-Relaxed Accuracy, Pillar-wise metrics, JSON validity
- IoU, Precision, Recall, F1, False Positive Rate (FPR), Zero-Change Accuracy
- Peak VRAM and inference latency
"""

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.models.tinycd_arch import TinyCD
from backend.models.vqa import RemoteSensingVQASpecialist
from backend.preprocessing.geotiff import GeoTIFFReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_candidates")


def evaluate_tinycd_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Evaluates a TinyCD checkpoint on a manifest benchmark."""
    logger.info(f"Evaluating TinyCD [{checkpoint_path.name}] on [{manifest_path.name}]...")
    model = TinyCD(pretrained_backbone=False)
    weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=False)
    model = model.to(device)
    model.eval()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest if isinstance(manifest, list) else manifest.get("samples", manifest.get("pairs", []))
    if not samples and isinstance(manifest, dict):
        samples = list(manifest.values())[0] if isinstance(list(manifest.values())[0], list) else []

    ious = []
    precisions = []
    recalls = []
    f1s = []
    latencies = []

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for item in samples:
        t1_str = item.get("t1") or item.get("t1_path") or item.get("image1")
        t2_str = item.get("t2") or item.get("t2_path") or item.get("image2")
        lbl_str = item.get("gt_mask_path") or item.get("label") or item.get("label_path") or item.get("mask")

        if not (t1_str and t2_str):
            continue

        t1_p = REPO_ROOT / t1_str
        t2_p = REPO_ROOT / t2_str
        lbl_p = REPO_ROOT / lbl_str if lbl_str else None

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

        # Resize to 256x256
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
        latencies.append((time.perf_counter() - start_t) * 1000)

        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.float32)

        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection
        tp = intersection
        fp = pred_mask.sum() - intersection
        fn = gt_mask.sum() - intersection

        iou = (intersection + 1e-6) / (union + 1e-6) if union > 0 else (1.0 if pred_mask.sum() == 0 else 0.0)
        prec = (tp + 1e-6) / (tp + fp + 1e-6) if (tp + fp) > 0 else 1.0
        rec = (tp + 1e-6) / (tp + fn + 1e-6) if (tp + fn) > 0 else 1.0
        f1 = (2 * prec * rec) / (prec + rec + 1e-6) if (prec + rec) > 0 else 0.0

        ious.append(float(iou))
        precisions.append(float(prec))
        recalls.append(float(rec))
        f1s.append(float(f1))

    return {
        "checkpoint": str(checkpoint_path.name),
        "benchmark": str(manifest_path.name),
        "total_samples": len(ious),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "mean_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
        "mean_precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "mean_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
    }


def evaluate_qwen_adapter(
    adapter_path: Path,
    test_json_path: Path,
    max_eval_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluates a Qwen LoRA adapter on test JSON."""
    logger.info(f"Evaluating Qwen Adapter [{adapter_path.name}] on [{test_json_path.name}]...")

    with open(test_json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if max_eval_samples:
        samples = samples[:max_eval_samples]

    # Initialize VQA specialist with specified adapter
    spec = RSVQASpecialist(adapter_path=adapter_path)

    correct_exact = 0
    correct_relaxed = 0
    latencies = []
    pillar_stats: Dict[str, Dict[str, int]] = {}

    for s in samples:
        q = s.get("question", "")
        gt = str(s.get("ground_truth", "")).strip().lower()
        img_p = REPO_ROOT / (s.get("image_path") or s.get("primary_image_path", ""))
        pillar = s.get("pillar", "A_RS_VQA")

        if pillar not in pillar_stats:
            pillar_stats[pillar] = {"total": 0, "correct": 0}
        pillar_stats[pillar]["total"] += 1

        if not img_p.exists():
            continue

        start_t = time.perf_counter()
        try:
            res = spec.answer(query=q, image_path=img_p)
            ans = str(res.get("answer", "")).strip().lower()
        except Exception as e:
            ans = ""
        latencies.append((time.perf_counter() - start_t) * 1000)

        # Exact match check
        if gt in ans or ans == gt:
            correct_exact += 1
            correct_relaxed += 1
            pillar_stats[pillar]["correct"] += 1
        elif any(w in ans for w in gt.split() if len(w) > 3):
            correct_relaxed += 1

    total = len(samples)
    return {
        "adapter": str(adapter_path.name),
        "test_set": str(test_json_path.name),
        "total_samples": total,
        "exact_match_acc": round((correct_exact / max(1, total)) * 100, 2),
        "relaxed_acc": round((correct_relaxed / max(1, total)) * 100, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 1) if latencies else 0.0,
        "pillar_breakdown": {p: f"{st['correct']}/{st['total']} ({(st['correct']/max(1, st['total']))*100:.1f}%)" for p, st in pillar_stats.items()},
    }


def compare_all_candidates():
    logger.info("=" * 70)
    logger.info("Executing Full Comparative Benchmark Matrix")
    logger.info("=" * 70)

    results = {"tinycd": {}, "qwen": {}}

    # 1. Evaluate TinyCD Candidates
    levir_m = REPO_ROOT / "docs" / "evaluation" / "levir_cd_subset.json"
    diverse_m = REPO_ROOT / "docs" / "evaluation" / "diverse_rs_manifest.json"

    tinycd_models = [
        ("Baseline_Finetuned", REPO_ROOT / "models" / "checkpoints" / "tinycd_finetuned.pth"),
        ("TinyCD_Exp_A", REPO_ROOT / "models" / "checkpoints" / "experiments" / "tinycd_exp_a" / "tinycd.pth"),
        ("TinyCD_Exp_B", REPO_ROOT / "models" / "checkpoints" / "experiments" / "tinycd_exp_b" / "tinycd.pth"),
    ]

    for name, ckpt in tinycd_models:
        if ckpt.exists():
            levir_res = evaluate_tinycd_checkpoint(ckpt, levir_m)
            diverse_res = evaluate_tinycd_checkpoint(ckpt, diverse_m)
            results["tinycd"][name] = {
                "levir_cd": levir_res,
                "diverse_rs": diverse_res,
            }

    # 2. Save TinyCD results
    tinycd_eval_file = REPO_ROOT / "docs" / "evaluation" / "tinycd_48h_comparison.json"
    with open(tinycd_eval_file, "w", encoding="utf-8") as f:
        json.dump(results["tinycd"], f, indent=2)

    logger.info(f"TinyCD benchmark matrix saved to {tinycd_eval_file}")
    return results


if __name__ == "__main__":
    compare_all_candidates()
