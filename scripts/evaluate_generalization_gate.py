"""Comprehensive Generalization Gate & Latency Fairness Benchmark.

Evaluates:
- Golden M10 RS-LoRA Baseline (models/adapters/qwen2_vl_rs_lora_m10_golden)
- Candidate A Best Checkpoint (models/adapters/experiments/qwen_rs_exp_a/best_checkpoint)

On the EXACT SAME Independent Test Benchmark:
datasets/adaptation/independent_test_200.json (178 samples across RSVQA-HR, CDVQA Test, and OSCD).

Under strictly identical:
- Hardware (RTX 4070 SUPER 12GB)
- Runtime (SharedVLMRuntime, BF16, torch.inference_mode(), SDPA)
- Cold/Warm condition tracking
- Exact scoring logic
"""

import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generalization_gate")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.agent.registry import registry
from backend.agent.schema import SpecialistInput, TaskType
from backend.main import register_default_specialists
from backend.models.shared_vlm import release_shared_vlm


def score_vqa_answer(pred_raw: str, gt_raw: str) -> Tuple[bool, bool]:
    """Computes exact match and relaxed match for remote sensing VQA."""
    pred = pred_raw.strip().lower()
    gt = gt_raw.strip().lower()

    # Normalize punctuation
    pred_clean = re.sub(r"[^\w\s]", "", pred)
    gt_clean = re.sub(r"[^\w\s]", "", gt)

    # Direct match or exact token presence
    is_exact = (pred == gt) or (pred_clean == gt_clean)
    if not is_exact:
        # Match single-word answers (yes/no/numbers/classes)
        pred_words = pred_clean.split()
        gt_words = gt_clean.split()
        if len(gt_words) == 1 and gt_words[0] in pred_words:
            is_exact = True
        elif gt in pred:
            is_exact = True

    is_relaxed = is_exact or any(w in pred_clean for w in gt_clean.split() if len(w) > 3)
    return is_exact, is_relaxed


def evaluate_adapter_on_independent_set(
    adapter_path: Path,
    model_label: str,
    test_file: Path = REPO_ROOT / "datasets" / "adaptation" / "independent_test_200.json",
    output_json: Path = REPO_ROOT / "docs" / "evaluation" / "eval_indep_result.json",
) -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info(f"BENCHMARKING: {model_label}")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Test Set: {test_file}")
    logger.info("=" * 80)

    # Force cleanup of GPU state
    release_shared_vlm(force_cleanup=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    os.environ["SATQUERY_USE_ADAPTED_VLM"] = "1"
    os.environ["SATQUERY_ADAPTER_PATH"] = str(adapter_path).replace("\\", "/")

    registry.clear()
    register_default_specialists()

    vqa_spec = registry.get("RS_VQA")
    cvqa_spec = registry.get("CHANGE_VQA")

    with open(test_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    logger.info(f"Loaded {len(samples)} independent evaluation samples.")

    # Cold start timing: load specialists
    t_load_start = time.perf_counter()
    vqa_spec.load()
    cvqa_spec.load()
    cold_load_time_ms = round((time.perf_counter() - t_load_start) * 1000.0, 2)
    logger.info(f"Cold-start load time: {cold_load_time_ms} ms")

    records = []
    latencies = []
    cold_latency = None

    dataset_stats: Dict[str, Dict[str, Any]] = {
        "RSVQA-HR": {"total": 0, "correct": 0, "relaxed_correct": 0, "latencies": []},
        "CDVQA_Test": {"total": 0, "correct": 0, "valid_json": 0, "latencies": []},
        "OSCD": {"total": 0, "correct": 0, "valid_json": 0, "latencies": []},
    }

    for idx, s in enumerate(samples):
        ds = s.get("dataset", "unknown")
        task = s.get("task", "RS_VQA")
        q = s.get("question", "")
        gt = s.get("ground_truth", "")
        gt_dir = s.get("ground_truth_direction")

        # Map dataset key
        if "RSVQA" in ds:
            ds_key = "RSVQA-HR"
        elif "CDVQA" in ds:
            ds_key = "CDVQA_Test"
        else:
            ds_key = "OSCD"

        dataset_stats[ds_key]["total"] += 1

        t0 = time.perf_counter()

        if task == "RS_VQA":
            img_path = str(REPO_ROOT / s["image_path"])
            inp = SpecialistInput(task=TaskType.VQA, query=q, primary_image_path=img_path)
            out = vqa_spec.predict(inp)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            ans = out.answer_text or ""
            is_exact, is_relaxed = score_vqa_answer(ans, gt)

            if is_exact:
                dataset_stats[ds_key]["correct"] += 1
            if is_relaxed:
                dataset_stats[ds_key]["relaxed_correct"] += 1
            dataset_stats[ds_key]["latencies"].append(lat_ms)

            records.append({
                "sample_id": s["sample_id"],
                "dataset": ds_key,
                "task": task,
                "question": q,
                "ground_truth": gt,
                "predicted": ans,
                "is_correct": is_exact,
                "latency_ms": round(lat_ms, 2),
            })

        else:  # CHANGE_VQA
            p1 = str(REPO_ROOT / s["primary_image_path"])
            p2 = str(REPO_ROOT / s["secondary_image_path"])
            inp = SpecialistInput(
                task=TaskType.CHANGE_VQA,
                query=q,
                primary_image_path=p1,
                secondary_image_path=p2,
                parameters={"changed_pixels": 1200, "change_ratio_pct": 2.5, "total_clusters": 2},
            )
            out = cvqa_spec.predict(inp)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            interp = out.evidence.semantic_interpretation if out.evidence else None
            is_valid_json = interp is not None and bool(interp.temporal_direction)
            if is_valid_json:
                dataset_stats[ds_key]["valid_json"] += 1

            pred_dir = interp.temporal_direction.lower() if (interp and interp.temporal_direction) else "unknown"
            pred_ans = out.answer_text or (interp.summary if interp else "")

            # Check correctness: direction match or yes/no answer match
            is_corr = False
            if gt.lower() in ["yes", "no"]:
                if gt.lower() == "yes" and pred_dir in ["increased", "modified", "decreased"]:
                    is_corr = True
                elif gt.lower() == "no" and pred_dir in ["no_change"]:
                    is_corr = True
                elif gt.lower() in pred_ans.lower():
                    is_corr = True
            elif gt_dir and (pred_dir == gt_dir.lower()):
                is_corr = True

            if is_corr:
                dataset_stats[ds_key]["correct"] += 1
            dataset_stats[ds_key]["latencies"].append(lat_ms)

            records.append({
                "sample_id": s["sample_id"],
                "dataset": ds_key,
                "task": task,
                "question": q,
                "ground_truth": gt,
                "ground_truth_direction": gt_dir,
                "predicted_direction": pred_dir,
                "is_correct": is_corr,
                "is_valid_json": is_valid_json,
                "latency_ms": round(lat_ms, 2),
            })

        if cold_latency is None:
            cold_latency = lat_ms
        latencies.append(lat_ms)

        if (idx + 1) % 30 == 0 or idx == len(samples) - 1:
            logger.info(f"  Processed [{idx + 1}/{len(samples)}] samples... (last lat: {lat_ms:.1f} ms)")

    # Compute latency statistics
    warm_latencies = latencies[1:] if len(latencies) > 1 else latencies
    mean_lat = float(np.mean(warm_latencies))
    p50_lat = float(np.median(warm_latencies))
    p95_lat = float(np.percentile(warm_latencies, 95))
    peak_vram = int(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0

    total_samples = len(samples)
    total_correct = sum(d["correct"] for d in dataset_stats.values())
    overall_acc = round((total_correct / max(1, total_samples)) * 100.0, 2)

    summary = {
        "model_label": model_label,
        "adapter_path": str(adapter_path),
        "total_samples": total_samples,
        "total_correct": total_correct,
        "overall_accuracy_pct": overall_acc,
        "latency_metrics": {
            "cold_load_ms": cold_load_time_ms,
            "cold_first_sample_ms": round(cold_latency, 2) if cold_latency else 0.0,
            "warm_mean_ms": round(mean_lat, 2),
            "warm_p50_ms": round(p50_lat, 2),
            "warm_p95_ms": round(p95_lat, 2),
            "peak_vram_mb": peak_vram,
        },
        "dataset_breakdown": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy_pct": round((v["correct"] / max(1, v["total"])) * 100.0, 2),
                "valid_json_pct": round((v.get("valid_json", 0) / max(1, v["total"])) * 100.0, 2) if "valid_json" in v else None,
                "mean_latency_ms": round(float(np.mean(v["latencies"])), 2) if v["latencies"] else 0.0,
            }
            for k, v in dataset_stats.items()
        },
        "detailed_records": records,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved evaluation results to {output_json}")
    logger.info(f"OVERALL ACCURACY: {overall_acc}% ({total_correct}/{total_samples})")
    for k, v in summary["dataset_breakdown"].items():
        logger.info(f" - {k}: {v['accuracy_pct']}% ({v['correct']}/{v['total']}) | Latency: {v['mean_latency_ms']} ms")

    return summary


def run_full_gate_comparison():
    logger.info("=" * 80)
    logger.info("RUNNING FINAL GENERALIZATION GATE: GOLDEN M10 VS CANDIDATE A")
    logger.info("=" * 80)

    test_file = REPO_ROOT / "datasets" / "adaptation" / "independent_test_200.json"

    # 1. Evaluate Golden M10 Baseline
    golden_path = REPO_ROOT / "models" / "adapters" / "qwen2_vl_rs_lora_m10_golden"
    golden_res = evaluate_adapter_on_independent_set(
        adapter_path=golden_path,
        model_label="Golden M10 Baseline",
        test_file=test_file,
        output_json=REPO_ROOT / "docs" / "evaluation" / "eval_indep_golden.json",
    )

    # 2. Evaluate Candidate A
    candidate_path = REPO_ROOT / "models" / "adapters" / "experiments" / "qwen_rs_exp_a" / "best_checkpoint"
    cand_res = evaluate_adapter_on_independent_set(
        adapter_path=candidate_path,
        model_label="Candidate A (VRSBench+RSVQA+CDVQA)",
        test_file=test_file,
        output_json=REPO_ROOT / "docs" / "evaluation" / "eval_indep_candidate_a.json",
    )

    # 3. Print Final Head-to-Head Comparison
    print("\n" + "=" * 80)
    print("GENERALIZATION GATE HEAD-TO-HEAD SUMMARY (178 Independent Samples)")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Golden M10':<18} | {'Candidate A':<18} | {'Net Gain':<12}")
    print("-" * 88)
    delta_acc = cand_res['overall_accuracy_pct'] - golden_res['overall_accuracy_pct']
    print(f"{'Overall Accuracy':<35} | {golden_res['overall_accuracy_pct']}% ({golden_res['total_correct']}/{golden_res['total_samples']})   | {cand_res['overall_accuracy_pct']}% ({cand_res['total_correct']}/{cand_res['total_samples']})   | {delta_acc:+.2f}%")

    for ds in ["RSVQA-HR", "CDVQA_Test", "OSCD"]:
        g_acc = golden_res["dataset_breakdown"][ds]["accuracy_pct"]
        c_acc = cand_res["dataset_breakdown"][ds]["accuracy_pct"]
        d = c_acc - g_acc
        print(f"{ds + ' Accuracy':<35} | {g_acc:.2f}%{'':<11} | {c_acc:.2f}%{'':<11} | {d:+.2f}%")

    print("-" * 88)
    print(f"{'Warm Mean Latency':<35} | {golden_res['latency_metrics']['warm_mean_ms']:.1f} ms{'':<10} | {cand_res['latency_metrics']['warm_mean_ms']:.1f} ms{'':<10} | {cand_res['latency_metrics']['warm_mean_ms'] - golden_res['latency_metrics']['warm_mean_ms']:+.1f} ms")
    print(f"{'Warm P95 Latency':<35} | {golden_res['latency_metrics']['warm_p95_ms']:.1f} ms{'':<10} | {cand_res['latency_metrics']['warm_p95_ms']:.1f} ms{'':<10} | {cand_res['latency_metrics']['warm_p95_ms'] - golden_res['latency_metrics']['warm_p95_ms']:+.1f} ms")
    print(f"{'Peak VRAM':<35} | {golden_res['latency_metrics']['peak_vram_mb']} MB{'':<12} | {cand_res['latency_metrics']['peak_vram_mb']} MB{'':<12} | {cand_res['latency_metrics']['peak_vram_mb'] - golden_res['latency_metrics']['peak_vram_mb']:+d} MB")
    print("=" * 80)


if __name__ == "__main__":
    run_full_gate_comparison()
