"""Comprehensive evaluation harness for M10 Remote-Sensing Adaptation.

Compares Base Qwen2-VL-2B against Adapted RS-LoRA model across:
1. Four Semantic Pillars:
   - Pillar A: Single-Image Remote-Sensing VQA
   - Pillar B: Land-Cover & Scene Semantics
   - Pillar C: Remote-Sensing Object Semantics
   - Pillar D: Bi-Temporal Change Semantics
2. Seven RS Land-Cover Categories:
   - Urban, Forest, Agriculture, Water, Infrastructure, Industrial/Mining, Nuisance
3. Canonical Failure Taxonomy (F5, F6, F7, F10, F12)
4. Diagnostic M5 Scenarios (Urban, Forest, Water, Agriculture, Zero-Change)
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.agent.registry import registry
from backend.agent.schema import SpecialistInput, TaskType
from backend.main import register_default_specialists


def evaluate(
    test_json_path: str = "datasets/adaptation/test.json",
    output_report_path: str = "docs/evaluation/m10_baseline_full.json",
    model_mode_label: str = "Unadapted Zero-Shot Baseline",
    use_adapted_flag: bool = False,
) -> Dict[str, Any]:
    print("=" * 75)
    print(f"SatQuery AI M10 Full Evaluation: {model_mode_label}")
    print(f"Flag SATQUERY_USE_ADAPTED_VLM: {'1' if use_adapted_flag else '0'}")
    print("=" * 75)

    os.environ["SATQUERY_USE_ADAPTED_VLM"] = "1" if use_adapted_flag else "0"

    registry.clear()
    register_default_specialists()

    vqa_spec = registry.get("RS_VQA")
    change_vqa_spec = registry.get("CHANGE_VQA")
    cd_spec = registry.get("CHANGE_DETECT")

    assert vqa_spec is not None, "RS_VQA specialist not found"
    assert change_vqa_spec is not None, "CHANGE_VQA specialist not found"
    assert cd_spec is not None, "CHANGE_DETECT specialist not found"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    print(f"Loaded {len(test_samples)} held-out test samples from {test_json_path}")

    # Track metrics per pillar and category
    pillar_totals = Counter()
    pillar_correct = Counter()
    category_totals = Counter()
    category_correct = Counter()
    failure_counts = Counter()

    vqa_latencies = []
    change_latencies = []
    sample_records = []

    valid_json_count = 0
    total_change_samples = 0

    for idx, sample in enumerate(test_samples):
        task = sample.get("task", "RS_VQA")
        pillar = sample.get("pillar", "A_RS_VQA")
        parent_scene = sample.get("parent_scene", "unknown")
        
        # Categorize land-cover domain
        cat = "other"
        for candidate in ["urban", "forest", "agri", "water", "infra", "coastal", "mining", "nuisance", "industrial"]:
            if candidate in parent_scene.lower() or candidate in sample.get("sample_id", "").lower() or candidate in sample.get("question", "").lower():
                cat = candidate
                break
        
        pillar_totals[pillar] += 1
        category_totals[cat] += 1

        if task == "RS_VQA":
            spec_in = SpecialistInput(
                task=TaskType.VQA,
                query=sample["question"],
                primary_image_path=sample["image_path"],
            )
            t0 = time.perf_counter()
            out = vqa_spec.predict(spec_in)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            vqa_latencies.append(lat_ms)

            pred_text = (out.answer_text or "").strip()
            gt_text = str(sample.get("ground_truth", "")).strip().lower()

            is_match = (
                gt_text in pred_text.lower()
                or pred_text.lower() in gt_text
                or any(w in pred_text.lower() for w in gt_text.split() if len(w) > 4)
            )

            if is_match:
                pillar_correct[pillar] += 1
                category_correct[cat] += 1
            else:
                failure_counts["F6_semantic_interpretation"] += 1

            sample_records.append({
                "sample_id": sample["sample_id"],
                "pillar": pillar,
                "category": cat,
                "parent_scene": parent_scene,
                "task": task,
                "question": sample["question"],
                "ground_truth": gt_text,
                "prediction": pred_text,
                "is_correct": is_match,
                "latency_ms": round(lat_ms, 2),
            })

        elif task == "CHANGE_VQA":
            total_change_samples += 1
            t1_p = sample["primary_image_path"]
            t2_p = sample["secondary_image_path"]
            query = sample["question"]

            # Run CHANGE_DETECT to get spatial evidence
            try:
                cd_in = SpecialistInput(
                    task=TaskType.CHANGE_DETECTION,
                    query=query,
                    primary_image_path=t1_p,
                    secondary_image_path=t2_p,
                    parameters={"candidate": "cva"},
                )
                cd_out = cd_spec.predict(cd_in)
                vqa_params = dict(cd_out.parameters_used)
                vqa_params["regions"] = [r.model_dump() for r in cd_out.evidence.regions]
                vqa_params["statistics"] = [s.model_dump() for s in cd_out.evidence.statistics]
                vqa_params["overlay_path"] = cd_out.parameters_used.get("overlay_path")
                vqa_params["mask_path"] = cd_out.parameters_used.get("mask_path")
            except Exception as cd_err:
                # Benchmark PNG format support (Rule 7) for unprojected datasets like LEVIR-CD
                from PIL import Image
                import numpy as np
                from scipy.ndimage import label as nd_label
                from backend.models.change import DeterministicCVASpecialist

                try:
                    arr1 = np.array(Image.open(t1_p).convert("RGB")).transpose(2, 0, 1)
                    arr2 = np.array(Image.open(t2_p).convert("RGB")).transpose(2, 0, 1)
                    cva = DeterministicCVASpecialist()
                    b_mask, _, _ = cva.predict(arr1, arr2)
                    changed_px = int(np.count_nonzero(b_mask > 0))
                    _, num_features = nd_label(b_mask > 0)
                except Exception as img_err:
                    changed_px = 0
                    num_features = 0
                    b_mask = np.zeros((1, 1))

                vqa_params = {
                    "changed_pixels": changed_px,
                    "change_ratio_pct": round((changed_px / max(b_mask.size, 1)) * 100.0, 4),
                    "total_clusters": int(num_features),
                    "regions": [{"region_id": i + 1, "label": f"cluster_{i+1}"} for i in range(min(num_features, 10))],
                }

            vqa_in = SpecialistInput(
                task=TaskType.CHANGE_VQA,
                query=query,
                primary_image_path=t1_p,
                secondary_image_path=t2_p,
                parameters=vqa_params,
            )

            t0 = time.perf_counter()
            vqa_out = change_vqa_spec.predict(vqa_in)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            change_latencies.append(lat_ms)

            interp = vqa_out.evidence.semantic_interpretation
            has_valid_json = interp is not None and bool(interp.temporal_direction)
            if has_valid_json:
                valid_json_count += 1

            gt_json = sample.get("ground_truth_json", {})
            gt_dir = gt_json.get("temporal_direction")
            pred_dir = interp.temporal_direction if interp else None

            direction_match = (pred_dir == gt_dir)
            if direction_match:
                pillar_correct[pillar] += 1
                category_correct[cat] += 1
            else:
                if vqa_params.get("changed_pixels", 0) == 0 and gt_dir != "no_change":
                    failure_counts["F5_spatial_detection_miss"] += 1
                else:
                    failure_counts["F6_semantic_interpretation"] += 1

            sample_records.append({
                "sample_id": sample["sample_id"],
                "pillar": pillar,
                "category": cat,
                "parent_scene": parent_scene,
                "task": task,
                "question": query,
                "ground_truth_direction": gt_dir,
                "predicted_direction": pred_dir,
                "is_correct": direction_match,
                "is_valid_json": has_valid_json,
                "latency_ms": round(lat_ms, 2),
            })

        if (idx + 1) % 15 == 0 or idx == len(test_samples) - 1:
            print(f"  Processed [{idx + 1}/{len(test_samples)}] evaluation samples...")

    # Diagnostic M5 Fixtures run
    from backend.evaluation.semantic_fixtures import generate_semantic_fixtures
    fixtures = generate_semantic_fixtures()
    m5_diagnostics = []
    print("\nRunning Diagnostic M5 Fixtures...")
    for f in fixtures:
        cd_in = SpecialistInput(
            task=TaskType.CHANGE_DETECTION,
            query=f.query,
            primary_image_path=str(f.t1_path),
            secondary_image_path=str(f.t2_path),
            parameters={"candidate": f.candidate_detector},
        )
        cd_out = cd_spec.predict(cd_in)

        vqa_params = dict(cd_out.parameters_used)
        vqa_params["regions"] = [r.model_dump() for r in cd_out.evidence.regions]
        vqa_params["statistics"] = [s.model_dump() for s in cd_out.evidence.statistics]
        vqa_params["overlay_path"] = cd_out.parameters_used.get("overlay_path")
        vqa_params["mask_path"] = cd_out.parameters_used.get("mask_path")

        vqa_in = SpecialistInput(
            task=TaskType.CHANGE_VQA,
            query=f.query,
            primary_image_path=str(f.t1_path),
            secondary_image_path=str(f.t2_path),
            parameters=vqa_params,
        )
        t0 = time.perf_counter()
        vqa_out = change_vqa_spec.predict(vqa_in)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        interp = vqa_out.evidence.semantic_interpretation

        pred_from = interp.transitions[0].from_class if (interp and interp.transitions) else "none"
        pred_to = interp.transitions[0].to_class if (interp and interp.transitions) else "none"

        m5_diagnostics.append({
            "scenario_id": f.scenario_id,
            "category": f.category,
            "expected_transition": f"{f.expected_from_class} -> {f.expected_to_class}",
            "predicted_transition": f"{pred_from} -> {pred_to}",
            "expected_direction": f.expected_direction,
            "predicted_direction": interp.temporal_direction if interp else "none",
            "semantic_uncertainty": interp.semantic_uncertainty if interp else 1.0,
            "latency_ms": round(lat_ms, 2),
        })
        print(f"  [{f.scenario_id}] Exp: {f.expected_from_class}->{f.expected_to_class} | Pred: {pred_from}->{pred_to} ({interp.temporal_direction if interp else 'none'})")

    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

    total_correct = sum(pillar_correct.values())
    overall_accuracy = (total_correct / len(test_samples)) * 100.0 if test_samples else 0.0

    pillar_summary = {}
    for p, tot in pillar_totals.items():
        corr = pillar_correct.get(p, 0)
        pillar_summary[p] = {
            "total": tot,
            "correct": corr,
            "accuracy_pct": round((corr / tot) * 100.0, 2) if tot else 0.0,
        }

    category_summary = {}
    for c, tot in category_totals.items():
        corr = category_correct.get(c, 0)
        category_summary[c] = {
            "total": tot,
            "correct": corr,
            "accuracy_pct": round((corr / tot) * 100.0, 2) if tot else 0.0,
        }

    report = {
        "model_mode": model_mode_label,
        "use_adapted_flag": use_adapted_flag,
        "peak_vram_mb": peak_vram_mb,
        "overall_metrics": {
            "total_samples": len(test_samples),
            "correct_count": total_correct,
            "accuracy_pct": round(overall_accuracy, 2),
            "mean_vqa_latency_ms": round(float(torch.tensor(vqa_latencies).mean().item()), 2) if vqa_latencies else 0.0,
            "mean_change_latency_ms": round(float(torch.tensor(change_latencies).mean().item()), 2) if change_latencies else 0.0,
        },
        "change_vqa_metrics": {
            "total_samples": total_change_samples,
            "valid_json_count": valid_json_count,
            "valid_json_rate_pct": round((valid_json_count / total_change_samples) * 100.0, 2) if total_change_samples else 0.0,
        },
        "pillar_breakdown": pillar_summary,
        "category_breakdown": category_summary,
        "failure_taxonomy": dict(failure_counts),
        "m5_diagnostics": m5_diagnostics,
        "sample_records": sample_records,
    }

    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print(f"Evaluation Complete: {model_mode_label}")
    print(f"Overall Accuracy: {report['overall_metrics']['accuracy_pct']}% ({total_correct}/{len(test_samples)})")
    print(f"Pillars: {json.dumps(pillar_summary, indent=2)}")
    print(f"Categories: {json.dumps(category_summary, indent=2)}")
    print(f"CHANGE_VQA JSON Validity: {report['change_vqa_metrics']['valid_json_rate_pct']}%")
    print(f"Peak VRAM: {peak_vram_mb} MB")
    print(f"Report saved to: {output_report_path}")
    print("=" * 75)
    return report


if __name__ == "__main__":
    is_adapted = "--adapted" in sys.argv or os.environ.get("SATQUERY_USE_ADAPTED_VLM", "0") == "1"
    label = "Adapted RS-LoRA Candidate" if is_adapted else "Unadapted Zero-Shot Baseline"
    out_file = (
        "docs/evaluation/m10_adapted_full.json"
        if is_adapted
        else "docs/evaluation/m10_baseline_full.json"
    )
    evaluate(output_report_path=out_file, model_mode_label=label, use_adapted_flag=is_adapted)
