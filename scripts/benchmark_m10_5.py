"""M10.5 Efficiency Optimization & Runtime Benchmark Harness.

Measures accuracy, latency (mean, median, p95), peak VRAM, breakdown (preprocessing,
inference, total request), and compatibility (JSON, M8, M9, M11) across the frozen
64-sample test set (datasets/adaptation/test.json).

Preserves existing M10 golden evaluation results and allows controlled, reproducible
A/B benchmarking of optimization candidates.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from backend.agent.registry import registry
from backend.agent.schema import (
    ConfidenceBreakdown,
    ConfidenceLevel,
    EvidenceBundle,
    SpecialistInput,
    StandardResultContract,
    TaskType,
)
from backend.main import register_default_specialists
from backend.reports import ReportBuilder


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Computes mean, median, and p95 for a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    arr = np.array(values)
    return {
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }


def run_benchmark(
    candidate_label: str = "Reference M10 (Untouched)",
    test_json_path: str = "datasets/adaptation/test.json",
    output_path: str = "docs/evaluation/m10_5_reference_benchmark.json",
    use_adapted_vlm: bool = True,
    baseline_reference_path: Optional[str] = None,
) -> Dict[str, Any]:
    print("=" * 80)
    print(f"SATQUERY AI M10.5 BENCHMARK: {candidate_label}")
    print(f"Test Set: {test_json_path} (Frozen)")
    print(f"Adapter Active (SATQUERY_USE_ADAPTED_VLM): {'1' if use_adapted_vlm else '0'}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB)")
    print("=" * 80)

    os.environ["SATQUERY_USE_ADAPTED_VLM"] = "1" if use_adapted_vlm else "0"

    # Reset registry, shared VLM runtime, and memory
    from backend.models.shared_vlm import cleanup_shared_vlm
    cleanup_shared_vlm()
    registry.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Measure cold-start load time
    t_load_start = time.perf_counter()
    register_default_specialists()

    vqa_spec = registry.get("RS_VQA")
    change_vqa_spec = registry.get("CHANGE_VQA")
    cd_spec = registry.get("CHANGE_DETECT")

    assert vqa_spec is not None, "RS_VQA specialist not registered"
    assert change_vqa_spec is not None, "CHANGE_VQA specialist not registered"
    assert cd_spec is not None, "CHANGE_DETECT specialist not registered"

    # Trigger explicit load to measure cold-start
    vqa_spec.load()
    change_vqa_spec.load()
    cold_load_time_ms = round((time.perf_counter() - t_load_start) * 1000.0, 2)
    print(f"Cold-Start Model Load Time: {cold_load_time_ms:.2f} ms")

    # Instrument model.generate to separate preprocessing vs inference vs total request latency
    timing_context = {"t_predict_start": 0.0, "prep_ms": 0.0, "infer_ms": 0.0}

    def make_timing_wrapper(orig_gen):
        def timed_gen(*args, **kwargs):
            t_gen_start = time.perf_counter()
            timing_context["prep_ms"] = max(0.0, (t_gen_start - timing_context["t_predict_start"]) * 1000.0)
            res = orig_gen(*args, **kwargs)
            timing_context["infer_ms"] = (time.perf_counter() - t_gen_start) * 1000.0
            return res
        return timed_gen

    if hasattr(vqa_spec.model, "generate"):
        vqa_spec.model.generate = make_timing_wrapper(vqa_spec.model.generate)
    if hasattr(change_vqa_spec.model, "generate"):
        change_vqa_spec.model.generate = make_timing_wrapper(change_vqa_spec.model.generate)

    # Load frozen test set
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)
    print(f"Loaded {len(test_samples)} frozen test samples.")

    pillar_totals = Counter()
    pillar_correct = Counter()
    category_totals = Counter()
    category_correct = Counter()
    failure_counts = Counter()

    preprocessing_latencies = []
    inference_latencies = []
    total_latencies = []
    vqa_latencies = []
    change_latencies = []
    sample_records = []

    valid_json_count = 0
    total_change_samples = 0
    m8_compat_count = 0
    m9_compat_count = 0
    m11_compat_count = 0

    for idx, sample in enumerate(test_samples):
        task = sample.get("task", "RS_VQA")
        pillar = sample.get("pillar", "A_RS_VQA")
        parent_scene = sample.get("parent_scene", "unknown")

        cat = "other"
        for candidate in ["urban", "forest", "agri", "water", "infra", "coastal", "mining", "nuisance", "industrial"]:
            if candidate in parent_scene.lower() or candidate in sample.get("sample_id", "").lower() or candidate in sample.get("question", "").lower():
                cat = candidate
                break

        pillar_totals[pillar] += 1
        category_totals[cat] += 1

        timing_context["prep_ms"] = 0.0
        timing_context["infer_ms"] = 0.0

        if task == "RS_VQA":
            spec_in = SpecialistInput(
                task=TaskType.VQA,
                query=sample["question"],
                primary_image_path=sample["image_path"],
            )

            t0 = time.perf_counter()
            timing_context["t_predict_start"] = t0
            out = vqa_spec.predict(spec_in)
            total_req_ms = (time.perf_counter() - t0) * 1000.0

            prep_ms = timing_context["prep_ms"]
            infer_ms = timing_context["infer_ms"]
            # Fallback if uninstrumented
            if infer_ms == 0.0:
                infer_ms = total_req_ms

            preprocessing_latencies.append(prep_ms)
            inference_latencies.append(infer_ms)
            total_latencies.append(total_req_ms)
            vqa_latencies.append(total_req_ms)

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

            # Check compatibility
            if out.evidence is not None:
                m8_compat_count += 1
            if 0.0 <= out.confidence <= 1.0:
                m9_compat_count += 1

            # Test M11 report generation compatibility
            try:
                conf_bd = ConfidenceBreakdown(
                    input_quality_score=1.0,
                    spatial_alignment_score=1.0,
                    model_confidence_score=out.confidence,
                    overall_confidence=out.confidence,
                    specialist_confidence=out.confidence,
                    confidence_level=ConfidenceLevel.HIGH if out.confidence >= 0.8 else ConfidenceLevel.MEDIUM,
                )
                contract = StandardResultContract(
                    task="RS_VQA",
                    answer=pred_text,
                    confidence=out.confidence,
                    confidence_breakdown=conf_bd,
                    evidence=out.evidence or EvidenceBundle(),
                    parameters=out.parameters_used,
                    warnings=out.warnings,
                    execution_time_ms=int(total_req_ms),
                )
                rep = ReportBuilder.build(contract, query=sample["question"], primary_path=sample["image_path"])
                if rep and rep.metadata and rep.metadata.report_id:
                    m11_compat_count += 1
            except Exception as e:
                pass

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
                "prep_ms": round(prep_ms, 2),
                "infer_ms": round(infer_ms, 2),
                "total_req_ms": round(total_req_ms, 2),
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
            except Exception:
                from PIL import Image
                from scipy.ndimage import label as nd_label
                from backend.models.change import DeterministicCVASpecialist

                try:
                    arr1 = np.array(Image.open(t1_p).convert("RGB")).transpose(2, 0, 1)
                    arr2 = np.array(Image.open(t2_p).convert("RGB")).transpose(2, 0, 1)
                    cva = DeterministicCVASpecialist()
                    b_mask, _, _ = cva.predict(arr1, arr2)
                    changed_px = int(np.count_nonzero(b_mask > 0))
                    _, num_features = nd_label(b_mask > 0)
                except Exception:
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
            timing_context["t_predict_start"] = t0
            vqa_out = change_vqa_spec.predict(vqa_in)
            total_req_ms = (time.perf_counter() - t0) * 1000.0

            prep_ms = timing_context["prep_ms"]
            infer_ms = timing_context["infer_ms"]
            if infer_ms == 0.0:
                infer_ms = total_req_ms

            preprocessing_latencies.append(prep_ms)
            inference_latencies.append(infer_ms)
            total_latencies.append(total_req_ms)
            change_latencies.append(total_req_ms)

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

            if vqa_out.evidence is not None:
                m8_compat_count += 1
            if 0.0 <= vqa_out.confidence <= 1.0:
                m9_compat_count += 1

            # Test M11 report generation compatibility
            try:
                conf_bd = ConfidenceBreakdown(
                    input_quality_score=1.0,
                    spatial_alignment_score=1.0,
                    model_confidence_score=vqa_out.confidence,
                    overall_confidence=vqa_out.confidence,
                    specialist_confidence=vqa_out.confidence,
                    confidence_level=ConfidenceLevel.HIGH if vqa_out.confidence >= 0.8 else ConfidenceLevel.MEDIUM,
                )
                contract = StandardResultContract(
                    task="CHANGE_VQA",
                    answer=vqa_out.answer_text or "",
                    confidence=vqa_out.confidence,
                    confidence_breakdown=conf_bd,
                    evidence=vqa_out.evidence or EvidenceBundle(),
                    parameters=vqa_out.parameters_used,
                    warnings=vqa_out.warnings,
                    execution_time_ms=int(total_req_ms),
                )
                rep = ReportBuilder.build(contract, query=query, primary_path=t1_p, secondary_path=t2_p)
                if rep and rep.metadata and rep.metadata.report_id:
                    m11_compat_count += 1
            except Exception as e:
                pass

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
                "prep_ms": round(prep_ms, 2),
                "infer_ms": round(infer_ms, 2),
                "total_req_ms": round(total_req_ms, 2),
            })

        if (idx + 1) % 16 == 0 or idx == len(test_samples) - 1:
            print(f"  Processed [{idx + 1}/{len(test_samples)}] evaluation samples...")

    # Peak VRAM measurement
    peak_vram_mb = 0
    if torch.cuda.is_available():
        peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

    total_samples = len(test_samples)
    total_correct = sum(pillar_correct.values())
    overall_acc = round((total_correct / total_samples) * 100.0, 2)
    json_rate = round((valid_json_count / max(total_change_samples, 1)) * 100.0, 2)

    prep_stats = compute_percentiles(preprocessing_latencies)
    infer_stats = compute_percentiles(inference_latencies)
    total_stats = compute_percentiles(total_latencies)
    vqa_stats = compute_percentiles(vqa_latencies)
    change_stats = compute_percentiles(change_latencies)

    # Compute failure deltas against baseline if provided
    failure_deltas = {"newly_introduced": [], "newly_fixed": [], "unchanged_failures": []}
    if baseline_reference_path and Path(baseline_reference_path).exists():
        with open(baseline_reference_path, "r", encoding="utf-8") as bf:
            base_data = json.load(bf)
        base_samples = {s["sample_id"]: s for s in base_data.get("sample_records", [])}
        for curr in sample_records:
            sid = curr["sample_id"]
            if sid in base_samples:
                base_c = base_samples[sid]["is_correct"]
                curr_c = curr["is_correct"]
                if not base_c and curr_c:
                    failure_deltas["newly_fixed"].append(sid)
                elif base_c and not curr_c:
                    failure_deltas["newly_introduced"].append(sid)
                elif not base_c and not curr_c:
                    failure_deltas["unchanged_failures"].append(sid)

    # Format pillar breakdown (A, B, C, D)
    pillar_order = ["A_RS_VQA", "B_SCENE_SEMANTICS", "C_OBJECT_SEMANTICS", "D_CHANGE_SEMANTICS"]
    pillar_breakdown = {}
    for p_name in pillar_order:
        tot = pillar_totals[p_name]
        corr = pillar_correct[p_name]
        pillar_breakdown[p_name] = {
            "total": tot,
            "correct": corr,
            "accuracy_pct": round((corr / tot) * 100.0, 2) if tot else 0.0,
        }

    results = {
        "candidate_label": candidate_label,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cold_load_time_ms": cold_load_time_ms,
        "peak_vram_mb": peak_vram_mb,
        "overall_metrics": {
            "total_samples": total_samples,
            "correct_count": total_correct,
            "accuracy_pct": overall_acc,
            "preprocessing_latencies": prep_stats,
            "inference_latencies": infer_stats,
            "total_latencies": total_stats,
            "vqa_latencies": vqa_stats,
            "change_latencies": change_stats,
        },
        "change_vqa_metrics": {
            "total_samples": total_change_samples,
            "valid_json_count": valid_json_count,
            "valid_json_rate_pct": json_rate,
        },
        "compatibility": {
            "json_validity_pct": json_rate,
            "m8_evidence_bundle_pct": round((m8_compat_count / total_samples) * 100.0, 2),
            "m9_confidence_breakdown_pct": round((m9_compat_count / total_samples) * 100.0, 2),
            "m11_report_generation_pct": round((m11_compat_count / total_samples) * 100.0, 2),
        },
        "pillar_breakdown": pillar_breakdown,
        "failure_counts": dict(failure_counts),
        "failure_deltas": failure_deltas,
        "sample_records": sample_records,
    }

    # Save output
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("M10.5 BENCHMARK RESULTS SUMMARY:")
    print(f"  Candidate: {candidate_label}")
    print(f"  Accuracy: Overall={overall_acc}% ({total_correct}/{total_samples})")
    print(f"    - Pillar A (RS_VQA): {pillar_breakdown['A_RS_VQA']['accuracy_pct']}% ({pillar_breakdown['A_RS_VQA']['correct']}/{pillar_breakdown['A_RS_VQA']['total']})")
    print(f"    - Pillar B (SCENE): {pillar_breakdown['B_SCENE_SEMANTICS']['accuracy_pct']}% ({pillar_breakdown['B_SCENE_SEMANTICS']['correct']}/{pillar_breakdown['B_SCENE_SEMANTICS']['total']})")
    print(f"    - Pillar C (OBJECT): {pillar_breakdown['C_OBJECT_SEMANTICS']['accuracy_pct']}% ({pillar_breakdown['C_OBJECT_SEMANTICS']['correct']}/{pillar_breakdown['C_OBJECT_SEMANTICS']['total']})")
    print(f"    - Pillar D (CHANGE): {pillar_breakdown['D_CHANGE_SEMANTICS']['accuracy_pct']}% ({pillar_breakdown['D_CHANGE_SEMANTICS']['correct']}/{pillar_breakdown['D_CHANGE_SEMANTICS']['total']})")
    print(f"  Runtime Breakdown:")
    print(f"    - Model Load (cold): {cold_load_time_ms:.2f} ms")
    print(f"    - Preprocessing (ms): Mean={prep_stats['mean']} | Median={prep_stats['median']} | P95={prep_stats['p95']}")
    print(f"    - Inference (ms):     Mean={infer_stats['mean']} | Median={infer_stats['median']} | P95={infer_stats['p95']}")
    print(f"    - Total Request (ms): Mean={total_stats['mean']} | Median={total_stats['median']} | P95={total_stats['p95']}")
    print(f"  Memory: Peak VRAM = {peak_vram_mb} MB")
    print(f"  Compatibility:")
    print(f"    - JSON validity: {json_rate}%")
    print(f"    - M8 Evidence:   {results['compatibility']['m8_evidence_bundle_pct']}%")
    print(f"    - M9 Confidence: {results['compatibility']['m9_confidence_breakdown_pct']}%")
    print(f"    - M11 Reporting: {results['compatibility']['m11_report_generation_pct']}%")
    if baseline_reference_path:
        print(f"  Failure Deltas vs Baseline: +{len(failure_deltas['newly_fixed'])} fixed, -{len(failure_deltas['newly_introduced'])} introduced, {len(failure_deltas['unchanged_failures'])} unchanged")
    print(f"  Saved benchmark report to: {out_p}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M10.5 Benchmark Harness")
    parser.add_argument("--candidate", type=str, default="Reference M10 (Untouched)", help="Label for this benchmark candidate")
    parser.add_argument("--output", type=str, default="docs/evaluation/m10_5_reference_benchmark.json", help="Output JSON path")
    parser.add_argument("--baseline", type=str, default=None, help="Path to reference benchmark JSON for delta comparison")
    parser.add_argument("--unadapted", action="store_true", help="Evaluate base unadapted model instead of RS-LoRA")
    args = parser.parse_args()

    run_benchmark(
        candidate_label=args.candidate,
        output_path=args.output,
        use_adapted_vlm=not args.unadapted,
        baseline_reference_path=args.baseline,
    )
