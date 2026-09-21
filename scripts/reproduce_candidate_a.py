"""Script to reproduce Candidate A evaluation on frozen 64-sample test set."""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_adaptation import evaluate

if __name__ == "__main__":
    res = evaluate(
        test_json_path="datasets/adaptation/test.json",
        output_report_path="docs/evaluation/eval_candidate_a_reproduced.json",
        model_mode_label="Candidate A Best Checkpoint Reproduction",
        use_adapted_flag=True,
        adapter_path="models/adapters/experiments/qwen_rs_exp_a/best_checkpoint",
    )
    print("=" * 60)
    print("CANDIDATE A REPRODUCTION SUMMARY:")
    print(f"Overall Accuracy: {res['overall_metrics']['accuracy_pct']}% ({res['overall_metrics']['correct_count']}/{res['overall_metrics']['total_samples']})")
    print(f"JSON Validity: {res['change_vqa_metrics']['valid_json_rate_pct']}% ({res['change_vqa_metrics']['valid_json_count']}/{res['change_vqa_metrics']['total_samples']})")
    for pillar, stats in res['pillar_breakdown'].items():
        print(f" - {pillar}: {stats['accuracy_pct']}% ({stats['correct']}/{stats['total']})")
    print("=" * 60)
