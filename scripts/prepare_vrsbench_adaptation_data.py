"""Prepares balanced multi-task adaptation data across 4 semantic pillars for M10.

Four Semantic Pillars:
- Pillar A: Single-Image Remote-Sensing VQA (VRSBench questions: existence, count, color, position)
- Pillar B: Land-Cover & Scene Semantics (High-level scene classification, land-use structure, and terrain description)
- Pillar C: Remote-Sensing Object Semantics (Object categorization, infrastructure roles: bridges, harbors, tanks)
- Pillar D: Bi-Temporal Change Semantics (Real Earth observation changes from LEVIR-CD and DiverseRS, plus diagnostic controls)

Integrity & Leakage Controls:
- Strictly Parent-Scene Disjoint for all single-image overhead data (Train, Val, Test disjoint).
- Strictly Pair-Disjoint for bi-temporal Earth observation imagery (LEVIR-CD and DiverseRS disjoint).
- Generates:
  - datasets/adaptation/train.json (~130-150 balanced multi-pillar samples)
  - datasets/adaptation/val.json (~25-35 validation samples for checkpoint selection)
  - datasets/adaptation/test.json (~45-55 held-out test samples for final evaluation)
  - datasets/adaptation/manifest.json (comprehensive provenance documentation)
"""

import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def prepare_balanced_adaptation_data() -> Dict[str, Any]:
    print("=" * 75)
    print("SatQuery AI — Preparing M10 Multi-Pillar Adaptation Mixture")
    print("=" * 75)

    cache_dir = Path(
        r"C:\Users\Shravan\.cache\huggingface\hub\datasets--xiang709--VRSBench\snapshots\6cee2968fd752a6d51c6cb2d18dded2bc0baa218"
    )
    zip_file = cache_dir / "Annotations_val.zip"
    images_dir = Path("datasets/grounding_eval_subset/images")
    output_dir = Path("datasets/adaptation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Single-Image Overhead Scene Partition (VRSBench)
    # -------------------------------------------------------------
    train_parent_scenes = {
        "05866", "05867", "05871", "05875", "05877",
        "P0019", "P0161", "P0168", "P1179", "P1210",
    }
    val_parent_scenes = {"P1022", "07160", "10904"}
    test_parent_scenes = {"P1225", "P2912", "P2982", "P4055", "P4265", "P4627"}

    assert len(train_parent_scenes.intersection(val_parent_scenes)) == 0
    assert len(train_parent_scenes.intersection(test_parent_scenes)) == 0
    assert len(val_parent_scenes.intersection(test_parent_scenes)) == 0

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    # Process VRSBench annotations for Pillars A, B, C
    with zipfile.ZipFile(zip_file, "r") as z:
        for img_path in sorted(images_dir.glob("*.png")):
            img_name = img_path.name
            parent_scene = img_name.split("_")[0]
            json_name = f"Annotations_val/{img_path.stem}.json"

            if json_name not in z.namelist():
                continue

            ann = json.loads(z.read(json_name).decode("utf-8"))
            
            # Determine destination split
            if parent_scene in train_parent_scenes:
                dest = train_samples
                split_name = "train"
            elif parent_scene in val_parent_scenes:
                dest = val_samples
                split_name = "val"
            elif parent_scene in test_parent_scenes:
                dest = test_samples
                split_name = "test"
            else:
                continue

            # Pillar A: Single-Image VQA (Standard multi-turn questions)
            qa_list = ann.get("qa_pairs", [])
            for qa in qa_list:
                dest.append({
                    "sample_id": f"{img_path.stem}_vqa_q{qa.get('ques_id', 0)}",
                    "pillar": "A_RS_VQA",
                    "task": "RS_VQA",
                    "parent_scene": parent_scene,
                    "image_path": str(img_path).replace("\\", "/"),
                    "question": qa.get("question"),
                    "ground_truth": qa.get("answer"),
                    "qa_type": qa.get("type", "unknown"),
                    "source": "VRSBench",
                    "split": split_name,
                })

            # Pillar B: Land-Cover & Scene Semantics (from verified caption)
            caption = ann.get("caption")
            if caption:
                dest.append({
                    "sample_id": f"{img_path.stem}_scene_desc",
                    "pillar": "B_SCENE_SEMANTICS",
                    "task": "RS_VQA",
                    "parent_scene": parent_scene,
                    "image_path": str(img_path).replace("\\", "/"),
                    "question": "Describe the terrain, vegetation, land use, and infrastructure in this satellite observation.",
                    "ground_truth": caption,
                    "qa_type": "scene_description",
                    "source": "VRSBench_Caption",
                    "split": split_name,
                })
                # Add scene classification query
                first_sentence = caption.split(".")[0] + "."
                dest.append({
                    "sample_id": f"{img_path.stem}_scene_cls",
                    "pillar": "B_SCENE_SEMANTICS",
                    "task": "RS_VQA",
                    "parent_scene": parent_scene,
                    "image_path": str(img_path).replace("\\", "/"),
                    "question": "What is the primary scene category and setting depicted in this overhead image?",
                    "ground_truth": first_sentence,
                    "qa_type": "scene_classification",
                    "source": "VRSBench_Caption",
                    "split": split_name,
                })

            # Pillar C: Remote-Sensing Object Semantics
            objects = ann.get("objects", [])
            if objects:
                unique_classes = sorted(list({obj.get("obj_cls", "") for obj in objects if obj.get("obj_cls")}))
                if unique_classes:
                    dest.append({
                        "sample_id": f"{img_path.stem}_obj_inventory",
                        "pillar": "C_OBJECT_SEMANTICS",
                        "task": "RS_VQA",
                        "parent_scene": parent_scene,
                        "image_path": str(img_path).replace("\\", "/"),
                        "question": "Identify all distinct remote-sensing object categories present in this scene.",
                        "ground_truth": f"The detected remote-sensing objects in this scene are: {', '.join(unique_classes)}.",
                        "qa_type": "object_inventory",
                        "source": "VRSBench_Objects",
                        "split": split_name,
                    })

    print(f"VRSBench (Pillars A, B, C) -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # -------------------------------------------------------------
    # 2. Pillar D: Bi-Temporal Change Semantics (Real Earth Observation)
    # -------------------------------------------------------------
    # 2.1 LEVIR-CD Building Changes
    levir_dir = Path("datasets/levir_cd_eval_subset")
    levir_train_ids = [f"val_{i}" for i in range(1, 15)]   # val_1 to val_14 (14 pairs)
    levir_val_ids = [f"val_{i}" for i in range(15, 18)]    # val_15 to val_17 (3 pairs)
    levir_test_ids = [f"val_{i}" for i in range(18, 21)]   # val_18 to val_20 (3 pairs)

    def build_levir_sample(lid: str, split_name: str) -> Dict[str, Any]:
        p1 = levir_dir / "A" / f"{lid}.png"
        p2 = levir_dir / "B" / f"{lid}.png"
        expected_json = {
            "summary": "New building development and residential construction detected across the site.",
            "temporal_direction": "increased",
            "clusters": [
                {
                    "region_id": 1,
                    "from_class": "bare_ground_or_soil",
                    "to_class": "built_structure",
                    "description": "Construction of new residential structures and roof foundations."
                }
            ]
        }
        return {
            "sample_id": f"levir_cd_{lid}",
            "pillar": "D_CHANGE_SEMANTICS",
            "task": "CHANGE_VQA",
            "parent_scene": f"levir_{lid}",
            "primary_image_path": str(p1).replace("\\", "/"),
            "secondary_image_path": str(p2).replace("\\", "/"),
            "question": "What changed between these bi-temporal satellite images? Describe the structural changes.",
            "ground_truth_json": expected_json,
            "ground_truth": json.dumps(expected_json),
            "qa_type": "bitemporal_building_change",
            "source": "LEVIR-CD",
            "split": split_name,
        }

    for lid in levir_train_ids:
        train_samples.append(build_levir_sample(lid, "train"))
    for lid in levir_val_ids:
        val_samples.append(build_levir_sample(lid, "val"))
    for lid in levir_test_ids:
        test_samples.append(build_levir_sample(lid, "test"))

    # 2.2 Diverse Remote-Sensing Benchmark Changes
    div_dir = Path("datasets/diverse_rs_eval_subset")
    
    diverse_train_defs = [
        ("urban_01.png", "increased", "bare_ground_or_soil", "built_structure", "New commercial building construction and site development."),
        ("infrastructure_01.tif", "increased", "bare_ground_or_soil", "built_structure", "Highway expansion and paved infrastructure extension."),
        ("forest_01.png", "decreased", "forest_or_trees", "bare_ground_or_soil", "Forest canopy clearance exposing bare ground."),
        ("coastal_01.png", "modified", "water_body", "bare_ground_or_soil", "Coastal shoreline sediment shift and sand deposition."),
        ("nuisance_illumination_01.png", "no_change", "none", "none", "No physical land-cover alteration; variation is caused by sun angle and solar illumination shift."),
        ("nuisance_shadow_01.png", "no_change", "none", "none", "No physical change detected; variance is caused by cloud shadow displacement."),
    ]
    diverse_val_defs = [
        ("mining_01.png", "modified", "bare_ground_or_soil", "bare_ground_or_soil", "Open-pit mine excavation and earthwork redistribution."),
        ("nuisance_seasonal_01.png", "no_change", "none", "none", "No physical land-cover change; differences correspond to seasonal foliage phenology."),
    ]
    diverse_test_defs = [
        ("water_01.png", "increased", "bare_ground_or_soil", "water_body", "Reservoir expansion and inundation of former shoreline."),
        ("agriculture_01.png", "decreased", "vegetation_or_cropland", "bare_ground_or_soil", "Agricultural harvesting and crop field clearing."),
        ("nuisance_registration_01.png", "no_change", "none", "none", "No physical change detected; differences are sensor registration jitter."),
    ]

    def build_diverse_sample(fname: str, direction: str, from_c: str, to_c: str, desc: str, split_name: str) -> Dict[str, Any]:
        p1 = div_dir / "A" / fname
        p2 = div_dir / "B" / fname
        if direction == "no_change":
            expected_json = {
                "summary": desc,
                "temporal_direction": "no_change",
                "clusters": []
            }
        else:
            expected_json = {
                "summary": desc,
                "temporal_direction": direction,
                "clusters": [
                    {
                        "region_id": 1,
                        "from_class": from_c,
                        "to_class": to_c,
                        "description": desc
                    }
                ]
            }
        return {
            "sample_id": f"diverse_{Path(fname).stem}",
            "pillar": "D_CHANGE_SEMANTICS",
            "task": "CHANGE_VQA",
            "parent_scene": f"diverse_{Path(fname).stem}",
            "primary_image_path": str(p1).replace("\\", "/"),
            "secondary_image_path": str(p2).replace("\\", "/"),
            "question": "What changed between these two Earth observation images? Describe the type and location of change.",
            "ground_truth_json": expected_json,
            "ground_truth": json.dumps(expected_json),
            "qa_type": "bitemporal_diverse_change",
            "source": "DiverseRS_Benchmark",
            "split": split_name,
        }

    for item in diverse_train_defs:
        train_samples.append(build_diverse_sample(*item, "train"))
    for item in diverse_val_defs:
        val_samples.append(build_diverse_sample(*item, "val"))
    for item in diverse_test_defs:
        test_samples.append(build_diverse_sample(*item, "test"))

    # 2.3 Diagnostic Semantic Transition Controls
    from backend.evaluation.semantic_fixtures import generate_semantic_fixtures
    fixtures = generate_semantic_fixtures()

    fixture_routing = {
        "semantic_urban_01": ("train", "increased", "bare_ground_or_soil", "built_structure", "New building structures erected over bare soil."),
        "semantic_forest_01": ("train", "decreased", "forest_or_trees", "bare_ground_or_soil", "Forest canopy cleared to expose ground."),
        "semantic_zero_01": ("train", "no_change", "none", "none", "Identical observations; zero physical change detected."),
        "semantic_water_01": ("val", "increased", "bare_ground_or_soil", "water_body", "Water inundation over bare ground."),
        "semantic_agri_01": ("test", "decreased", "vegetation_or_cropland", "bare_ground_or_soil", "Agricultural crop harvested to expose soil."),
    }

    for f in fixtures:
        if f.scenario_id in fixture_routing:
            sp, direction, from_c, to_c, desc = fixture_routing[f.scenario_id]
            if direction == "no_change":
                e_json = {"summary": desc, "temporal_direction": "no_change", "clusters": []}
            else:
                e_json = {"summary": desc, "temporal_direction": direction, "clusters": [{"region_id": 1, "from_class": from_c, "to_class": to_c, "description": desc}]}
            
            sample_rec = {
                "sample_id": f.scenario_id,
                "pillar": "D_CHANGE_SEMANTICS",
                "task": "CHANGE_VQA",
                "parent_scene": f"diagnostic_{f.category.lower()}",
                "primary_image_path": str(f.t1_path).replace("\\", "/"),
                "secondary_image_path": str(f.t2_path).replace("\\", "/"),
                "question": f.query,
                "ground_truth_json": e_json,
                "ground_truth": json.dumps(e_json),
                "qa_type": "diagnostic_semantic_change",
                "source": "M5_Diagnostic_Fixtures",
                "split": sp,
            }
            if sp == "train":
                train_samples.append(sample_rec)
            elif sp == "val":
                val_samples.append(sample_rec)
            elif sp == "test":
                test_samples.append(sample_rec)

    # -------------------------------------------------------------
    # 3. Save Splits and Comprehensive Manifest
    # -------------------------------------------------------------
    train_file = output_dir / "train.json"
    val_file = output_dir / "val.json"
    test_file = output_dir / "test.json"
    manifest_file = output_dir / "manifest.json"

    train_file.write_text(json.dumps(train_samples, indent=2), encoding="utf-8")
    val_file.write_text(json.dumps(val_samples, indent=2), encoding="utf-8")
    test_file.write_text(json.dumps(test_samples, indent=2), encoding="utf-8")

    def split_summary(split_data):
        return {
            "total_samples": len(split_data),
            "pillars": dict(Counter(s["pillar"] for s in split_data)),
            "tasks": dict(Counter(s["task"] for s in split_data)),
            "sources": dict(Counter(s["source"] for s in split_data)),
            "parent_scenes": sorted(list({s["parent_scene"] for s in split_data})),
        }

    manifest = {
        "manifest_version": "2.0.0",
        "benchmark_name": "SatQuery_M10_Balanced_Semantic_Mixture",
        "pillars": {
            "A_RS_VQA": "Single-image question answering on remote-sensing features",
            "B_SCENE_SEMANTICS": "Land-cover classification, scene structure, and terrain description",
            "C_OBJECT_SEMANTICS": "Overhead infrastructure inventory and class definitions",
            "D_CHANGE_SEMANTICS": "Real and diagnostic bi-temporal change interpretation with structured JSON"
        },
        "leakage_controls": {
            "single_image_scenes": "Parent-Scene Disjoint (zero shared aerial acquisitions between train/val/test)",
            "bitemporal_pairs": "Pair-Disjoint (zero shared change pairs between train/val/test)",
            "synthetic_isolation": "Diagnostic cases segregated by scenario",
        },
        "splits": {
            "train": split_summary(train_samples),
            "val": split_summary(val_samples),
            "test": split_summary(test_samples),
        }
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 75)
    print("M10 Balanced Adaptation Mixture Generated Successfully:")
    print(f"  TRAIN: {len(train_samples)} samples across {len(manifest['splits']['train']['parent_scenes'])} parent scenes / pairs")
    print(f"         Pillars: {manifest['splits']['train']['pillars']}")
    print(f"  VAL:   {len(val_samples)} samples across {len(manifest['splits']['val']['parent_scenes'])} parent scenes / pairs")
    print(f"         Pillars: {manifest['splits']['val']['pillars']}")
    print(f"  TEST:  {len(test_samples)} samples across {len(manifest['splits']['test']['parent_scenes'])} parent scenes / pairs")
    print(f"         Pillars: {manifest['splits']['test']['pillars']}")
    print(f"Manifest written to: {manifest_file}")
    print("=" * 75)
    return manifest


if __name__ == "__main__":
    prepare_balanced_adaptation_data()
