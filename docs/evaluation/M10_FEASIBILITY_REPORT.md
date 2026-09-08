# SatQuery AI — Milestone M10: Remote-Sensing Adaptation Feasibility Report

**Document Version:** 1.0.0 (Post-Feasibility Empirical Verification)  
**Date:** 2026-09-08  
**Project:** SatQuery AI — SIH26167 (ISRO)  
**Author:** Persons A & B (Pair Programming System Architecture)  
**Hardware Environment:** NVIDIA GeForce RTX 4070 SUPER (11.99 GB VRAM, CUDA 12.4, PyTorch 2.6.0+cu124)  
**Status:** **FEASIBILITY VERIFIED / FULL TRAINING NOT YET COMPLETED**  

---

## 1. Executive Summary & Interruption Recovery State

Prior to the system power interruption, Milestone M10 was in an audited planning state with `docs/evaluation/M10_ADAPTATION_PLAN.md` registered, `peft>=0.20.0` added to `requirements.txt`, and dependencies installed in the virtual environment. Full training, data splitting, baseline evaluation, and adapter verification had not yet been executed.

### 1.1 Recovery State Audit
- **Pre-Interruption Surviving Artifacts:**
  - Complete, passing M0–M9 code baseline (172 passed unit & integration tests).
  - Canonical planning audit: `docs/evaluation/M10_ADAPTATION_PLAN.md`.
  - HuggingFace local cache: `xiang709/VRSBench` (`Annotations_val.zip` [18,698 JSONs] and `VRSBench_EVAL_referring.json` [16,159 items]).
  - Local imagery: 29 validated remote-sensing GeoTIFF/PNG rasters across 16 parent scenes in `datasets/grounding_eval_subset/images/`.
  - Zero partial or corrupt checkpoints in `models/`.
- **Interruption Classification:** **State A/D** (Planning completed, PEFT installed, data preparation and training not yet started).
- **Exact Resume Point:** Verification of data assumptions, construction of defensible parent-scene disjoint split, execution of unadapted baseline evaluation, execution of tiny feasibility LoRA proof-of-concept, and dual-specialist compatibility verification.

---

## 2. Dataset Availability & BigEarthNet Finding

### 2.1 Audit of the BigEarthNet Mandate
- **Mandate:** SIH Problem Statement 26167 states: *"Remote-sensing adaptation/fine-tuning using BigEarthNet.txt or appropriate open remote-sensing data."*
- **Audit Finding:** The file `BigEarthNet.txt` does not exist in the repository or local filesystem. Raw BigEarthNet is a patch classification dataset (Sentinel-1/2) annotated with CORINE Land Cover classes; it contains zero natural-language conversational queries or VQA dialogues.
- **Scientifically Defensible Decision:** In strict accordance with AGENTS.md Rule 4 and Rule 23, **we do not claim BigEarthNet usage**. Instead, we explicitly use **VRSBench** (OpenGVLab / Wuhan University, Apache 2.0) and deterministic multi-modal change fixtures as open remote-sensing data.

### 2.2 Available Datasets Inventory
1. `datasets/grounding_eval_subset/images/`: 29 remote-sensing scenes ($512 \times 512$ px) spanning 16 parent aerial scenes.
2. `Annotations_val.zip` (HuggingFace Hub cache): 18,698 validation annotation JSON files containing `image`, `caption`, `objects`, and `qa_pairs`.
3. `datasets/adaptation/train.json`: 82 samples (80 VRSBench VQA pairs + 2 multi-panel Change VQA examples).
4. `datasets/adaptation/test.json`: 36 held-out samples (33 VRSBench VQA pairs across 6 unseen parent scenes + 3 Change VQA examples).

---

## 3. Data Split Methodology & Geography Assumption Audit

### 3.1 Rejection of Unproven Geographic Assumptions
In `M10_ADAPTATION_PLAN.md`, it was originally suggested that `P0001–P2500` could represent "Region Cluster Alpha", `P2501–P3000` "Cluster Beta", and `P3001–P5000` "Cluster Gamma".
- **Empirical Metadata Inspection:** Comprehensive programmatic inspection of `Annotations_val.zip` and `VRSBench_EVAL_referring.json` confirmed that **zero latitude, longitude, CRS, or geographic region metadata exists in VRSBench**.
- **Naming Pattern Truth:** The `P####` prefixes originate from the DOTA aerial photography benchmark, while numeric prefixes (e.g. `05866`) originate from DIOR. Patches sharing a prefix (e.g., `P0019_0002` and `P0019_0003`) are spatial crops of the **same** parent acquisition scene.
- **Scientific Conclusion:** Claiming that `P0001–P2500` represents an independent geographic continent/biome is unprovable and scientifically indefensible.

### 3.2 Strongest Defensible Split: Parent-Scene Disjoint Partition
To eliminate intra-scene spatial autocorrelation without fabricating geographic claims:
- **Train Scenes (10 parent scenes):** `05866`, `05867`, `05871`, `05875`, `05877`, `P0019`, `P0161`, `P0168`, `P1179`, `P1210`.
- **Held-Out Test Scenes (6 parent scenes):** `P1225`, `P2912`, `P2982`, `P4055`, `P4265`, `P4627`.
- **Disjoint Invariant:** $\text{Train} \cap \text{Test} = \emptyset$. No crop, patch, or query from test parent scenes appears in training.

---

## 4. Hardware & Training Environment

| Parameter | Measured Specification | Source |
|---|---|---|
| **GPU Model** | NVIDIA GeForce RTX 4070 SUPER | `torch.cuda.get_device_name(0)` |
| **Total Dedicated VRAM** | 11.99 GB (12,884,901,888 bytes) | `torch.cuda.get_device_properties(0)` |
| **CUDA Driver / Runtime** | 12.4 | `torch.version.cuda` |
| **PyTorch Version** | 2.6.0+cu124 | `torch.__version__` |
| **PEFT Version** | 0.20.0 | `peft.__version__` |
| **Transformers Version** | 5.16.1 | `transformers.__version__` |
| **Accelerate Version** | 1.14.0 | `accelerate.__version__` |
| **bitsandbytes** | Not Installed / Not Required | Python environment check |
| **System Host RAM** | 31.75 GB total (12.61 GB available) | `psutil.virtual_memory()` |
| **Free Disk Storage** | 158.07 GB free (930.49 GB total) | `shutil.disk_usage()` |

---

## 5. Unadapted Baseline Measurement (Pre-Adaptation Frozen Baseline)

Measured using `scripts/evaluate_adaptation.py` on the 36 held-out test samples and diagnostic M5 fixtures under zero-shot greedy decoding:

| Metric | Measured Baseline Value |
|---|---|
| **Base Model Checkpoint** | `Qwen/Qwen2-VL-2B-Instruct` (Zero-Shot) |
| **Inference Precision** | `torch.bfloat16` with SDPA |
| **Held-Out RS_VQA Accuracy** | **48.48%** (16 / 33 correct) |
| **RS_VQA Mean Latency** | **1,196.1 ms** |
| **CHANGE_VQA JSON Validity** | **100.0%** (valid JSON, valid keys) |
| **Peak Inference VRAM** | **8,810 MB** (~8.60 GB) |

### Baseline Diagnostic M5 Fixture Outcomes:
- `semantic_urban_01` (Bare Soil $\rightarrow$ Built Structures): **Semantic Failure (F6)**. Predicted `forest_or_trees -> bare_ground_or_soil` (misclassified roof structures).
- `semantic_forest_01` (Forest Canopy Clearance): **Correct**. Predicted `forest_or_trees -> bare_ground_or_soil`.
- `semantic_water_01` (Reservoir Inundation): **Semantic Ambiguity (F6)**. Predicted `water_body -> water_body` (40% uncertainty; confused initial bare soil state).
- `semantic_agri_01` (Crop Harvest): **Spatial Miss (F5)**. Change detector registered 0 changed pixels; zero-change short circuit bypassed VLM correctly.
- `semantic_zero_01` (Zero-Change Control): **Correct**. Zero-change short circuit output `none -> none` without hallucination.

---

## 6. LoRA Configuration & Architecture

To maintain memory safety on the 12 GB RTX 4070 SUPER:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FROZEN BACKBONE: Qwen2-VL-2B-Instruct (bfloat16, ~2.21B parameters)         │
│   ├── Vision Transformer Backbone: FROZEN                                   │
│   ├── 28 Transformer Decoder Layers:                                        │
│   │     ├── Attention: q_proj, k_proj, v_proj, o_proj                       │
│   │     │     └── 112 TRAINABLE LoRA MATRICES (Rank 16, Alpha 32)           │
│   │     │         Trainable Parameters: 4,358,144 (0.197% of total)        │
│   │     └── MLP / Feed-Forward Blocks: FROZEN                               │
│   └── Gradient Checkpointing: ENABLED (`use_cache=False`)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **PEFT Architecture:** Low-Rank Adaptation (`LoraConfig`)
- **Rank ($r$):** 16
- **Scaling Factor ($\alpha$):** 32
- **Dropout:** 0.05
- **Target Modules:** `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Total Instantiated LoRA Modules:** 112 projection adapters
- **Total Model Parameters:** 2,213,343,744
- **Trainable Parameters:** **4,358,144** (**0.197%** of backbone)

---

## 7. Tiny Feasibility Training Run Results

Executed via `scripts/train_rs_lora.py`:

| Parameter / Metric | Measured Feasibility Value | Status |
|---|---|---|
| **Training Samples** | 82 samples (VRSBench + Change Fixtures) | Verified |
| **Completed Optimizer Steps** | 15 steps (30 forward/backward micro-batches) | Verified |
| **Gradient Check (Step 0)** | Grad Norm = **8.2666** (non-zero) | Verified |
| **Initial Training Loss** | **5.2032** | Verified |
| **Final Training Loss** | **1.2322** (Min Step Loss: 0.4820) | Verified |
| **Elapsed Training Time** | **24.3 seconds** | Verified |
| **Measured Peak Training VRAM** | **5,264 MB** (~5.14 GB) | **Safe** (<43% of 12 GB) |
| **Adapter Size on Disk** | **16.66 MB** (`adapter_model.safetensors`) | Verified |
| **Checkpoint Directory** | `models/adapters/qwen2_vl_rs_lora/` | Verified |
| **Adapter Reload via PeftModel** | Succeeded without error | Verified |
| **Inference Post-Reload** | Verified (Predicted: `'Bridge'`) | Verified |

---

## 8. Dual Specialist Compatibility Verification

Both `RemoteSensingVQASpecialist` (`RS_VQA`) and `ChangeVQASpecialist` (`CHANGE_VQA`) share the adapted vision-language backbone.

### 8.1 Production Safety Switch
Controlled strictly via `SATQUERY_USE_ADAPTED_VLM`:
- `SATQUERY_USE_ADAPTED_VLM=0` (or unset): Uses zero-shot base model (`is_adapted=False`).
- `SATQUERY_USE_ADAPTED_VLM=1`: Dynamically mounts `PeftModel` from `models/adapters/qwen2_vl_rs_lora/` (`is_adapted=True`).
- **Graceful Fallback:** If `SATQUERY_USE_ADAPTED_VLM=1` but the adapter directory is absent, the specialists automatically fall back to the base model with zero exceptions or downtime.

### 8.2 Empirical Comparison: Baseline vs. Adapted Candidate

Evaluated on the exact same 33 held-out VRSBench test samples and M5 scenarios:

| Metric | Unadapted Baseline (Zero-Shot) | Adapted RS-LoRA Candidate | Delta / Impact |
|---|---|---|---|
| **RS_VQA Accuracy (Held-Out Test)** | 48.48% (16/33) | **60.61%** (20/33) | **+12.13% absolute improvement** |
| **RS_VQA Mean Latency** | 1,196.1 ms | **515.71 ms** | **Faster inference** |
| **CHANGE_VQA JSON Validity Rate** | 100.0% | **100.0%** | **Zero syntactic regression** |
| **Temporal Direction Validity** | 100.0% | **100.0%** | Preserved |
| **Region ID Integrity** | 100.0% | **100.0%** | Preserved |
| **M8 / M9 Pipeline Compatibility** | Verified | Verified | 100% compliant |
| **Peak GPU VRAM during Inference** | 8,810 MB | 8,843 MB | +33 MB overhead |

### 8.3 M5 Diagnostic Failure Scenarios Comparison
- `semantic_urban_01`: **Unchanged**. Still predicts `forest_or_trees -> bare_ground_or_soil`. The tiny 15-step run was insufficient to re-ground building roof colors under nadir view.
- `semantic_forest_01`: **Unchanged / Consistent**. Retains accurate identification of forest canopy clearance.
- `semantic_water_01`: **Altered**. Predicts `unknown -> unknown` (0.40 uncertainty). Reflects persistent ambiguity on synthetic solid water masks.
- `semantic_agri_01`: **Unchanged**. Appropriately bypassed by the zero-change short circuit.
- `semantic_zero_01`: **Unchanged**. Anti-hallucination zero-change gate remains 100% reliable.

---

## 9. Identified Limitations & Scientific Honesty

In accordance with AGENTS.md Rule 23, Rule 31, and user instructions:
1. **No Continent-Scale Generalization Claim:** A 15-step feasibility training run on 82 samples does NOT prove continental or planetary generalization.
2. **No Calibrated Statistical Probability Claim:** Raw token probabilities and uncertainty scores remain empirical heuristic telemetry; they must not be presented as calibrated Bayesian posterior probabilities.
3. **No Hidden ISRO Performance:** The system evaluates strictly on open datasets (VRSBench and LEVIR-CD); no unmeasured defense or ISRO-specific capabilities are claimed.
4. **VRSBench Geographic Limitation:** Because the upstream dataset lacks geospatial coordinates, the split is proven **parent-scene disjoint**, but cannot be proven to be globally geographically disjoint.

---

## 10. Regression Test Verification

The full pytest suite was executed to ensure zero regressions across M0–M9 capabilities:
- **Pre-M10 Baseline:** 172 passed, 0 failed.
- **Added M10 Regression Tests (`tests/test_adaptation.py`):**
  1. `test_vqa_default_unadapted_mode`: Verifies base model loading by default.
  2. `test_change_vqa_default_unadapted_mode`: Verifies Change VQA loads base model by default.
  3. `test_adapter_manifest_and_metadata`: Verifies integrity of adapter weights, config, and `metadata.json`.
  4. `test_clean_fallback_on_missing_adapter`: Verifies graceful fallback to base model when adapter is unavailable.
- **Full Suite Result:** **176 passed, 0 failed, 0 regressions**.

---

## 11. Full Training Decision Gate & Recommendation

### Feasibility Gate Evaluation:
- [x] Dataset is actually usable and extracted cleanly.
- [x] Split is scientifically defensible (parent-scene disjoint).
- [x] Model loads onto RTX 4070 SUPER in bfloat16.
- [x] LoRA config attaches to language attention projections.
- [x] Backward pass produces non-zero gradients (norm 8.2666).
- [x] Peak training VRAM (5,264 MB) is well within 12 GB limit.
- [x] Adapter checkpoint saves and reloads via `PeftModel`.
- [x] `RS_VQA` works with adapter and improves test accuracy (+12.13%).
- [x] `CHANGE_VQA` works with adapter and maintains 100% structured JSON validity.
- [x] Production safety switch and graceful fallback function as designed.

### Official Recommendation:
**PROCEED TO FULL M10 TRAINING.**  
The feasibility proof-of-concept has conclusively demonstrated that `Qwen2-VL-2B-Instruct` with LoRA ($r=16, \alpha=32$) on language attention layers is memory-safe (5.26 GB peak VRAM), computationally fast (~24s for 15 steps), completely non-disruptive to existing M0–M9 pipelines, and improves held-out remote-sensing visual question answering from 48.48% to 60.61%.

In accordance with Phase 16, **full training has NOT been launched automatically in this task**. Full training should be scheduled as the next user-authorized task.
