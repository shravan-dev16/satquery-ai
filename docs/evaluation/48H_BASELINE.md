# SatQuery AI — 48-Hour Model Adaptation Experiment: Baseline Freeze

**Document ID:** `docs/evaluation/48H_BASELINE.md`  
**Timestamp:** 2026-09-20T19:20:00+05:30  
**Status:** **FROZEN RELEASE BASELINE (DO NOT OVERWRITE)**  
**Hardware Environment:** NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM), CUDA 12.4, PyTorch 2.6.0+cu124  

---

## 1. Release Integrity & Repository State

| Property | Value | Verification Status |
|---|---|---|
| **Git Commit** | `419d150b06157019ea3e91948fc9921be20208c6` | Verified (`git rev-parse HEAD`) |
| **Git Working Tree** | Clean (`nothing to commit, working tree clean`) | Verified |
| **Active Branch** | `main` (synced with `origin/main`) | Verified |
| **Golden Adapter Path** | `models/adapters/qwen2_vl_rs_lora_m10_golden/adapter_model.safetensors` | **IMMUTABLE** |
| **Golden Adapter SHA-256** | `31e004da959170e69d8b8e7d280943ac02d780be7a70ad80ce5746f4e50d29e2` | Verified (`Get-FileHash`) |
| **Golden Adapter Size** | 17,476,336 bytes (16.66 MB) | Verified |
| **Active TinyCD Checkpoint** | `models/checkpoints/tinycd_finetuned.pth` | **IMMUTABLE** |
| **TinyCD Checkpoint Size** | 1,283,283 bytes (1.22 MB) | Verified |
| **Pytest Test Suite Status** | 225 passed, 0 failed (160.57s) | Verified |

---

## 2. Frozen Qwen2-VL Remote-Sensing VLM Baseline

Evaluated on the frozen 64-sample test benchmark (`datasets/adaptation/test.json`) across 13 isolated parent scenes:

### 2.1 Overall Accuracy & Pillar Breakdown
| Metric / Pillar | Base Qwen2-VL-2B (Zero-Shot) | Golden M10 RS-LoRA Adapter | Net Gain |
|---|---|---|---|
| **Overall Accuracy** | **68.75%** (44 / 64) | **75.00%** (48 / 64) | **+6.25%** |
| **Pillar A: Single-Image RS VQA** | 51.52% (17 / 33) | **60.61%** (20 / 33) | **+9.09%** |
| **Pillar B: Land-Cover / Scene Semantics** | 87.50% (14 / 16) | **87.50%** (14 / 16) | 0.00% |
| **Pillar C: RS Object Semantics** | 100.00% (8 / 8) | **100.00%** (8 / 8) | 0.00% |
| **Pillar D: Bi-Temporal Change Semantics** | 42.86% (3 / 7) | **71.43%** (5 / 7) | **+28.57%** |

### 2.2 System Safety & Efficiency Metrics
| Metric | Baseline Value | Standard / Threshold |
|---|---|---|
| **`CHANGE_VQA` JSON Validity** | 100.0% (7 / 7 structured JSON outputs) | Required: 100% |
| **Mean Inference Latency** | 1,154 ms per query | Target: < 2,000 ms |
| **Peak Training VRAM** | 7,994 MB (64.8% of 12 GB GPU) | Budget: < 10,000 MB |
| **Peak Inference VRAM** | 4,218 MB | Budget: < 6,000 MB |
| **Trainable Parameter Ratio** | 4,358,144 / 2,213,343,744 (0.1969%) | PEFT LoRA $r=16, \alpha=32$ |

---

## 3. Frozen TinyCD Bi-Temporal Change Detection Baseline

Evaluated on held-out test splits with strict geographic isolation (`FROZEN_TEST_SCENES = {"val_18", "val_19", "val_20"}`):

### 3.1 LEVIR-CD Formal Validation Benchmark (20 Samples)
| Metric | CVA Baseline (Deterministic) | TinyCD Specialist (Neural Siamese) | Delta |
|---|---|---|---|
| **Mean IoU** | 0.0327 | **0.8347** | **+0.8020** |
| **Mean F1 Score** | 0.0619 | **0.9091** | **+0.8472** |
| **Mean Precision** | 0.0913 | **0.9216** | **+0.8303** |
| **Mean Recall** | 0.0547 | **0.8985** | **+0.8438** |
| **Mean False Positive Rate (FPR)**| 0.0346 | **0.0052** | **-0.0294** |
| **Mean Inference Latency** | 52.64 ms | **95.45 ms** | +42.81 ms |
| **Peak Model VRAM** | 0.0 MB (CPU) | **219.4 MB** | < 500 MB budget |

### 3.2 Diverse Remote Sensing & Nuisance Benchmark (11 Samples)
| Metric | Measured Value | Evaluation Context |
|---|---|---|
| **Mean IoU** | 0.2637 | Cross-domain evaluation across coastal, mining, forest, urban, agricultural |
| **Mean F1** | 0.2684 | Zero-shot transfer to non-building land-cover categories |
| **Mean False Positive Rate (FPR)** | **0.0021** | Robust resistance to seasonal hue, shadow, and illumination shifts |
| **Zero-Change Accuracy** | **100.0%** | Zero false alarm rate on negative test pairs |

---

## 4. Current Training Configurations

### 4.1 Qwen2-VL LoRA Configuration
- **Script:** `scripts/train_rs_lora.py`
- **Base Model ID:** `Qwen/Qwen2-VL-2B-Instruct`
- **LoRA Targets:** `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **LoRA Rank ($r$):** 16, **Alpha ($\alpha$):** 32, **Dropout:** 0.05
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay 0.01)
- **Learning Rate:** $1.2 \times 10^{-4}$ with linear warmup
- **Effective Batch Size:** 4 (per-device batch 1 $\times$ gradient accumulation 4)
- **Precision:** BF16 with native PyTorch gradient checkpointing

### 4.2 TinyCD Architecture & Hyperparameters
- **Script:** `scripts/train_tinycd.py`
- **Architecture:** Siamese encoder with dual-scale spatial-temporal cross-attention
- **Parameters:** 316,301
- **Loss Function:** Combined BCE + Soft Dice Loss ($0.5 \times \text{BCE} + 0.5 \times \text{Dice}$)
- **Crop Size:** $256 \times 256$
- **Augmentation:** Random horizontal/vertical flips, JPEG compression (Q=35–85), brightness/contrast jitter
- **Best Validation Loss:** 0.0903 (Epoch 1)

---

## 5. Candidate Experiment Output Directories (Protected Isolation)

All experimental models in the 48-Hour sprint will be directed exclusively to:

- `models/adapters/experiments/qwen_rs_exp_a/`
- `models/adapters/experiments/qwen_rs_exp_b/`
- `models/adapters/experiments/qwen_rs_exp_c/`
- `models/checkpoints/experiments/tinycd_exp_a/`
- `models/checkpoints/experiments/tinycd_exp_b/`

The golden directory `models/adapters/qwen2_vl_rs_lora_m10_golden/` remains write-protected.
