# Final Model Configuration

**Document ID:** `docs/evaluation/FINAL_MODEL_CONFIGURATION.md`  
**Evaluation Date:** 2026-09-21  
**Project:** SatQuery AI — Vision-Language Assistant for Multimodal Remote Sensing (SIH26167 / ISRO)  
**Hardware Environment:** NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM), CUDA 12.4, PyTorch 2.6.0+cu124  
**Release Status:** **READY_FOR_SIH_DEMO**

---

## 1. Vision-Language Model (VLM)

- **Base Foundation Model:** `Qwen/Qwen2-VL-2B-Instruct` (Apache 2.0)
- **Adaptation Method:** Parameter-Efficient Fine-Tuning (PEFT) with LoRA ($r=16, \alpha=32$, target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
- **Active Promoted Checkpoint:** `models/adapters/experiments/qwen_rs_exp_a/best_checkpoint/`
- **Frozen Benchmark Result (Frozen 64-Sample Test Set):**
  - Base Qwen2-VL-2B: 68.75% (44 / 64)
  - Golden M10 RS-LoRA Adapter: 75.00% (48 / 64)
  - **Promoted Candidate A (`qwen_rs_exp_a`):** **84.38%** (54 / 64) — Net Gain: **+9.38 pp** over Golden baseline, **+15.63 pp** over zero-shot
  - *Pillar Breakdown:*
    - Pillar A (Single-Image RS VQA): **75.76%** (25 / 33)
    - Pillar B (Land-Cover / Scene Semantics): **100.00%** (16 / 16)
    - Pillar C (RS Object Semantics): **100.00%** (8 / 8)
    - Pillar D (Bi-Temporal Change Semantics): **71.43%** (5 / 7)
    - Downstream JSON Output Validity: **100.0%** (7 / 7 structured JSON records)
- **Independent Benchmark Result (Independent 178-Sample Evaluation Set):**
  - Golden M10 Adapter: 61.24% (109 / 178)
  - **Promoted Candidate A (`qwen_rs_exp_a`):** **69.66%** (124 / 178) — Net Gain: **+8.42 pp** over Golden baseline
  - *Cross-Dataset Generalization Breakdown:*
    - RSVQA-HR: **57.60%** (72 / 125)
    - CDVQA: **100.00%** (20 / 20)
    - OSCD: **95.65%** (22 / 23)
    - Hard Negatives / Anti-Hallucination: **100.00%** (10 / 10)
- **Scientific Finding:** Candidate A demonstrated improved performance on the frozen internal benchmark and the independent 178-sample evaluation used in this study. Candidate A is not claimed to be universally superior across every remote-sensing task or sensor, but achieved measurable, reproducible superiority across the multi-pillar benchmark.

---

## 2. Change Detector

- **Model Architecture:** TinyCD Siamese Neural Network with EfficientNet-B4 backbone & lightweight cross-attention (~316k parameters)
- **Active Promoted Checkpoint:** `models/checkpoints/tinycd_finetuned.pth`
- **Benchmark Results:**
  - LEVIR-CD Formal Validation Benchmark (20 pairs):
    - Macro Mean IoU: **0.8347**
    - Macro Mean F1: **0.9091**
    - Macro Precision: **0.9216**
    - Macro Recall: **0.8985**
    - False-Positive Ratio: **0.0052**
  - Nuisance Invariance Benchmark (Registration Jitter & Illumination Shifts):
    - False Positive Pixels: **0** (100% precision against illumination shifts and sub-pixel jitter)
- **Reason Baseline Retained (Crucial Engineering Decision):**
  While TinyCD Exp A achieved higher Diverse RS IoU (0.6036 vs 0.4202) on complex agricultural terrain, it exhibited increased false positives on subtle spectral variations and non-urban boundaries. For the SIH live demonstration, the fine-tuned baseline checkpoint (`models/checkpoints/tinycd_finetuned.pth`) remains the safest, most conservative, and highest-precision default to prevent false-alarm overlays during evaluators' tests. TinyCD Exp A is preserved strictly as an experimental artifact (`models/checkpoints/experiments/tinycd_exp_a/tinycd.pth`).

---

## 3. Golden Artifacts & Cryptographic Integrity

Both golden baseline artifacts are 100% intact, immutable, and isolated:

| Artifact Name | Path | SHA-256 Hash | Integrity Status |
|---|---|---|---|
| **M10 Golden VLM Adapter** | `models/adapters/qwen2_vl_rs_lora_m10_golden/adapter_model.safetensors` | `31e004da959170e69d8b8e7d280943ac02d780be7a70ad80ce5746f4e50d29e2` | **VERIFIED (EXACT MATCH)** |
| **Baseline Finetuned TinyCD** | `models/checkpoints/tinycd_finetuned.pth` | `04eb7c032d23a67e23014fe4238cc660f55fcff1c78d5c5c9b42bd6f2338e400` | **VERIFIED (EXACT MATCH)** |
| **Promoted Candidate A Adapter** | `models/adapters/experiments/qwen_rs_exp_a/best_checkpoint/adapter_model.safetensors` | `2b392bf2a7c39c19c28394bb4318ecb35ad7554a275960d0438d1a988aac3f56` | **VERIFIED** |

---

## 4. Runtime & Hardware Telemetry

Measured on active FastAPI live server (`uvicorn backend.main:app`) running on NVIDIA RTX 4070 SUPER (12 GB VRAM):

| Parameter | Measured Value | Standard / Budget |
|---|---|---|
| **VLM Model Load Status** | Success (`is_adapted=True`) | Required: True |
| **Active VLM Adapter Path** | `models/adapters/experiments/qwen_rs_exp_a/best_checkpoint` | Verified |
| **Active TinyCD Path** | `models/checkpoints/tinycd_finetuned.pth` | Verified |
| **Warm Single-Image VQA Latency** | **467 ms** | Target: < 2,000 ms |
| **Change Detection Latency** | **629 ms** | Target: < 1,500 ms |
| **Bi-Temporal Change VQA Latency** | **14,343 ms** (full pipeline: align + TinyCD + 3-panel composite + VLM reasoning) | Stable |
| **Optical-SAR Fusion Latency** | **173 ms** | Real-time |
| **Peak GPU VRAM Usage** | **5,153 MB** (41.9% of 12 GB GPU) | Budget: < 7,500 MB |
| **Representative Confidence Values** | VQA: `0.9900`, Grounding: `0.6156`, Change Detect: `0.9900`, Semantic Change: `0.7964`, Optical-SAR: `0.6347` | Bounded $[0.05, 0.99]$ |

---

## 5. End-to-End Verification & Workflow Audit

### 5.1 Full Pytest Regression Suite
- Total Tests: **225**
- Passed: **225**
- Failed: **0**
- Errors: **0**

### 5.2 11 Production Workflows Audit (`scripts/audit_production_workflows.py`)
- Single-Image VQA (`single_image_vqa`): **PASS** (RS_VQA, Conf: 0.9900)
- Text-Guided Grounding (`single_image_grounding`): **PASS** (RS_GROUND, Conf: 0.7158)
- Pure Bi-Temporal Change (`bitemporal_change_detection`): **PASS** (CHANGE_DETECT, Conf: 0.9900)
- Building Semantic Change (`bitemporal_change_vqa`): **PASS** (CHANGE_DETECT + CHANGE_VQA, Conf: 0.8241)
- Semantic Quantity Query (`bitemporal_change_vqa`): **PASS** (CHANGE_DETECT + CHANGE_VQA, Conf: 0.7964)
- Vegetation Change (`bitemporal_change_vqa`): **PASS** (CHANGE_DETECT + CHANGE_VQA, Conf: 0.8672)
- Water Flood Change (`bitemporal_change_vqa`): **PASS** (CHANGE_DETECT + CHANGE_VQA, Conf: 0.8871)
- Optical + SAR Joint Analysis (`optical_sar_analysis`): **PASS** (OPTICAL_SAR_FUSION, Conf: 0.6444)
- PNG $\rightarrow$ PNG Bi-Temporal (`bitemporal_change_detection`): **PASS** (CHANGE_DETECT, Conf: 0.7961)
- JPEG $\rightarrow$ JPEG Bi-Temporal (`bitemporal_change_detection`): **PASS** (CHANGE_DETECT, Conf: 0.7339)
- GeoTIFF $\rightarrow$ GeoTIFF Metric (`bitemporal_change_detection`): **PASS** (CHANGE_DETECT, Conf: 0.9900)

### 5.3 6 Official Live Demo Scenarios (`scripts/run_final_live_demo_scenarios.py`)
- **A. VQA ("What is visible in this scene?"):** **PASS** (Candidate A active, direct answer, trace shown)
- **B. GROUNDING ("Locate the water body."):** **PASS** (RS_GROUND active, 2 boxes detected, pixel and geo coords verified)
- **C. CHANGE ("What changed between these two dates?"):** **PASS** (CHANGE_DETECT active, mask & regions generated, metric area 64.0 ha computed)
- **D. SEMANTIC CHANGE ("Has the built-up area increased..."):** **PASS** (CHANGE_DETECT $\to$ CHANGE_VQA chained, semantic interpretation separated from physical mask)
- **E. SEMANTIC QUANTITY ("By how many hectares did the built-up area increase?"):** **PASS** (Total physical change reported; no fabricated semantic hectares, qualified limitation notice emitted)
- **F. OPTICAL + SAR ("Analyze optical and SAR features together."):** **PASS** (OPTICAL_SAR_FUSION active, cross-modal spectral/backscatter complementarity confirmed)

---

## 6. Known Limitations

1. **Benchmark Scope:** The performance metrics reflect evaluation across the frozen internal 64-sample benchmark and the independent 178-sample test corpus. Generalization performance on uncalibrated sensors or out-of-distribution geographical domains may vary.
2. **OSCD Candidate Regression:** While Candidate A achieved 95.65% on OSCD change verification, fine-grained multi-temporal urban classification on high-resolution Sentinel-2 triplets requires ongoing multi-sensor pre-training.
3. **Confidence is an Evidence-Driven Heuristic:** System confidence is an evidence-weighted heuristic (`CONFIDENCE = EVIDENCE_DRIVEN_HEURISTIC`) combining model certainty, registration PSR, and cross-model agreement. It is not an uncalibrated probability or arbitrary score, but neither does it constitute a statistically calibrated Bayesian posterior.
4. **Semantic Class-Specific Area Limitation:** Physical change areas ($m^2$, hectares) are calculated from the verified binary change mask. Qualitative semantic models describe transitions but cannot reliably allocate exact hectare counts to specific land-cover classes without pixel-level multi-class semantic segmentation.
5. **CRS Requirement for Geographic/Metric Coordinates:** Ground metric units ($m^2$, hectares) and GeoJSON polygon boundaries require a valid Projected Coordinate System (PCS). For non-georeferenced images (PNG/JPEG), only normalized pixel coordinates and pixel percentages are reported.
