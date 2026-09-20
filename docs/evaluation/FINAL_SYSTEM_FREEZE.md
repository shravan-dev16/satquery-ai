# SatQuery AI — Final System Freeze & Technical Verification Report

**Milestone:** Extended Maximum Training, Cross-Format Alignment Hardening & Freeze Readiness  
**Target:** Smart India Hackathon — SIH26167 (ISRO)  
**Status:** FROZEN & VERIFIED  
**Audit Date:** 2026-09-20  

---

## 1. Executive Summary

SatQuery AI is an agentic, multimodal remote-sensing analysis assistant designed for natural-language interpretation over single, paired, bi-temporal, and cross-modal Earth observation imagery. 

This document records the final system freeze following the completion of the extended training campaign, the dataset isolation audit, and the targeted cross-format alignment investigation. All headline metrics have been verified against frozen test benchmarks, held-out splits, and end-to-end production endpoints.

---

## 2. Final System Architecture

```
                                  [ User Request + Imagery ]
                                               │
                                               ▼
                                   [ Input / Modality Sniffer ]
                                               │
                         ┌─────────────────────┴─────────────────────┐
                         ▼                                           ▼
               [ Mode A: Georeferenced ]                   [ Mode B: Non-Georeferenced ]
                 • GeoTIFF with CRS                          • PNG / JPEG / Local TIFF
                 • Reprojection & Resampling                 • 2D FFT Phase Correlation (PSR)
                 • Metric Areas (ha, m²)                     • Sub-Pixel Offset (dx, dy)
                 • Exact Geospatial Polygons                 • Pixel Coordinate Space (px)
                         │                                           │
                         └─────────────────────┬─────────────────────┘
                                               │
                                               ▼
                                  [ Agentic Query Router ]
                                               │
           ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
           ▼                   ▼                               ▼                   ▼
      [ RS_VQA ]         [ RS_GROUND ]                 [ CHANGE_DETECT ]    [ OPTICAL_SAR ]
    Qwen2-VL-2B +      Grounding DINO Tiny            TinyCD Fine-Tuned    Dual-Stream dB +
    Domain LoRA         Open-Vocabulary               + Bimodal CVA Fallback NDVI Fusion
           │                   │                               │                   │
           └───────────────────┴───────────────┬───────────────┴───────────────────┘
                                               │
                                               ▼
                                  [ Evidence Bundle Packager ]
                                               │
                                               ▼
                                 [ M8 Consistency Checker ]
                                               │
                                               ▼
                                [ M9 Explainable Confidence ]
                                               │
                                               ▼
                              [ Structured Result + UI Overlays ]
```

---

## 3. Active Models & Checkpoints

| Component | Identifier | Base Architecture | Checkpoint / Adapter Path | Trainable Parameters | Execution Role |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **VQA Specialist** | `RS_VQA` | Qwen2-VL-2B-Instruct | `models/adapters/qwen2_vl_rs_lora/` | 13,893,632 | Primary visual reasoning specialist |
| **Grounding Specialist** | `RS_GROUND` | Grounding DINO Tiny | HuggingFace Hub cached (`grounding-dino-tiny`) | 172,000,000 | Open-vocabulary spatial grounding |
| **Bi-Temporal Change** | `CHANGE_DETECT` | TinyCD (EfficientNet-B4) | `models/checkpoints/tinycd_finetuned.pth` | 285,128 | Primary neural change detector |
| **Change Fallback** | `CHANGE_DETECT_CVA` | Deterministic CVA | N/A (Algorithmic Otsu thresholding) | 0 | Autonomous fallback detector (Rule 24) |
| **Optical-SAR Fusion** | `OPTICAL_SAR_FUSION` | Dual-Stream Multi-Modal | N/A (Deterministic dB calibration + ratio) | 0 | Structural cross-modal reasoning |

### Checkpoint Integrity Verification:
- **TinyCD Production Checkpoint:** `models/checkpoints/tinycd_finetuned.pth` exists (`1,280,183 bytes`) and is loaded by default by `ChangeDetectionSpecialist`.
- **M10 Golden Adapter Backup:** `models/adapters/qwen2_vl_rs_lora_m10_golden/adapter_model.safetensors` verified with SHA-256 hash `31e004da959170e69d8b8e7d280943ac02d780be7a70ad80ce5746f4e50d29e2` (100% identical and unchanged).

---

## 4. Evaluation Methodology & Performance Metrics

### 4.1 Vision-Language Adaptation (Qwen2-VL LoRA)
Evaluated on the frozen 64-sample test benchmark (`datasets/adaptation/test.json`), strictly isolated with zero parent scene overlap with training or validation data:

| Metric | Target | Verified Score | Sample Count | Evaluation Split |
| :--- | :---: | :---: | :---: | :--- |
| **Exact String Match** | $\ge 70.0\%$ | **75.00%** (48/64) | 64 | `datasets/adaptation/test.json` (Frozen Test) |
| **Semantic Relaxed Match** | $\ge 75.0\%$ | **76.56%** (49/64) | 64 | `datasets/adaptation/test.json` (Frozen Test) |
| **Object Identification Accuracy** | $\ge 85.0\%$ | **100.00%** (16/16) | 16 | Category: Objects / Ground Vehicles |
| **Scene Semantic Accuracy** | $\ge 80.0\%$ | **93.75%** (15/16) | 16 | Category: Land Cover / Scene Classification |

### 4.2 Bi-Temporal Change Detection (TinyCD Fine-Tuning)
Evaluated on held-out LEVIR-CD validation scenes `val_15`, `val_16`, and `val_17` (strictly excluded from training):

| Metric | Pretrained Baseline | Fine-Tuned Checkpoint | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Building Change IoU** | 0.842 | **0.884** | **+4.99%** |
| **Inference Latency (256x256)** | 18.2 ms | **18.2 ms** | 0.0 ms (Zero latency penalty) |
| **VRAM Footprint** | ~450 MB | ~450 MB | Well within 12 GB GPU budget |

---

## 5. Cross-Format Alignment & Robustness Audit

### 5.1 Sub-Pixel Phase Correlation with PSR Prominence
The spatial correspondence engine evaluates 2D FFT phase correlation with Peak-to-Sidelobe Ratio (PSR) prominence:

$$\text{PSR} = \frac{P_{\text{phase}} - \mu_{\text{sidelobe}}}{\max(10^{-6}, \sigma_{\text{sidelobe}})}$$

$$P_{\text{prominence}} = \min\left(1.0, \max\left(\frac{P_{\text{phase}}}{0.65}, \frac{\text{PSR}}{60.0}\right)\right)$$

$$S_{\text{align}} = 0.55 \cdot P_{\text{prominence}} + 0.35 \cdot C_{\text{robust}} + 0.10 \cdot \max(0.0, \text{NCC}^*)$$

- **Resolution of False Alignment Warnings:** The previously identified false alignment warnings caused by JPEG spectral attenuation were resolved; remaining cross-format benchmark misses are attributable to the synthetic TinyCD change-magnitude case, not alignment failure.
- **PNG $\rightarrow$ JPEG Q85:** Alignment score improved from **0.5883** to **0.8128** ($PSR = 107.2\sigma$, offset $(0.0, 0.0)$).
- **JPEG Q85 $\rightarrow$ TIFF:** Alignment score improved from **0.5502** to **0.7761** ($PSR = 106.7\sigma$, offset $(0.0, 0.0)$).
- **03_water Scenario:** Alignment score improved from **0.4856** to **0.6189** (58% water coverage no longer triggers false misalignment).
- **Unrelated Pair Rejection:** Unrelated images produce peak $0.0185$ and $PSR = 4.8$, resulting in alignment score **0.0349** ($\ll 0.15$ threshold, strictly rejected).
- **Translation Shift Detection:** Large spatial offsets remain accurately detected ($(-30.0, -40.0)$ and $(-55.0, -45.0)$ px) with full translation penalties applied.
- **Total Alignment Failures:** Reduced from **3 to 0** across all 12 evaluated pair combinations.

### 5.2 Format Evaluation Matrix (12 Pairs)
Derived directly from [`scripts/audit_format_matrix.py`](file:///c:/Users/Shravan/Desktop/Projects/satquery-ai/scripts/audit_format_matrix.py):

| Format Combination | Pair Type | Samples | Correct | Accuracy | Alignment Failures | False Positives |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **GeoTIFF $\rightarrow$ GeoTIFF** | Same format | 1 | 1 | **100.0%** | **0** | 0 |
| **JPEG $\rightarrow$ JPEG** | Same format | 2 | 2 | **100.0%** | **0** | 0 |
| **JPEG $\rightarrow$ TIFF** | Mixed format | 1 | 0 | **0.0%** | **0** *(was 1)* | 0 |
| **PNG $\rightarrow$ JPEG** | Mixed format | 1 | 0 | **0.0%** | **0** *(was 1)* | 0 |
| **PNG $\rightarrow$ PNG** | Same format | 6 | 5 | **83.3%** | **0** *(was 1)* | 0 |
| **PNG $\rightarrow$ TIFF** | Mixed format | 1 | 1 | **100.0%** | **0** | 0 |
| **TOTAL** | **All Format Permutations** | **12** | **9** | **75.0%** | **0** *(was 3)* | **0** |

### 5.3 Operational Benchmark Scenarios (10 Scenarios)
Evaluated on `datasets/change_robustness/`:
- **Scenario Success Rate:** 9/10 (90.0%) under pure TinyCD (>1.0% threshold) / 10/10 (100.0%) with CVA baseline.
- **Zero-Change Scene Accuracy:** 100.0% (`06_no_change` produces exactly 0 changed pixels).
- **Seasonal Variation Invariance:** 100.0% (`07_seasonal` produces 0 changed pixels; phenology shift does not cause false change).
- **Compression Artifact Resistance:** 100.0% (`08_compression` produces 85 pixels / 0.13%, rejected as noise).
- **Misalignment Detection:** 100.0% (`09_misalignment` detects exact shift $(-55.0, -45.0)$ px and emits alignment warning).

---

## 6. Confidence Characterization (`CONFIDENCE = EVIDENCE_DRIVEN_HEURISTIC`) & Reliability

> **Specification (`CONFIDENCE = EVIDENCE_DRIVEN_HEURISTIC`):** Confidence estimation in M9/M13 operates as an **evidence-driven heuristic** combining model certainty, input raster quality, co-registration alignment PSR, and cross-specialist consistency gating. M9 is strictly not a generally calibrated posterior probability mechanism. Empirical evaluation on the stated calibration evaluation set ($N=10$) produced $\text{ECE} = 0.0380$ and $\text{Brier} = 0.0412$, but these empirical results do not claim to establish general calibrated posterior probabilities.

Confidence is computed dynamically by `ConfidenceEngine` using five verifiable signals:
$$C_{\text{system}} = 0.35 \cdot Q_{\text{model}} + 0.25 \cdot Q_{\text{align}} + 0.20 \cdot Q_{\text{evidence}} + 0.10 \cdot Q_{\text{input}} + 0.10 \cdot Q_{\text{consistency}}$$

| Calibration Metric | Target | Measured Value | Status |
| :--- | :---: | :---: | :---: |
| **Expected Calibration Error (ECE)** | $\le 0.08$ | **0.0380** | **PASS** |
| **Brier Score** | $\le 0.10$ | **0.0412** | **PASS** |

### Verified Confidence Tier Behavior (Cases A–G):
- **Case A (Aligned Pair + Obvious Change):** Confidence = `0.8186` (**HIGH** Tier)
- **Case B (Aligned Pair + Verified Zero Change):** Confidence = `0.9235` (**HIGH** Tier)
- **Case C (Unrelated Images):** Confidence = `0.2903` (**UNSUPPORTED / LOW** Tier)
- **Case D (Badly Misaligned Pair):** Confidence = `0.3162` (**UNSUPPORTED / LOW** Tier)
- **Case E (Compressed Pair Q=30):** Confidence = `0.5956` (**MEDIUM** Tier, degraded proportionally)
- **Case F (Contradictory Evidence):** Confidence = `0.0500` (**UNSUPPORTED / LOW** Tier)
- **Case G (Non-Georeferenced Image):** Metric areas omitted; confidence uninflated

---

## 7. Geospatial Scientific Boundaries & Known Limitations

1. **Geospatial Area Policy:**
   - **Mode A (GeoTIFF with CRS and Transform):** Calculates and reports true ground surface area in hectares ($10,000\,m^2$) and square meters.
   - **Mode B (PNG, JPEG, local unprojected TIFF):** Metric units ($m^2$, ha) are strictly omitted. Areas are reported in pixel counts and relative percentages only, accompanied by explicit qualification warnings.
2. **Known Synthetic Benchmark Limitation:**
   - `01_building` in `datasets/change_robustness` is a synthetic test fixture with flat computer-generated polygons. Neural TinyCD (trained on real aerial imagery) activates primarily on boundary edges (396 pixels = 0.60% < 1.0% threshold) in native PNG $\rightarrow$ PNG, and drops to 106 pixels (0.16%) under lossy JPEG Q85. Because `mixed_png_jpeg` and `mixed_jpeg_tiff` synthesize from `01_building`, they score 0% under the benchmark script's $>1.0\%$ threshold. On natural scene categories (`04_roads`, `10_cross_format`) and under CVA, cross-format change detection succeeds (2.85% to 18.63% > 1.0%).

---

## 8. Exact Verification Test Counts

- **Full Pytest Suite:** **222 passed, 0 failed** in 163.47s across 33 test files.
- **Dedicated Format Robustness Suite:** **16 passed, 0 failed** in `tests/test_png_jpeg_bitemporal.py`.
- **Dedicated Confidence Suite:** **18 passed, 0 failed** in `tests/test_confidence.py`.
- **Production API Workflows Audit:** **11 passed, 0 failed** through `POST /api/v1/analyze`.

---

## 9. Final Freeze Status

The system is verified clean, reproducible, and robust.

```
FINAL_SYSTEM_STATUS = FROZEN_AND_VERIFIED
QWEN_EXACT_MATCH = 75.00%
QWEN_SEMANTIC_RELAXED = 76.56%
TINYCD_HELD_OUT_IOU = 0.884
FORMAT_PAIR_ACCURACY = 75.00%
FORMAT_SCENARIO_ACCURACY = 90.00%
ALIGNMENT_FAILURES = 0
FALSE_POSITIVES = 0
CONFIDENCE_ECE = 0.0380
FULL_TESTS = 222/222
M10_GOLDEN_UNCHANGED = TRUE
READY_FOR_M12 = TRUE
```
