# SatQuery AI — System Evaluation Matrix

**Document Version:** 1.0.0 (Post-M7 Architecture Lock)  
**Standard:** Compliance with `AGENTS.md` Rule 4 (Mandatory RS Specialization), Rule 15 (Dataset Policy), Rule 25 (Testing Requirements), Rule 26 (Evaluation Requirements), and SIH26167 Requirements.

---

## 1. Evaluation Architecture & Boundary Policy

To maintain scientific integrity and prevent benchmark leakage or false claims, SatQuery AI enforces a strict three-tier separation of all verification and evaluation activities:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TIER A: UNIT & INTEGRATION TEST SUITE                                       │
│ Purpose: Code correctness, schema contract enforcement, boundary tests     │
│ Datasets: Deterministic synthetic fixtures + mini sample (<1.5 MB)         │
│ Status: 137 tests passing (0 failures, 0 regressions)                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ TIER B: ENGINEERING DEVELOPMENT BENCHMARKS                                  │
│ Purpose: Empirical model selection, ablation studies, failure diagnosis     │
│ Datasets: LEVIR-CD (20-sample cut), VRSBench (35-sample cut), Diverse RS    │
│ Status: Completed during M2.1, M4, M5, M6, M7 (documented in evaluation/)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ TIER C: PRESCRIBED SIH BENCHMARK SUITE (SCHEDULED FOR M10)                  │
│ Purpose: Official evaluation on full standard remote-sensing benchmarks     │
│ Datasets: VRSBench, RSVQA, CDVQA, BigEarthNet                               │
│ Status: Scheduled for M10 domain adaptation & evaluation                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Synthetic Data Policy (Non-Negotiable Principle 14)
- **Synthetic Fixtures are for:** Pipeline correctness, schema validation, affine projection verification, boundary edge-case handling, and deterministic regression testing.
- **Real Remote-Sensing Imagery is Required for:** Model performance claims, land-cover accuracy, cross-modal complementary benefits, and benchmark reporting.
- **Rule:** Never cite synthetic test pass rates as evidence of real-world remote-sensing accuracy.

### 1.2 Benchmark Separation Policy (Non-Negotiable Principle 15)
- Engineering benchmark cuts (e.g. 20-sample LEVIR-CD or 35-sample VRSBench) were curated strictly for architectural model selection and regression bounds on developer hardware (RTX 4070 12GB).
- They must **never** be misrepresented as official full-benchmark SIH evaluations.

---

## 2. Tier A: Unit & Integration Test Suite (M0 – M7)

The automated test suite runs via `pytest` and verifies code integrity without requiring external internet access or heavy checkpoint re-downloads.

| Test File | Focus Area | Test Count | Key Invariants Verified |
|---|---|---|---|
| `tests/test_contracts.py` | Core Schemas & Contracts | 12 | Pydantic validation, serialization/deserialization, `StandardResultContract` completeness. |
| `tests/test_registry.py` | Specialist Registry | 5 | `ModelRegistry` thread-safe registration, capability querying, duplicate detection, unregistration. |
| `tests/test_geotiff.py` | Ingestion & Affine Geometry | 10 | Rasterio reading, affine transform calculation, bounding box derivation, coordinate validity. |
| `tests/test_validation.py` | Input & Geospatial Validation | 12 | CRS verification, non-zero dimension checks, band verification, spatial overlap calculation. |
| `tests/test_vqa.py` | Single-Image VQA Specialist | 6 | `RS_VQA` pipeline, prompt conditioning, answer formatting, execution trace generation. |
| `tests/test_grounding.py` | Text-Guided Grounding | 8 | Grounding DINO integration, `[xmin, ymin, xmax, ymax]` ordering, NMS, degeneracy filter ($\ge 98\%$). |
| `tests/test_alignment.py` | Bi-Temporal Alignment | 8 | Reprojection, pixel grid resampling, intersection bounding box clipping. |
| `tests/test_change_metrics.py` | Scientific Change Metrics | 10 | Mathematical IoU, Precision, Recall, F1, FPR, edge-case handling (empty GT, zero-change). |
| `tests/test_change_detection.py` | Bi-Temporal Change Detection | 9 | TinyCD loader, CVA fallback baseline, binary mask derivation, physical area in hectares. |
| `tests/test_change_benchmark.py` | Evaluation Harness & Taxonomy | 9 | Harness manifest loading, macro vs micro aggregation semantics, failure taxonomy classification. |
| `tests/test_change_api.py` | Change Detection API Endpoint | 6 | `POST /api/v1/analyze` bi-temporal payload routing, visual preview generation, 9-step trace. |
| `tests/test_semantic_fixtures.py`| Semantic Change Interpretation | 5 | Zero-change anti-hallucination gate, structured transition schemas, 10-step trace. |
| `tests/test_cross_modal_alignment.py` | Optical-SAR Ingestion & Grid | 6 | Multi-modality validation, order normalization, disjoint pair rejection, raster alignment. |
| `tests/test_optical_sar.py` | Cross-Modal Joint Analysis | 13 | Dual-modality dependence verification, 6 RS classes, `ComplementarityReport`, 10-step trace. |
| `tests/test_orchestrator.py` | M7 Agentic Orchestrator | 16 | Dynamic intent routing, physical input configuration inspection, 5 multi-stage DAGs. |
| `tests/test_api.py` | HTTP Endpoints & Errors | 8 | Unified `/api/v1/analyze`, `/api/v1/validate`, health checks, error status codes. |
| `tests/test_schemas.py` | Schema Edge Cases | 6 | Strict type coercion, bounds checking, optional field validation. |
| **Total Test Suite** | **All Modules** | **137** | **Zero failures, zero regressions, ~150s execution time.** |

---

## 3. Tier B: Engineering Development Benchmarks

These benchmarks were executed during development milestones to make evidence-based architectural choices:

### B.1 M2.1 Text-Guided Region Grounding Benchmark
- **Dataset / Cut:** VRSBench validation subset (`docs/evaluation/vrsbench_grounding_subset.json`).
- **Sample Count:** 35 held-out remote-sensing scenes with verified referring expressions and bounding boxes.
- **Task:** Text-guided spatial localization (`RS_GROUND`).
- **Evaluated Models:**
  - *Candidate A (Zero-Shot VLM):* `Qwen/Qwen2-VL-2B-Instruct`
  - *Candidate B (Dedicated Detector):* `IDEA-Research/grounding-dino-base`
- **Measured Metrics:**
  - **Grounding DINO Base:** Mean IoU **0.2469**, Median IoU **0.0587**, Success@0.5 **22.86%**, Valid Prediction Rate **85.71%**, Mean Latency **189.9 ms**, Peak VRAM **1,732 MB**.
  - **Qwen2-VL Baseline:** Mean IoU **0.0728**, Median IoU **0.0000**, Success@0.5 **0.00%**, Valid Prediction Rate **60.00%**, Mean Latency **1,740.4 ms**, Peak VRAM **4,288 MB**.
- **Outcome:** Grounding DINO adopted as canonical `RS_GROUND` specialist; Qwen2-VL retained for language generation. Full details in `docs/GROUNDING_MODEL_COMPARISON.md`.

### B.2 M4 Bi-Temporal Change Detection Benchmark
- **Dataset / Cut:** LEVIR-CD held-out validation subset (`docs/evaluation/levir_cd_subset.json`).
- **Sample Count:** 20 pairs ($1024 \times 1024$ px, high-resolution optical).
- **Task:** Binary building change detection (`CHANGE_DETECT`).
- **Evaluated Models:**
  - *Candidate A (Neural Detector):* `TinyCD` (~316k parameters, EfficientNet-B4 backbone).
  - *Candidate B (Baseline / Fallback):* `DeterministicCVASpecialist` (Adaptive Change Vector Analysis).
- **Measured Metrics:**
  - **TinyCD:** Mean IoU **0.8347**, Global IoU **0.8291**, Precision **0.9216**, Recall **0.8985**, F1 **0.9091**, FPR **0.0052**, Latency **92.3 ms**, Peak VRAM **219.4 MB**.
  - **CVA Baseline:** Mean IoU **0.0327**, Global IoU **0.0315**, Precision **0.0913**, Recall **0.0547**, F1 **0.0619**, FPR **0.0346**, Latency **44.6 ms**, Peak VRAM **0.0 MB**.
- **Outcome:** TinyCD established as primary neural specialist; CVA retained as deterministic baseline and Rule 24 graceful fallback. Full details in `docs/evaluation/M4_BENCHMARK_REPORT.md`.

### B.3 M4 Diverse Remote-Sensing Generalization Evaluation
- **Dataset / Cut:** Diverse RS manifest (`docs/evaluation/diverse_rs_manifest.json`).
- **Categories Covered:** Urban, Forest, Agriculture, Water, Coastal, Mining, Infrastructure, Nuisance Stress (Illumination, Coregistration, Seasonal phenology).
- **Findings:** Verified high performance on structural/urban domains, but revealed expected domain sensitivity to agricultural harvest phenology and forest canopy seasonal shifts (diagnosed via `FailureCategory`). Full details in `docs/evaluation/M4_BENCHMARK_REPORT.md`.

### B.4 M5 Semantic Change Interpretation Evaluation
- **Dataset / Cut:** 4 canonical bi-temporal scenarios (Urban expansion, Zero-change baseline, Forest clearing, Agricultural cycling).
- **Specialist:** `ChangeVQASpecialist` (`CHANGE_VQA`).
- **Findings:** Anti-hallucination zero-change gate successfully bypassed VLM on identical pairs; structured transition classes (`bare_land -> residential_construction`) populated accurately. Full details in `docs/evaluation/M5_SEMANTIC_EVALUATION.md`.

### B.5 M6 Optical + SAR Cross-Modal Evaluation
- **Dataset / Cut:** Co-registered optical RGB and SAR GRD fixtures across 6 target classes.
- **Specialist:** `OpticalSARSpecialist` (`OPTICAL_SAR_FUSION`).
- **Findings:** Dual-modality dependence verified (swapping SAR backscatter shifts classification from smooth bare ground to double-bounce built structure; cloud-obscured optical is resolved by SAR backscatter). Full details in `docs/evaluation/M6_OPTICAL_SAR_EVALUATION.md`.

### B.6 M7 Agentic Orchestrator Dynamic Routing Evaluation
- **Dataset / Cut:** 16-case test matrix covering all multimodal input configurations and referring query intents.
- **Engine:** `DynamicRouter`, `ExecutionPlanner`, `PlanExecutor`.
- **Findings:** 100% intent classification accuracy (16/16), multi-stage DAG generation, dynamic parameter propagation (`target_classes`, `focus_regions`), robust single-specialist fallbacks. Full details in `docs/evaluation/M7_AGENTIC_ORCHESTRATION_EVALUATION.md`.

---

## 4. Tier C: Prescribed SIH Benchmark Suite (Scheduled for M10)

Milestone M10 will implement domain adaptation and comprehensive benchmark evaluation across official remote-sensing datasets prescribed by SIH26167:

| Benchmark Dataset | Primary Task | Target Modality | Key Metrics | Target Checkpoint / Adapter | Preprocessing Pipeline | Known Limitations & Mitigation |
|---|---|---|---|---|---|---|
| **VRSBench** | Single-image VQA, Captioning, Grounding | High-res Optical Aerial / Satellite | VQA Acc (Open-ended), BLEU-4, METEOR, CIDEr, IoU@0.5 | LoRA adapter on Vision-Language Model; fine-tuned Grounding DINO | $512 \times 512$ tile normalization, referring expression cleaning | High annotation density requires mini-batch gradient accumulation. |
| **RSVQA** | Low-res & High-res Remote Sensing VQA | Sentinel-2 / Landsat & Aerial | Accuracy, Top-1 Precision, Semantic Macro F1 | LoRA on multimodal RS backbone | Band selection (RGB + NIR), contrast normalization | Low-res split requires resolution-aware spatial pooling. |
| **CDVQA** | Change Detection Vision-Language QA | Bi-Temporal Optical Pairs | Answer Accuracy, Semantic Transition F1, CIDEr | Bi-temporal cross-attention adapter | Co-registration, relative radiometric normalization | Heavy bitemporal token memory requires mixed precision (FP16). |
| **BigEarthNet-v2.0** | Remote Sensing Representation & Cross-Modal Adaptation | Sentinel-2 (Multispectral) + Sentinel-1 (SAR) | Multi-label Mean Average Precision (mAP), Macro F1 | Shared dual-encoder / Cross-modal projection layer | Band calibration, dB backscatter normalization, nodata masking | Large volume (~600k patches); train on curated representative subset (~50k). |

---

## 5. Benchmark Execution & Reporting Protocols

1. **Deterministic Seeds:** All evaluation scripts must set seeds for `random`, `numpy`, and `torch` (`seed=42`).
2. **Telemetry Recording:** Every evaluation report must log:
   - Git commit hash
   - CUDA device name and driver version
   - Peak allocated and reserved VRAM (`torch.cuda.max_memory_allocated`)
   - Per-sample and aggregate latency (mean, median, 95th percentile)
3. **Artifact Archiving:** Detailed per-sample predictions must be serialized to JSON in `docs/evaluation/` alongside summary Markdown documentation.
