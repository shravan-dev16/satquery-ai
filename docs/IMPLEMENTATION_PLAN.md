# SatQuery AI — Engineering Implementation Plan

**Document Version:** 2.7.0 (Milestone M7 Completion)  
**Problem Statement:** Smart India Hackathon — SIH26167 (ISRO / Space Technology)  
**Standard:** Compliance with `AGENTS.md` Rule 19 (Team Ownership), Rule 20 (Task Reporting Format), and Rule 21 (Milestones M0 to M14).

---

## 1. Milestone Status & Progress Tracking

| Milestone ID | Title | Status | Primary Deliverables | Key Verification |
|---|---|---|---|---|
| **M0** | Scaffolding & Environment Layer | **COMPLETED** | Pydantic contracts, BaseSpecialist, ModelRegistry, test fixtures (<1.5 MB), PyTorch CUDA 12.4, Rasterio | 20 unit tests passing in 2.10s |
| **M1-A** | GeoTIFF Ingestion & Metadata Validation | **COMPLETED** | `geotiff.py`, `metadata.py`, `modality.py`, `validation.py`, `POST /api/v1/validate` | Deterministic CRS/band extraction, verified vs heuristic modality detection |
| **M1-B** | Real VQA Smoke Test & Analyze Endpoint | **COMPLETED** | `backend/models/vqa.py`, `POST /api/v1/analyze`, `scripts/smoke_test_vqa.py` | Real GeoTIFF test image inference, GPU memory verification, StandardResultContract |
| **M2** | Text-Guided Grounding & Spatial Projection | **COMPLETED** | `backend/models/grounding.py`, `backend/evidence/spatial.py` | Initial Qwen2-VL candidate, strict coordinate convention `[xmin, ymin, xmax, ymax]` |
| **M2.1** | Grounding Recovery & Empirical Selection | **COMPLETED** | Grounding DINO Base integration, query normalizer, NMS, degeneracy filter | Evaluated on 35-sample VRSBench subset, IoU improved 0.0728 -> 0.2469, 59 tests passing |
| **M3** | Bi-Temporal Change Analysis Foundation | **COMPLETED** | 11-point validator, deterministic aligner, TinyCD + CVA baseline, physical area, 9-step trace | 77 unit/integration tests passing in 3.60s, baseline LEVIR-CD evaluated |
| **M4** | Dedicated Change Detection & Benchmark Evaluation | **COMPLETED** | Reproducible benchmark harness, LEVIR-CD verification, diverse-scene generalization, failure taxonomy | Machine-readable benchmark reports, macro/micro metrics, failure diagnostics, 95 tests passing |
| **M5** | Change Semantic Interpretation & Change VQA | **COMPLETED** | `backend/models/change_vqa.py`, structured transitions, anti-hallucination zero-change gate, 10-step trace | Real model smoke test on 4 scenarios, 108 tests passing, `docs/evaluation/M5_SEMANTIC_EVALUATION.md` |
| **M6** | Optical + SAR Cross-Modal Analysis | **COMPLETED** | `backend/models/optical_sar.py`, `CrossModalValidator`, `CrossModalAligner`, 10-step trace | Verifiable dual-sensor dependence, 6 RS classes, 121 tests passing, `docs/evaluation/M6_OPTICAL_SAR_EVALUATION.md` |
| **M7** | Agentic Orchestrator & Dynamic Routing | **COMPLETED** | `backend/agent/router.py`, `planner.py`, `executor.py`, 16-case test matrix, automatic UI default | Intent classification, physical input configuration inspection, multi-stage DAGs, 137 tests passing, `docs/evaluation/M7_AGENTIC_ORCHESTRATION_EVALUATION.md` |
| **M8** | Evidence Fusion & Consistency Checking | **CURRENT NEXT MILESTONE** | `backend/evidence/consistency.py`, `fusion.py` | Conflict detection (e.g. change detector vs VLM) and confidence penalty |
| **M9** | Defensible Confidence Engine | *Scheduled* | `backend/evidence/confidence.py` | Multi-factor evidence-weighted confidence heuristic calculation |
| **M10** | Remote-Sensing Adaptation & Benchmark Evaluation | *Scheduled* | `backend/evaluation/`, LoRA training on BigEarthNet / open RS data | Defensible RS domain adaptation with held-out validation |
| **M11** | Report Generation (PDF & JSON) | *Scheduled* | `backend/reports/pdf_generator.py` | Exportable PDF with embedded evidence, maps, and trace |
| **M12** | Interactive Analyst UI | *Scheduled* | `frontend/` (Static UI slice mounted at `/ui`) | Map viewer, split-slider, mask overlay, and trace inspector (Vertical slice verified) |
| **M13** | End-to-End Demo Hardening | *Scheduled* | Live rehearsed scripts for Demo 1 to Demo 4 | Flawless sub-3s query execution |

> [!IMPORTANT]
> **Notice on Evaluation Taxonomy:** Milestones M1 and M2 constitute smoke tests for architectural plumbing, affine geometry transformations, and GPU memory safety. They are strictly distinguished from dedicated change benchmark evaluation (**M4**), domain-wide remote-sensing benchmark adaptation (**M10**), and report generation (**M11**).

---

## 2. Milestone M2 Technical Summary: Text-Guided Spatial Region Grounding

### 2.1 Specialist Model Architecture (`RS_GROUND`)
- **Module:** `backend/models/grounding.py` (`RemoteSensingGroundingSpecialist`).
- **Subclasses:** `BaseSpecialist`, registered via `ModelRegistry`.
- **Initial Candidate:** Zero-shot `Qwen/Qwen2-VL-2B-Instruct` (Apache 2.0, natively supported in `transformers`, dynamic resolution, zero-code stability).
- **Coordinate Parser:** `GroundingCoordinateParser` extracts normalized `[ymin, xmin, ymax, xmax]` tokens from model text and maps them deterministically to pixel coordinates.
- **Empty Result Handling:** When referring expressions match no detected features, returns an explicit warning (`"No spatial regions matching referring expression were detected."`) with lowered confidence rather than hallucinating fake boxes.

### 2.2 Strict Coordinate Convention & Deterministic Geospatial Projection
- **Module:** `backend/evidence/spatial.py` (`PixelBoundingBox`, `SpatialTransformer`, `GeospatialRegion`).
- **Coordinate Convention:**
  - Origin: Top-Left $(0, 0)$
  - $x$: Horizontal column index (left-to-right)
  - $y$: Vertical row index (top-to-bottom)
  - Order: strictly **`[xmin, ymin, xmax, ymax]`** across all schemas.
- **Geospatial Projection:** Converts pixel bounding boxes to projected GeoJSON Polygons via the GeoTIFF affine matrix:
  $$\begin{bmatrix} X_{\text{proj}} \\ Y_{\text{proj}} \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
- **Honesty Clause:** All derived polygons are explicitly labeled with `derivation="axis_aligned_bounding_box"`. No bounding box is misrepresented as semantic segmentation.

### 2.3 Grounding Output Contract
Adheres strictly to the structured schema:
- `evidence.boxes`: list of `BoundingBox` (`box_id`, `label`, `confidence`, `coordinates_normalized`, `coordinates_pixel`, `geojson`).
- `evidence.regions`: list of `DetectedRegion` (`region_name`, `label`, `bbox_pixel: [xmin, ymin, xmax, ymax]`, `confidence`, `polygon_pixel: null`).

### 2.4 M2-B Scene Captioning Preparation (`RS_CAPTION`)
- **Module:** `backend/models/caption.py` (`RemoteSensingCaptionSpecialist`).
- Declares capability `RS_CAPTION` with `TaskType.CAPTION`, registered in `ModelRegistry`.
- Implements `BaseSpecialist` interface without generating fake captions, preserving architectural readiness for subsequent captioning milestones.

### 2.5 Provenance Audit & Terminology Corrections
- **Smoke-Test Dataset:** `tests/fixtures/real_rs_sample.tif` is formally labeled:
  *"Rasterio repository GeoTIFF test image used for geospatial pipeline smoke testing."*  
  Source: `https://raw.githubusercontent.com/rasterio/rasterio/master/tests/data/RGB.byte.tif` (BSD 3-Clause).
- **Model Terminology:** `Qwen2-VL-2B-Instruct` is labeled:
  *"initial VQA and grounding specialist candidate using zero-shot Qwen2-VL-2B-Instruct"* (not domain-adapted or fine-tuned).

---

## 3. Definition of Done Checklist for M2

- [x] `RS_GROUND` specialist exists behind `BaseSpecialist`.
- [x] Registered through `ModelRegistry` alongside `RS_VQA` and `RS_CAPTION`.
- [x] Grounding query executes on a real satellite GeoTIFF image.
- [x] Valid pixel-space bounding box returned in strict `[xmin, ymin, xmax, ymax]` format.
- [x] Bounding box deterministically transformed into geospatial GeoJSON Polygon geometry.
- [x] CRS is preserved and documented (`EPSG:32618`).
- [x] The `/api/v1/analyze` endpoint routes grounding queries and returns `StandardResultContract`.
- [x] No fake masks, fake segmentations, or fabricated results exist.
- [x] All 50 unit tests pass in 1.85s (`pytest tests/ -m "not smoke"`).
- [x] No large benchmark dataset downloaded.
- [x] Existing M1 functionality continues to pass all tests.

---

## 4. Milestone M2.1 Technical Summary: Grounding Recovery & Empirical Model Selection

### 4.1 Objective & Candidate Evaluation
To resolve the M2 zero-shot full-image collapse (`[0, 0, width, height]`), Person A evaluated a dedicated vision-language grounding detector candidate against the zero-shot baseline:
- **Candidate A:** `Qwen/Qwen2-VL-2B-Instruct` (Zero-shot VLM baseline).
- **Candidate B:** `IDEA-Research/grounding-dino-base` (Dedicated zero-shot grounding detector candidate).

Both were evaluated on a deterministic 35-sample VRSBench validation referring-expression subset with exact image/annotation coordinate verification.

### 4.2 Measured Results & Selection
- **Grounding DINO Base:** Mean IoU 0.2469, median IoU 0.0587, success@0.5 22.86%, valid prediction rate 85.71%, mean latency 189.9 ms, peak VRAM 1732.2 MB.
- **Qwen2-VL Baseline:** Mean IoU 0.0728, median IoU 0.0000, success@0.5 0.0%, valid prediction rate 60.0%, mean latency 1740.4 ms, peak VRAM 4288.5 MB.
- **Decision:** Grounding DINO is selected as the primary specialist for `RS_GROUND`. Qwen2-VL is retained exclusively for `RS_VQA` semantic reasoning.

### 4.3 Definition of Done Checklist for M2.1
- [x] Grounding DINO candidate integrated via official Hugging Face `transformers` on CUDA.
- [x] Deterministic 35-sample VRSBench evaluation manifest generated (`docs/evaluation/vrsbench_grounding_subset.json`).
- [x] Query normalizer implemented deterministically (`QueryNormalizer.normalize`).
- [x] Degenerate box rule ($\ge 98\%$ area coverage) implemented, tested, and active.
- [x] Non-Maximum Suppression (NMS) active on predicted candidates.
- [x] Comparative evaluation completed and detailed in `docs/GROUNDING_MODEL_COMPARISON.md`.
- [x] Real remote-sensing multi-query smoke test verified on `real_rs_sample.tif` (`docs/SMOKE_TEST_GROUNDING_RESULTS.json`).
- [x] ModelRegistry decoupling preserved (`POST /api/v1/analyze` routes to swappable specialist).
- [x] All 59 unit tests pass in 1.85s (`pytest tests/ -m "not smoke"`).
- [x] Stopped prior to beginning captioning, change detection, optical-SAR, or fine-tuning.

---

## 5. Milestone M3: Bi-Temporal Change Analysis Foundation (COMPLETED)

### 5.1 Objective
Establish an empirical, geospatially verified bi-temporal change detection foundation with 11-point validation, deterministic alignment/co-registration, candidate model comparison (TinyCD vs CVA baseline), physical area estimation, visual preview generation, and auditable 9-step execution trace via `StandardResultContract`.

### 5.2 Evaluated Candidates Behind `CHANGE_DETECT`
- **Candidate B (Baseline):** `DeterministicCVASpecialist` (Spectral difference + adaptive threshold + morphology).
- **Candidate A (Learned Specialist):** `TinyCDSpecialist` (~316k params, Siamese EfficientNet-B4 + mixing attention, pretrained on LEVIR-CD).

### 5.3 Measured Results (Held-Out 20-Sample LEVIR-CD Validation Subset)
- **TinyCD Specialist:** Mean IoU **0.8347**, Precision **0.9216**, Recall **0.8985**, F1 **0.9091**, False-Positive Ratio **0.0052**, Latency **92.3 ms**, Peak VRAM **219.4 MB**.
- **CVA Baseline:** Mean IoU **0.0327**, Precision **0.0913**, Recall **0.0547**, F1 **0.0619**, False-Positive Ratio **0.0346**, Latency **44.6 ms**, Peak VRAM **0.0 MB**.
- **Decision:** TinyCD is selected as primary `CHANGE_DETECT` specialist; CVA retained as deterministic baseline and Rule 24 fallback.

### 5.4 Definition of Done Checklist for M3
- [x] 1. Two valid compatible images ingested via `POST /api/v1/validate` and `POST /api/v1/analyze`.
- [x] 2. 11-point temporal and spatial compatibility verified (`BiTemporalValidator`).
- [x] 3. Deterministic reprojection, resampling, and intersection cropping implemented (`BiTemporalAligner`).
- [x] 4. TinyCD neural detector loaded cleanly with strict state-dict matching (`models/checkpoints/levir_best.pth`).
- [x] 5. Deterministic CVA baseline implemented with adaptive thresholding.
- [x] 6. Both candidates generate valid binary masks and probability maps.
- [x] 7. Scientific change metrics computed against ground truth (`ChangeDetectionMetrics`: IoU, Precision, Recall, F1, FPR).
- [x] 8. Spatial evidence generated: bounding boxes `[xmin, ymin, xmax, ymax]` and projected GeoJSON Polygons.
- [x] 9. Physical area calculated strictly when CRS uses linear meters ($640,000\text{ m}^2$, $64.0\text{ ha}$ on UTM fixture); pixel area with warning otherwise.
- [x] 10. Visual preview artifacts generated and servable: before ($T_1$), after ($T_2$), binary mask, and semi-transparent red overlay.
- [x] 11. API returns results via unified `StandardResultContract`.
- [x] 12. Auditable 9-step observable execution trace returned (`InputValidation`, `TemporalValidation`, `SpatialCompatibility`, `Alignment`, `SpecialistSelection`, `ModelExecution`, `ChangeMaskValidation`, `Statistics`, `EvidenceAssembly`).
- [x] 13. Peak VRAM (219.4 MB) and inference latency (92.3 ms) measured live on RTX 4070.
- [x] 14. 77 fast unit/integration tests pass in 3.60s (`tests/test_alignment.py`, `tests/test_change_metrics.py`, `tests/test_change_detection.py`, `tests/test_change_api.py`).
- [x] 15. All existing M0–M2.1 functionality remains 100% intact.
- [x] 16. Documentation synchronized (`ARCHITECTURE.md`, `MODEL_PLAN.md`, `API_CONTRACT.md`, `IMPLEMENTATION_PLAN.md`, `README.md`).
- [x] 17. No fake results, metrics, confidence, or masks exist.
- [x] **STOPPED** after M3. Did not begin semantic change-VQA, optical-SAR fusion, BigEarthNet adaptation, or full agentic orchestration.

---

## 6. Milestone M4 Implementation Plan: Dedicated Change Detection & Benchmark Evaluation

### 6.1 Objective
Turn the empirical bi-temporal change detection foundation established in M3 into a rigorous, reproducible benchmark and evaluation layer. The objective is NOT to train or fine-tune models (reserved for M10), but to:
1. Objectively evaluate TinyCD relative to the deterministic CVA baseline under identical experimental conditions.
2. Reproduce and verify baseline LEVIR-CD validation results through a standardized benchmark harness rather than hardcoded metrics.
3. Systematically evaluate detector behavior on diverse remote-sensing land-cover categories (urban, forest, agriculture, water, coastal, mining, infrastructure) and under non-change nuisance variations.
4. Establish an empirical failure taxonomy that diagnoses weaknesses to guide subsequent M10 domain adaptation.

### 6.2 Scope Boundaries & Preservations
- **Strict Scope Boundary:** M4 MUST NOT begin change-VQA implementation (M5), optical-SAR cross-modal analysis (M6), agentic DAG routing (M7), evidence fusion (M8), confidence heuristics (M9), or model training / PEFT / LoRA fine-tuning (M10). Those remain strictly scheduled for later milestones.
- **Zero Regression Principle:** All existing M0–M3 capabilities, `/api/v1/analyze`, `StandardResultContract`, visual previews, 9-step execution trace, and the frontend web application mounted at `/ui/` must remain completely intact.

### 6.3 Technical Work Packages

#### A. Reproducible Benchmark Harness (`backend/evaluation/change_benchmark.py`)
- Standardized evaluation entry point supporting `TinyCDSpecialist`, `DeterministicCVASpecialist`, and any `ChangeDetectionSpecialist` candidate.
- Supports deterministic manifest ingestion, per-sample inference, warmup passes, latency profiling (mean, median, min, max), and peak CUDA memory measurement (`torch.cuda.max_memory_allocated`).
- Produces machine-readable JSON output reports containing metadata, environment telemetry, model configuration, per-sample records, and aggregate summary metrics.

#### B. Formal LEVIR-CD Benchmark Reproduction (`docs/evaluation/levir_cd_subset.json`)
- Ingests the deterministic 20-sample held-out LEVIR-CD validation subset ($1024 \times 1024$ px, CC-BY-4.0).
- Runs both TinyCD and CVA through the harness without modifying the sample split.
- Records for every sample: `sample_id`, `filename`, `prediction_validity`, `iou`, `precision`, `recall`, `f1`, `false_positive_ratio`, `true_positive_pixels`, `false_positive_pixels`, `false_negative_pixels`, `true_negative_pixels`, `changed_pixels_pred`, `changed_pixels_gt`, `latency_ms`, and warnings.
- Re-verifies baseline metrics (TinyCD: IoU ~0.8347, F1 ~0.9091; CVA: IoU ~0.0327, F1 ~0.0619) without hardcoded numbers.

#### C. Exact Metric Validation & Aggregation Semantics (`backend/evaluation/metrics.py`)
- Mathematically validates confusion counts and metrics:
  - $\text{IoU} = \frac{TP}{TP + FP + FN}$
  - $\text{Precision} = \frac{TP}{TP + FP}$
  - $\text{Recall} = \frac{TP}{TP + FN}$
  - $\text{F1} = \frac{2 \cdot P \cdot R}{P + R}$
  - $\text{FPR} = \frac{FP}{FP + TN}$
- Explicit edge-case handling:
  - True Negative agreement (both GT and Pred are empty): $\text{IoU}=1.0$, $\text{Precision}=1.0$, $\text{Recall}=1.0$, $\text{F1}=1.0$, $\text{FPR}=0.0$, `is_empty=True`.
  - Zero-change prediction on positive GT: $\text{Recall}=0.0$, $\text{Precision}=0.0$, $\text{IoU}=0.0$, $\text{F1}=0.0$, `is_empty=True`.
  - False alarm prediction on zero-change GT: $\text{Precision}=0.0$, $\text{Recall}=0.0$, $\text{IoU}=0.0$, $\text{F1}=0.0$.
  - Full-image prediction ($\ge 98\%$ pixels changed): `is_full_image=True`.
- **Explicit Aggregation Taxonomy:**
  - **Macro-Averaging:** Arithmetic mean, median, standard deviation, minimum, and maximum computed across per-sample metric values.
  - **Pixel-Global (Micro-Averaging):** Summed $TP, FP, FN, TN$ across all evaluated samples to calculate global dataset-level IoU, Precision, Recall, and F1.

#### D. Geospatial Evidence & Physical Area Validation
- For samples with projected linear-meter CRS (e.g. UTM / EPSG:32643):
  - Validates exact pixel counts against physical surface area in $m^2$ and hectares ($ha$).
  - Preserves affine transform matrices and spatial bounding box bounds.
  - Validates polygon coordinate consistency in strict `[xmin, ymin, xmax, ymax]` ordering.
- For geographic or unprojected samples (e.g. EPSG:4326):
  - Strictly prohibits fabricating physical meter measurements.
  - Returns pixel-space measurements with an explicit warning (`"CRS does not use linear meters; physical area cannot be deterministically computed."`).

#### E. Diverse Remote-Sensing Generalization Evaluation Track (`backend/evaluation/manifests.py`)
- Evaluates the current detector across 7 major land-cover categories and 1 nuisance-variation category:
  1. `URBAN` (Building construction, demolition, settlement expansion)
  2. `FOREST` (Canopy disturbance, logging, deforestation)
  3. `AGRICULTURE` (Crop cycling, harvest vs permanent conversion)
  4. `WATER` (Reservoir fluctuation, inundation, flood extent)
  5. `COASTAL` (Shoreline retreat, sediment deposition, intertidal shift)
  6. `MINING` (Surface excavation, open-pit quarrying, bare earth)
  7. `INFRASTRUCTURE` (Roadways, runways, bridges, commercial logistics)
  8. `NUISANCE_VARIATION` (Seasonal vegetation color change, solar illumination/shadow angle shifts, atmospheric haze, coregistration jitter)
- **Quantitative vs Qualitative Separation:**
  - Samples with verified ground-truth change masks have `is_quantitative = True` and generate full quantitative metrics.
  - Natural complex scenes without verified ground truth have `is_quantitative = False`, execute inference only, record detected cluster counts/pixel area, and are strictly labeled as **qualitative-only** (never fabricating IoU, precision, recall, or F1).

#### F. Structured Failure Taxonomy (`backend/evaluation/failure_taxonomy.py`)
- Standardized failure categories:
  - `missed_change` (Low recall, undetected valid structures)
  - `false_positive` (Low precision, excessive false alarms)
  - `boundary_error` (Discrepancy at building/object boundaries)
  - `small_change_missed` (Small structures below receptive field)
  - `seasonal_variation_false_positive` (Vegetation seasonal shift flagged as change)
  - `shadow_false_positive` (Sun angle / cloud shadow discrepancy)
  - `cloud_related_error` (Cloud reflection or cloud shadow artifacts)
  - `vegetation_error` (Natural growth mistaken for structural change)
  - `water_error` (Surface glint or turbidity changes)
  - `agriculture_error` (Crop phenology mistaken for land conversion)
  - `urban_error` (Non-structural surface reflectance shifts)
  - `alignment_related_error` (Edge halo false alarms from registration jitter)
  - `unclassified_failure` (No single defensible cause attributable)
- Implements a deterministic diagnostic heuristic that classifies sample errors without hallucinating unsupported semantic causes.

#### G. Model Comparison & Verification
- Produces a side-by-side comparative evaluation of `TinyCD` vs `DeterministicCVASpecialist`:
  - Mean & Global IoU, Precision, Recall, F1, FPR.
  - Empty prediction rate, full-image degeneracy rate.
  - Mean and median inference latency (ms).
  - Peak allocated VRAM (MB).
  - Failure category distribution across evaluated samples.
- The evaluation will be conducted without tuning thresholds or parameters to artificially favor either model.

#### H. Documentation & Scientific Grounding
- Create `docs/evaluation/M4_BENCHMARK_REPORT.md` documenting the benchmark methodology, empirical results, failure distributions, and operational implications.
- Clearly state the current system's capabilities: TinyCD is strongly verified on the evaluated LEVIR-CD building change subset, but broad remote-sensing generalization across vegetation, agriculture, and water remains an open adaptation goal for M10.

### 6.4 Definition of Done Checklist for M4
- [x] 1. A reproducible change-detection benchmark harness exists in `backend/evaluation/change_benchmark.py`.
- [x] 2. Both TinyCD and CVA can be evaluated through the same harness with identical inputs.
- [x] 3. LEVIR-CD held-out evaluation is reproducible from manifest `docs/evaluation/levir_cd_subset.json`.
- [x] 4. Exact metrics (IoU, Precision, Recall, F1, FPR) and edge cases are mathematically validated.
- [x] 5. Macro-averaging and pixel-global (micro) metrics are explicitly distinguished and reported.
- [x] 6. Per-sample and aggregate metrics are saved in machine-readable JSON format (`docs/evaluation/m4_levir_cd_benchmark_report.json`, `docs/evaluation/m4_diverse_rs_evaluation_report.json`).
- [x] 7. Latency (ms) and peak VRAM (MB) are measured and recorded consistently.
- [x] 8. Physical area calculations are verified, CRS-aware, and never fabricated for non-meter CRS.
- [x] 9. Diverse remote-sensing generalization evaluation track is established across 8 categories (`docs/evaluation/diverse_rs_manifest.json`).
- [x] 10. Quantitative benchmark samples and qualitative-only samples are strictly distinguished (no fake IoU).
- [x] 11. Structured failure taxonomy classifies empirical failure modes without semantic hallucination.
- [x] 12. Unit and integration tests cover benchmark loading, metric calculation, edge cases, area, and taxonomy (`tests/test_change_benchmark.py`).
- [x] 13. All 95 unit and integration tests pass across the repository with zero regressions.
- [x] 14. Existing frontend UI at `/ui/` and `/api/v1/analyze` remain completely functional.
- [x] 15. Documentation synchronized (`IMPLEMENTATION_PLAN.md`, `MODEL_PLAN.md`, `README.md`, `docs/evaluation/M4_BENCHMARK_REPORT.md`).
- [x] 16. M4 does NOT begin model training, fine-tuning, or M5–M10 milestones.
- [x] 17. M4 concludes with a clear statement of verified strengths and unadapted limitations.
- [x] **STOPPED** after M4. Did not begin semantic change-VQA, optical-SAR fusion, BigEarthNet adaptation, or full agentic orchestration.

---

## 7. Milestone M5 Technical Summary: Change Semantic Interpretation & Change VQA

### 7.1 Objective & Architecture (`CHANGE_VQA`)
Milestone M5 delivers the grounded vision-language semantic interpretation layer for bi-temporal remote sensing. It answers:
- **WHAT changed?** (e.g. `bare_land -> residential_construction`, `forest -> residential_construction`)
- **WHERE did it change?** (Linked to verified spatial bounding boxes and cluster regions from `CHANGE_DETECT`)
- **HOW did it change?** (`temporal_direction`: `increased`, `decreased`, `modified`, `no_change`, `uncertain`)
- **WHAT evidence supports the interpretation?** (Visual crop comparison, mask overlap, pixel statistics)

### 7.2 Core Modules & Design Principles
1. **Pydantic Schema Extensions (`backend/agent/schema.py`):**
   - `SemanticTransition`: `transition_id`, `from_class`, `to_class`, `description`, `region_id`, `confidence_support`, `is_uncertain`.
   - `SemanticChangeInterpretation`: `temporal_direction`, `predominant_transition`, `transitions`, `semantic_uncertainty`, `summary_answer`, `change_driver`.
   - Integrated into `EvidenceBundle.semantic_interpretation`.
2. **Specialist Model Implementation (`backend/models/change_vqa.py`):**
   - `ChangeVQASpecialist(BaseSpecialist)` declared with capability `CHANGE_VQA` and registered in `ModelRegistry`.
   - Reuses `Qwen/Qwen2-VL-2B-Instruct` in an evidence-conditioned composite visual workflow:
     - Constructs a 3-panel composite: `Time 1 (Before) | Time 2 (After) | Change Overlay`.
     - Injects spatial bounding boxes, pixel change statistics, and cluster summaries into a structured system prompt.
     - Enforces JSON output schema with defensive markdown code-block parsing.
3. **Scientific Honesty & Anti-Hallucination Guarantees:**
   - **Zero-Change Short-Circuit:** When `changed_pixels == 0`, the specialist immediately bypasses VLM generation and returns `temporal_direction="no_change"`, `predominant_transition="none"`, and zero transitions. Hallucination on identical scenes is mathematically impossible.
   - **Defensive Tagging:** Ambiguous land-cover classes default to `"unknown"` or `"uncertain"` with explicit advisory warnings (`"Change detector evidence is localized/limited; semantic interpretation should be treated as provisional."`).
   - **Defensible Confidence Decoupling:** Model uncertainty from `CHANGE_VQA` is recorded purely as `semantic_uncertainty` within the semantic interpretation schema. It does NOT overwrite or contaminate the system evidence confidence score, ensuring strict isolation from the future M9 confidence engine.
4. **Observable 10-Step Execution Trace:**
   - When the user query asks a semantic question (e.g., *"What changed?"*, *"Describe the land cover change"*), the pipeline inserts Step 9: `SemanticInterpretation`, producing an auditable 10-step trace.
   - Pure geometric queries (e.g., *"Detect changes"*) preserve the exact 9-step M3/M4 trace with zero regression.
5. **Interactive UI Integration (`frontend/`):**
   - Updated pipeline stepper to 10 stages (`Semantic VQA`).
   - Added interactive `.semantic-card` with direction badge, predominant transition tag, structured transition items, and semantic uncertainty indicator.

### 7.3 Definition of Done Checklist for M5
- [x] 1. `ChangeVQASpecialist` implemented in `backend/models/change_vqa.py` behind `BaseSpecialist`.
- [x] 2. Capability `CHANGE_VQA` registered with `ModelRegistry`.
- [x] 3. `SemanticTransition` and `SemanticChangeInterpretation` schemas integrated into `EvidenceBundle`.
- [x] 4. Evidence-conditioned composite preview (`T1 | T2 | Overlay`) passed to VLM.
- [x] 5. Zero-change anti-hallucination gate verified (0 changed pixels bypasses VLM generation).
- [x] 6. Defensive handling of ambiguous classes (`"unknown"` / advisory warnings).
- [x] 7. 10-step auditable execution trace verified on semantic queries.
- [x] 8. Strict non-regression: 9-step trace preserved on pure detection queries.
- [x] 9. Defensible confidence separation maintained (semantic uncertainty separated from system evidence score).
- [x] 10. Frontend UI updated with semantic stepper, direction badges, and transition cards.
- [x] 11. Real-model smoke test executed across 4 scenarios (`docs/evaluation/M5_SEMANTIC_EVALUATION.md`).
- [x] 12. Full test suite passes: 108 passed, 0 failed across entire repository.
- [x] 13. M4 benchmark artifacts and metrics preserved byte-for-byte.
- [x] 14. **STOPPED** after M5. Did not begin M6 (Optical+SAR), M7 (Agentic DAG), M8 (Fusion), M9 (Confidence Engine), or M10 (LoRA/Fine-tuning).

---

## 8. Milestone M6 Technical Summary: Optical + SAR Cross-Modal Joint Analysis

### 8.1 Objective & Specialist Architecture (`OPTICAL_SAR_FUSION`)
Milestone M6 delivers the mandatory cross-modal analysis capability combining co-registered optical/multispectral and synthetic aperture radar (SAR) Earth observation imagery:
- **Module:** `backend/models/optical_sar.py` (`OpticalSARSpecialist`).
- **Capability:** `OPTICAL_SAR_FUSION`, registered in central `ModelRegistry`.
- **Supported Tasks:** `TaskType.OPTICAL_SAR_ANALYSIS`.
- **Supported Modalities:** `[ModalityType.OPTICAL, ModalityType.SAR, ModalityType.CROSS_MODAL]`.
- **Supported Input Count:** Exactly 2 rasters.

### 8.2 Input Contract, Validation & Alignment
1. **`CrossModalValidator` (`backend/preprocessing/alignment.py`):**
   - Enforces exactly one optical and one SAR raster.
   - Rejects identical-modality pairs (dual optical or dual SAR).
   - Enforces valid CRS and affine transforms on both rasters.
   - Rejects disjoint spatial coverage or insufficient spatial overlap (< 20%).
   - Detects and transparently normalizes reversed input ordering (SAR primary, optical secondary) with audit advisory warnings.
2. **`CrossModalAligner` (`backend/preprocessing/alignment.py`):**
   - Deterministically co-registers SAR raster to optical cropped spatial grid.
   - Preserves independent optical (RGB) and SAR (polarimetric backscatter) band counts and dtypes.

### 8.3 Raster-Level Joint Feature Fusion Layer
Fuses complementary spectral and physical features into a shared representation:
- **Optical Spectral Cues:** Normalized RGB reflectance, Excess Green Index ($\text{ExG}$), lightness, and blue/red color ratios.
- **SAR Physical Cues:** Calibrated backscatter intensity, $5\times 5$ local standard deviation texture roughness ($\sigma_{\text{local}}$), corner reflection (double bounce), and specular reflectance.
- **Supported Classes (6 Remote-Sensing Classes):**
  1. `water_body`: Low optical brightness / blue tint + near-zero SAR backscatter ($\le 0.20$) + low texture roughness ($\le 0.08$).
  2. `built_structure`: Geometric optical contrast + high SAR backscatter / double bounce ($\ge 0.50$ or $\ge 0.30$ with roughness $\ge 0.12$).
  3. `vegetation_or_cropland`: High optical greenness ($\text{ExG} \ge 0.06$) + moderate diffuse SAR volume scattering ($0.15 - 0.75$).
  4. `bare_ground_or_soil`: Warm tan/brown reflectance ($R > G > B$) + low-to-moderate backscatter ($0.15 - 0.50$) without high roughness.
  5. `road_or_infrastructure`: Neutral grey reflectance + low smooth pavement backscatter ($\le 0.25$, $\sigma \le 0.08$).
  6. `unknown`: Ambiguous or conflicting cross-modal signatures.

### 8.4 Query-Influenced Filtering & Evidence Assembly
- **Query Targeting:** Automatically identifies target classes (e.g., *"identify built-up and water-covered regions"* $\rightarrow$ `built_structure`, `water_body`) and tailors analytical summary narrative.
- **Visual Evidence:** Standardized 3-panel composite preview (`Panel 1: Optical RGB | Panel 2: SAR Grayscale | Panel 3: Joint Fused Land-Cover Map`) and color-coded categorical mask.
- **Structured Evidence:** Populates `EvidenceBundle` with bounding boxes, spatial detected regions with detailed `optical_evidence`, `sar_evidence`, and `joint_evidence`, class-wise zonal statistics, and a dedicated `ComplementarityReport`.
- **Observable 10-Step Trace:**
  `InputValidation` $\rightarrow$ `ModalityValidation` $\rightarrow$ `SpatialCompatibility` $\rightarrow$ `Alignment` $\rightarrow$ `SpecialistSelection` $\rightarrow$ `OpticalFeatureExtraction` $\rightarrow$ `SARFeatureExtraction` $\rightarrow$ `JointFusion` $\rightarrow$ `RegionExtraction` $\rightarrow$ `EvidenceAssembly`.

### 8.5 Definition of Done Checklist for M6
- [x] 1. Optical + SAR pair accepted and validated via `CrossModalValidator`.
- [x] 2. Reversed modality ordering handled transparently with audit trace warning.
- [x] 3. Dual optical and dual SAR pairs rejected with clean structured errors.
- [x] 4. Missing CRS and disjoint image pairs rejected deterministically.
- [x] 5. Independent channels and dtypes preserved via `CrossModalAligner`.
- [x] 6. Real raster-level joint feature fusion implemented in `OpticalSARSpecialist`.
- [x] 7. Dual-modality dependence verified: output changes when SAR changes AND when optical changes.
- [x] 8. 6 remote-sensing semantic classes supported without forced classification.
- [x] 9. Natural-language query influences targeting and narrative answer.
- [x] 10. `ComplementarityReport` populated with optical limitations, SAR penetration, and structural contrast.
- [x] 11. Visual 3-panel composite and segmentation mask generated and linked to evidence.
- [x] 12. `OpticalSARSpecialist` registered as `OPTICAL_SAR_FUSION` in `ModelRegistry`.
- [x] 13. API integrated via Branch C in `/api/v1/analyze` exposing 10-step auditable execution trace.
- [x] 14. Deterministic M6 fixtures created in `backend/evaluation/optical_sar_fixtures.py`.
- [x] 15. Comprehensive test suite added in `tests/test_optical_sar.py` (13 tests passing).
- [x] 16. Baseline evaluation report created in `docs/evaluation/M6_OPTICAL_SAR_EVALUATION.md`.
- [x] 17. Full regression test suite passes (all 108 existing tests + 13 new M6 tests = 121 tests passing).
- [x] 18. **STOPPED** after M6. Did not begin M7 (Agentic Orchestration & DAGs), M8 (General Evidence Fusion), M9 (Confidence Engine), or M10 (LoRA/Fine-tuning).

---

## 9. Milestone M7 Technical Summary: Agentic Orchestration & Dynamic Routing (COMPLETED)

### 9.1 Objective & Architectural Mandate
The SIH26167 problem statement mandates:
*"The system must automatically select, sequence, and execute the appropriate specialist models or tools according to the query and input configuration."*

Milestone M7 delivers a deterministic, evidence-grounded agentic orchestration subsystem that:
1. Replaces manual pipeline selection with query intent understanding and physical raster inspection.
2. Constructs executable multi-stage Directed Acyclic Graphs (DAGs) rather than relying on brittle keyword matching or ungrounded LLM tool-calling.
3. Automatically chains specialists with evidence passing (e.g. `CHANGE_DETECT` spatial masks forwarded into `CHANGE_VQA` semantic prompts).
4. Generates an observable, auditable execution trace exposing every intermediate step, execution time, and confidence.

### 9.2 Core Architecture & Modules
- **`backend/agent/router.py` (`DynamicRouter`):**
  - Synthesizes raster input characteristics (count, CRS, spatial overlap, modalities) and referring query text.
  - Classifies user intent into `TaskIntent`:
    - `SINGLE_IMAGE_VQA`
    - `SINGLE_IMAGE_GROUNDING`
    - `BITEMPORAL_CHANGE_DETECTION`
    - `BITEMPORAL_CHANGE_VQA`
    - `OPTICAL_SAR_ANALYSIS`
    - `UNSUPPORTED`
  - Extracts targeted semantic concepts (`target_classes`, referring objects) for downstream specialist conditioning.
- **`backend/agent/planner.py` (`ExecutionPlanner`):**
  - Translates `TaskIntent` into an executable `ExecutionPlan` consisting of sequenced `PlanStep` nodes.
  - Implements 5 canonical plan templates:
    1. `single_vqa_plan`: `InputValidation` $\rightarrow$ `RS_VQA` $\rightarrow$ `EvidenceAssembly`
    2. `single_grounding_plan`: `InputValidation` $\rightarrow$ `RS_GROUND` $\rightarrow$ `SpatialProjection` $\rightarrow$ `EvidenceAssembly`
    3. `bitemporal_change_detect_plan`: `InputValidation` $\rightarrow$ `BiTemporalAlignment` $\rightarrow$ `CHANGE_DETECT` $\rightarrow$ `EvidenceAssembly`
    4. `bitemporal_change_vqa_plan`: `InputValidation` $\rightarrow$ `BiTemporalAlignment` $\rightarrow$ `CHANGE_DETECT` $\rightarrow$ `CHANGE_VQA` $\rightarrow$ `EvidenceAssembly`
    5. `optical_sar_plan`: `InputValidation` $\rightarrow$ `CrossModalAlignment` $\rightarrow$ `OPTICAL_SAR_FUSION` $\rightarrow$ `EvidenceAssembly`
- **`backend/agent/executor.py` (`PlanExecutor`):**
  - Executes DAG steps deterministically.
  - Manages context state, passing spatial masks, changed bounding boxes, and pixel statistics from detection stages to semantic VLM stages.
  - Records step-by-step latency, status (`success`, `warning`, `skipped`, `failed`), and parameters into the auditable `ExecutionTrace`.
- **API Integration (`backend/main.py`):**
  - Updated `POST /api/v1/analyze` to set `pipeline="auto"` as default.
  - Preserved backward compatibility for explicit legacy pipeline requests (`vqa`, `grounding`, `change`, `optical_sar`).

### 9.3 Definition of Done Checklist for M7
- [x] 1. `DynamicRouter` implemented in `backend/agent/router.py` with raster configuration and query intent classification.
- [x] 2. `ExecutionPlanner` implemented in `backend/agent/planner.py` supporting 5 multi-stage DAG plan templates.
- [x] 3. `PlanExecutor` implemented in `backend/agent/executor.py` with state passing and auditable trace logging.
- [x] 4. Multi-stage execution validated: `CHANGE_DETECT` $\rightarrow$ `CHANGE_VQA` evidence forwarding verified.
- [x] 5. Default pipeline changed to `"auto"` in `POST /api/v1/analyze` without breaking manual pipeline overrides.
- [x] 6. 16-case test matrix implemented in `tests/test_orchestrator.py` verifying 100% routing accuracy.
- [x] 7. Evaluation report published in `docs/evaluation/M7_AGENTIC_ORCHESTRATION_EVALUATION.md`.
- [x] 8. Full test suite passes: 137 passed, 0 failed across the entire repository.
- [x] 9. Zero regressions across M0–M6 capabilities and contracts.
- [x] 10. **STOPPED** after M7. Did not implement M8 (Evidence Fusion & Consistency), M9 (Defensible Confidence Engine), M10 (RS Adaptation), M11 (Reports), M12 (UI), or M13 (Hardening).


