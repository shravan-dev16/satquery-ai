# SatQuery AI — Specialist Model & AI Engineering Plan

**Document Version:** 2.4.0 (Milestone M5 Completion)  
**Project:** SatQuery AI (SIH26167 / ISRO)  
**Host Hardware:** NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM, CUDA 12.4, Windows 11)  
**Core Directives:** Compliance with `AGENTS.md` Rule 4 (RS Specialization), Rule 5 (Model Registry), Rule 16 (Compute Constraints), and Rule 23 (No Fake Implementations).

---

## 1. Terminology & Evaluation Phase Taxonomy

To maintain scientific integrity and comply with SIH26167 guidelines, SatQuery AI strictly distinguishes three distinct operational phases:

1. **Smoke Test (Current Stage — M1 & M2):**
   - **Purpose:** Architectural verification, pipeline plumbing, tensor dimensions, affine matrix projections, API contract compliance, and GPU memory safety.
   - **Data:** Synthetic raster fixtures and verified smoke-test rasters (`tests/fixtures/real_rs_sample.tif`).
   - **Claims:** Verifies system execution and engineering plumbing. **Makes no claims of benchmark accuracy, domain adaptation, or production detection performance.**

2. **Benchmark Evaluation (Scheduled for M11):**
   - **Purpose:** Rigorous quantitative evaluation across standardized open remote-sensing benchmark datasets (RSVQA, CDVQA, VRSBench, LEVIR-CC).
   - **Data:** Frozen test splits with documented ground-truth labels and bounding boxes.
   - **Metrics:** Top-1 VQA accuracy, Bounding Box mIoU, BLEU/CIDEr caption scores, F1/OA change detection metrics.

3. **Remote-Sensing Adaptation (Scheduled Post-M4):**
   - **Purpose:** Domain-specific adaptation using open remote-sensing datasets (BigEarthNet-S2, RSVQAxBEN, or VRSBench).
   - **Methods:** Parameter-Efficient Fine-Tuning (PEFT / LoRA) or specialist head fine-tuning within the 12 GB VRAM constraint.
   - **Terminology Rule:** `Qwen2-VL-2B-Instruct` is currently an unadapted general VLM. It is strictly referred to as the **"initial VQA and grounding specialist candidate using zero-shot Qwen2-VL-2B-Instruct."** It is NOT remote-sensing adapted or fine-tuned at this stage.

---

## 2. Provenance Audit of Real Smoke-Test Image

- **Fixture Path:** `tests/fixtures/real_rs_sample.tif`
- **Designation:** **"Rasterio repository GeoTIFF test image used for geospatial pipeline smoke testing."**
- **Exact Provenance URL:** `https://raw.githubusercontent.com/rasterio/rasterio/master/tests/data/RGB.byte.tif`
- **Origin / Upstream Repository:** Mapbox / Rasterio open-source geospatial test suite.
- **License:** BSD 3-Clause License.
- **Verification Note:** While the metadata indicates UTM Zone 18N (`EPSG:32618`) and 30m spatial resolution resembling Landsat 7 ETM+ Band 4-3-2 composites, until exact USGS scene IDs and acquisition timestamps are independently cross-referenced with the USGS EarthExplorer catalog, it is formally labeled strictly by its verified source.
- **Image Characteristics:**
  - Dimensions: $791 \times 718$ pixels (3 bands, byte datatype)
  - Coordinate Reference System: `EPSG:32618` (UTM Zone 18N)
  - Bounding Box: `(101985.0, 2611485.0, 339315.0, 2826915.0)`
  - Ground Sample Distance: 30.0 meters/pixel

---

## 3. M2 Specialist Selection: Text-Guided Region Grounding (`RS_GROUND`)

### 3.1 Candidate Model Evaluation
- **Primary Candidate:** `Qwen/Qwen2-VL-2B-Instruct`
  - **Developer:** Alibaba Cloud / Qwen Team
  - **Parameters:** ~2.21 Billion parameters
  - **License:** Apache 2.0 (Permissive open-source)
  - **Grounding Capability:** Supports referring expression grounding via formatted JSON/token output (`box_2d: [ymin, xmin, ymax, xmax]` normalized to $[0, 1000]$ or `<|box_start|>(ymin, xmin),(ymax, xmax)<|box_end|>`).
  - **Status:** Selected for M2 smoke testing.
- **Evaluated Alternative:** `microsoft/Florence-2-base`
  - **Findings:** `Florence-2-base` contains hardcoded legacy token references (`forced_bos_token_id`) that fail with `AttributeError` under modern `transformers>=5.x`. Until upstream issues are patched or isolated in a separate container, `Qwen2-VL-2B-Instruct` serves as the primary native candidate.

### 3.2 Strict Coordinate Convention
To prevent coordinate inversion bugs and ensure mathematical determinism across all modules, SatQuery AI enforces **ONE single internal coordinate convention**:

| Parameter | Specification | Description |
|---|---|---|
| **Origin $(0, 0)$** | Top-Left Corner | Raster pixel $(0, 0)$ corresponds to top-left corner |
| **X-axis** | Horizontal (Left-to-Right) | Column index: $0 \le x \le \text{width}$ |
| **Y-axis** | Vertical (Top-to-Bottom) | Row index: $0 \le y \le \text{height}$ |
| **Bbox Ordering** | **`[xmin, ymin, xmax, ymax]`** | Strictly minimums followed by maximums |
| **Normalized Range** | $[0.0, 1.0]$ | $x_{\text{norm}} = x / \text{width}$, $y_{\text{norm}} = y / \text{height}$ |

> [!CAUTION]
> **Strict Convention Rule:** Never mix `[ymin, xmin, ymax, xmax]` with `[xmin, ymin, xmax, ymax]`. Any external candidate model that outputs alternate coordinate orderings (such as Qwen2-VL's normalized `[ymin, xmin, ymax, xmax]`) is explicitly transformed in `GroundingCoordinateParser` immediately upon extraction. Internally and in all API contracts, only `[xmin, ymin, xmax, ymax]` exists.

### 3.3 Deterministic Pixel-to-Geospatial Transformation
Implemented in `backend/evidence/spatial.py`:
- Converts valid pixel bounding boxes to geospatial coordinates using the GeoTIFF affine transform matrix:
  $$\begin{bmatrix} X \\ Y \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
- Produces a closed 5-point GeoJSON Polygon ring: $[TL, TR, BR, BL, TL]$.
- **Honesty Rule:** Geometries derived from bounding boxes are explicitly flagged as `is_axis_aligned_polygon=True` and `derivation="axis_aligned_bounding_box"`. **The system never creates fake segmentation masks or mislabels bounding box polygons as semantic segmentations.**
- Preserves source CRS metadata (e.g. `EPSG:32618`).

### 3.4 Grounding Output Contract
Adheres strictly to the structured schema:
```json
{
  "regions": [
    {
      "region_name": "water body",
      "label": "water body",
      "bbox_pixel": [0.0, 0.0, 791.0, 718.0],
      "confidence": 0.85,
      "polygon_pixel": null
    }
  ]
}
```

---

## 4. M2-B Scene Captioning Preparation (`RS_CAPTION`)

- Created `RemoteSensingCaptionSpecialist` in `backend/models/caption.py` subclassing `BaseSpecialist`.
- Declared capability with `TaskType.CAPTION` and identifier `RS_CAPTION`.
- Registered with `ModelRegistry`.
- Returns an explicit initialization notice without generating fake captions, preserving architectural readiness for subsequent captioning evaluation without compromising M2 grounding quality.

---

## 5. Live GPU Smoke-Test Measurements (M2)

*(Measurements recorded live on local NVIDIA GeForce RTX 4070 SUPER, CUDA 12.4)*

| Metric | Measured VQA (M1) | Measured Grounding (M2) | Target Criterion | Status |
|---|---|---|---|---|
| **Model Initialization Time** | ~14,500 ms (warm: ~10,800 ms) | 10,831 ms | Informational | One-time cold start |
| **Inference Latency** | 4,671 ms | **1,977 ms** | $\le 3000\text{ ms}$ (warm) | Compliant |
| **VRAM Allocated (Load)** | 4,213 MB | 4,213 MB | $\le 6000\text{ MB}$ | Compliant |
| **Peak VRAM Allocated** | 4,367 MB | **4,367 MB** | $\le 7500\text{ MB}$ | Compliant (Leaves > 7.6 GB free) |
| **Peak VRAM Reserved** | 4,616 MB | 4,622 MB | $\le 8000\text{ MB}$ | Compliant |
| **VRAM After Cleanup** | 8.12 MB | **8.12 MB** | Clean deallocation | Verified zero memory leaks |
| **Grounded Boxes Produced** | N/A (VQA narrative) | 1 box / GeoJSON Polygon | Valid spatial evidence | Verified |

---

## 7. M2.1 Grounding Recovery & Empirical Model Evaluation

### 7.1 Candidates Evaluated Behind `RS_GROUND`
To resolve the M2 zero-shot full-image collapse (`[0, 0, width, height]`), two candidates were empirically evaluated on a deterministic 35-sample VRSBench validation referring-expression subset:
- **Candidate A (Baseline):** `Qwen/Qwen2-VL-2B-Instruct` (Zero-shot general VLM baseline, unadapted).
- **Candidate B (Detector):** `IDEA-Research/grounding-dino-base` (Dedicated zero-shot grounding detector candidate, 232M parameters, unadapted).

> [!IMPORTANT]
> **Scientific Integrity Rule:** Neither Qwen2-VL nor Grounding DINO is remote-sensing adapted or fine-tuned at this stage. Both are evaluated in their official permissive open-source configurations. Furthermore, VRSBench represents public overhead aerial RGB imagery; it provides external public empirical evidence for grounding capability and does not substitute for testing on proprietary ISRO Cartosat or multi-spectral/SAR payloads.

### 7.2 Measured Comparative Performance (VRSBench 35-Sample Subset)

| Evaluation Metric | Qwen2-VL-2B-Instruct (Baseline) | Grounding DINO Base (Dedicated Detector) | Improvement / Delta |
|---|---|---|---|
| **Architecture Role** | Zero-shot Multimodal VLM | Dedicated Zero-Shot Grounding Detector | Specialized Vision-Language Detector |
| **Model Checkpoint** | `Qwen/Qwen2-VL-2B-Instruct` | `IDEA-Research/grounding-dino-base` | Permissive Open Source |
| **Parameters** | ~2.21 Billion | ~232 Million | **9.5x smaller footprint** |
| **Valid Prediction Rate** | 60.0% | **85.71%** | +25.7% |
| **Mean IoU** | 0.0728 | **0.2469** | **+0.1741 (3.4x higher)** |
| **Median IoU** | 0.0000 | **0.0587** | +0.0587 |
| **Success@IoU 0.5** | 0.0% | **22.86%** | **+22.9%** |
| **Degenerate Full-Image Rate** | **11.43%** (Failure Mode) | **8.57%** (Safe) | **Filtered by rule** |
| **Empty Result Rate** | 40.0% | 14.29% | Filtered noise |
| **Mean Inference Latency** | 1740.4 ms | **189.9 ms** | **9.2x faster inference** |
| **Peak VRAM Allocated** | 4288.48 MB | **1732.17 MB** | **2556.3 MB less VRAM** |

### 7.3 Query Normalization & Degeneracy Rules
1. **Query Normalizer (`QueryNormalizer.normalize`):** Deterministically transforms user queries into detector entity phrases (e.g. `"Where is the water body?"` $\to$ `"water body."`) while retaining the original referring expression in the execution trace.
2. **Degeneracy Rule ($\ge 98\%$ area coverage):** Bounding boxes covering $\ge 98\%$ of image area are flagged as `is_degenerate=True` with reason `full_image_coverage_...` and excluded from valid API spatial evidence.
3. **Non-Maximum Suppression (NMS):** Overlapping duplicate bounding boxes are suppressed with an IoU threshold of $0.70$.
4. **Architectural Recommendation:** Grounding DINO Base is selected as the primary specialist for `RS_GROUND`. Qwen2-VL-2B-Instruct is preserved exclusively for `RS_VQA` semantic reasoning. Both remain dynamically swappable behind `BaseSpecialist` and `ModelRegistry`.

---

## 8. M3 Bi-Temporal Change Detection Evaluation & Specialist Selection

### 8.1 Evaluated Candidates Behind `CHANGE_DETECT`
To establish an empirical, geospatially verified bi-temporal change detection foundation, two candidates were evaluated:
- **Candidate B (Baseline):** `DeterministicCVASpecialist` (Change Vector Analysis with adaptive std-dev threshold $T = \mu_D + 1.5\sigma_D$ and morphological filtering).
- **Candidate A (Learned Specialist):** `TinyCDSpecialist` (~316k parameters, Siamese EfficientNet-B4 backbone + spatial-spectral mixing attention, pretrained on LEVIR-CD, PyTorch 2.6).

### 8.2 Evaluation Context Classification
1. **Synthetic Pipeline Test:** Executed on `time1_pre_change.tif` and `time2_post_change.tif` (UTM EPSG:32643, 10 m resolution) with a known $80 \times 80$ px ($6,400\text{ px}$, $64.0\text{ ha}$) building block. Confirmed 100% pipeline and physical area integrity.
2. **Public Benchmark Evaluation:** Executed on held-out 20-sample validation subset of LEVIR-CD ($1024 \times 1024$ pixels, CC-BY-4.0) documented in `docs/evaluation/levir_cd_subset.json`.
3. **Future ISRO/SAC Evaluation:** Planned for M11+ using open Cartosat/Resourcesat/RISAT bi-temporal observations.

### 8.3 Measured Benchmark Performance (LEVIR-CD 20-Pair Subset)

| Evaluation Metric | Deterministic CVA Baseline | TinyCD Learned Specialist | Empirical Delta |
|---|---|---|---|
| **Architecture** | Spectral Euclidean Diff + Adaptive Threshold | Siamese EfficientNet-B4 + Mixing Attention | Deep Structural Discrepancy |
| **Parameters** | 0 (Rule-based) | **~316,000** | Lightweight Neural Network |
| **Checkpoint** | None | `models/checkpoints/levir_best.pth` | Verified State Dict |
| **Mean IoU** | 0.0327 | **0.8347** | **+0.8020 (25.5x higher)** |
| **Mean Precision** | 0.0913 | **0.9216** | **+0.8303** |
| **Mean Recall** | 0.0547 | **0.8985** | **+0.8438** |
| **Mean F1 Score** | 0.0619 | **0.9091** | **+0.8472 (14.7x higher)** |
| **False Positive Ratio** | 0.0346 | **0.0052** | **-0.0294 (6.7x lower false alarms)** |
| **Empty Prediction Rate** | 0.0% | **0.0%** | Reliable Detection |
| **Full-Image Prediction Rate**| 0.0% | **0.0%** | Zero Degeneracy |
| **Mean Inference Latency** | **44.6 ms** | **92.3 ms** | Extremely responsive (< 100 ms) |
| **Peak VRAM Allocated** | **0.0 MB** (CPU) | **219.4 MB** (CUDA) | **Only 2.9% of working budget** |

### 8.4 Architectural Decision
1. **Primary Selection:** TinyCD is selected as the primary specialist for `CHANGE_DETECT` in `ModelRegistry`.
2. **Deterministic Baseline Retention:** CVA is preserved as an auditable fallback (Rule 24) when non-photographic or synthetic fixtures are submitted.
3. **Physical Area Rule:** Physical area ($m^2$ and hectares) is computed strictly when the CRS uses linear meters (UTM/EPSG:3857); otherwise pixel statistics are returned with an explicit warning.

---

## 9. Milestone M4 Benchmark Evaluation & Cross-Domain Generalization

### 9.1 Evaluation Objectives & Framework
Milestone M4 converted the M3 foundation into an automated, machine-readable evaluation suite in `backend/evaluation/` (`metrics.py`, `manifests.py`, `failure_taxonomy.py`, `change_benchmark.py`).

### 9.2 Reproducible LEVIR-CD Formal Benchmark (20 Pairs)
- **Harness Confirmation:** TinyCD achieved **Macro Mean IoU 0.8347**, **Macro Mean F1 0.9091**, and **Pixel-Global Micro F1 0.9195** across 20.97 million evaluated pixels on the held-out validation manifest (`docs/evaluation/levir_cd_subset.json`).
- **Deterministic Baseline Comparison:** `DeterministicCVASpecialist` scored **Macro Mean IoU 0.0327** and **F1 0.0619**, demonstrating that simple radiometric difference is incapable of discerning genuine structural additions.
- **Latency & Compute:** Measured live on local NVIDIA RTX 4070 (CUDA 12.4): TinyCD executed in **95.5 ms** mean latency and **219.4 MB** peak allocated VRAM.

### 9.3 Diverse Remote-Sensing Generalization Track (12 Pairs)
Evaluated across 8 operational categories (`docs/evaluation/diverse_rs_manifest.json`):
1. **Urban (`diverse_urban_001`):** TinyCD achieved **IoU 0.8846** and **F1 0.9388** (1,140 px detected), demonstrating high spatial specificity on building additions.
2. **Non-Urban Natural Land-Cover (Forest, Agriculture, Water, Coastal, Mining):** TinyCD exhibited low recall (missed changes, 0 px predicted on forest clearing and agricultural plowing) due to pre-training domain bias toward rectangular building structures.
3. **Nuisance Sensitivity:**
   - **Illumination Shifts:** 0 false alarm pixels (TinyCD is highly robust to uniform solar intensity shifts).
   - **Coregistration Jitter (2px):** 0 false alarm pixels (immune to sub-pixel edge jitter).
   - **Seasonal Canopy Phenology:** 5,643 false positive pixels (detects green-to-yellow canopy changes as structural change).
   - **Sun-Azimuth Shadow Shifts:** 149 false positive pixels.
4. **Qualitative Satellite Sample:** `tests/fixtures/real_rs_sample.tif` was evaluated in pure qualitative mode (no ground truth mask), proving the harness executes real natural terrain inference without fabricating fake metrics or IoU.

### 9.4 Empirical Failure Taxonomy Distribution
Across the diverse evaluation track, empirical failure modes were categorized deterministically without semantic hallucination:
- `unclassified_failure` / non-urban domain shift: 50.0% (6 samples)
- `small_change_missed`: 8.3% (1 sample)
- `seasonal_variation_false_positive`: 8.3% (1 sample)
- Reliable / Zero Error: 33.3% (4 samples)

### 9.5 Connection to Future Remote-Sensing Adaptation (M10)
These empirical findings prove that while pre-trained TinyCD is an exceptional specialist for urban structural change, it cannot be claimed as universally adapted across general remote sensing. This directly justifies and scopes **Milestone M10 (Remote-Sensing Adaptation)** to fine-tune multi-spectral representations for natural land-cover and seasonal invariance.

---

## 10. Milestone M5: Change Semantic Interpretation & Change VQA (`CHANGE_VQA`)

### 10.1 Architecture & Role
- **Specialist Class:** `ChangeVQASpecialist` (`backend/models/change_vqa.py`), subclassing `BaseSpecialist`.
- **Registry Identifier:** `CHANGE_VQA`, registered in `ModelRegistry` under `TaskType.CHANGE_VQA`.
- **Underlying Engine:** `Qwen/Qwen2-VL-2B-Instruct` operating in an evidence-conditioned visual mode.
- **Hardware Profile:** 4,601–4,692 MB peak allocated VRAM during composite vision generation. Runs within the local 12 GB RTX 4070 VRAM budget.

### 10.2 Conditioning Workflow
The VLM is never called blindly on raw images with ungrounded text prompts. It is strictly conditioned on:
1. **Visual Evidence:** A horizontal 3-panel composite ($W \times 3 \times H$): `Time 1 (Before) | Time 2 (After) | Change Overlay` with semi-transparent red highlighting detected change masks.
2. **Geometric & Statistical Invariants:**
   - Total changed pixel count and coverage percentage.
   - Connected component cluster bounding boxes `[xmin, ymin, xmax, ymax]`.
   - Cluster identifiers (e.g. `changed_region_01`, `changed_region_02`).
3. **Structured Prompt Constraints:**
   - Requires JSON schema compliance: `temporal_direction`, `predominant_transition`, `transitions` (`from_class`, `to_class`, `region_id`, `is_uncertain`), and `summary_answer`.
   - Disallows hallucinated land-cover labels: unidentifiable classes default to `"unknown"` or `"uncertain"`.

### 10.3 Scientific Honesty Guarantees
1. **Zero-Change Short-Circuit:** If `changed_pixels == 0`, generation is immediately bypassed with $0\text{ ms}$ latency. Emits `no_change` and 0 transitions.
2. **Defensible Confidence Separation:** Any model uncertainty returned by the VLM is isolated as `semantic_uncertainty` in `SemanticChangeInterpretation`. It does not overwrite the mathematically derived evidence confidence score ($w_{\text{input}} \cdot C_{\text{input}} + w_{\text{reg}} \cdot C_{\text{reg}} + w_{\text{model}} \cdot C_{\text{model}}$).

### 10.4 Empirical Telemetry (RTX 4070 SUPER, CUDA 12.4)
- **Mean Semantic Inference Latency:** 5,880 ms (when positive change detected); 0 ms (when zero change detected).
- **Peak VRAM Allocated:** 4,692 MB (leaving > 7.3 GB free VRAM).
- **Zero Memory Leaks:** Tensor allocations released after generation.


