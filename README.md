# SatQuery AI

**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

- **Hackathon:** Smart India Hackathon — SIH26167
- **Organization:** Indian Space Research Organisation (ISRO)
- **Domain:** Space Technology

---

## 1. System Overview

SatQuery AI is an agentic remote-sensing analysis system designed to interpret single, paired, multimodal (Optical + SAR), and bi-temporal Earth-observation imagery through natural language queries. 

Unlike generic image chatbots, SatQuery AI:
- Employs a **hybrid architecture** combining deterministic geospatial computer vision (rasterio, pixel-accurate change masking, zonal statistics) with remote-sensing adapted vision-language models.
- Demonstrates genuine **Optical + SAR physical complementarity** (C-band microwave radar penetrating clouds/shadows, while optical resolves surface spectral reflectance).
- Provides an **auditable execution trace** and an **evidence-weighted confidence heuristic** grounded in verifiable input and model signals.
- Decouples task routing and specialized analysis behind an extensible **Model Registry**.

---

## 2. Directory Structure

```text
satquery-ai/
├── AGENTS.md                  # Project constitution & technical rules
├── README.md                  # Project overview & setup guide
├── pyproject.toml             # Project build configuration
├── requirements.txt           # Python dependency specification
│
├── backend/
│   ├── config.py              # System configuration, hardware limits & paths
│   ├── main.py                # FastAPI entrypoint (/api/v1)
│   ├── agent/                 # Agentic controller, registry & Pydantic schemas
│   │   ├── registry.py        # Central ModelRegistry
│   │   └── schema.py          # StandardResultContract & data models
│   ├── preprocessing/         # GeoTIFF ingestion, CRS, & alignment
│   ├── models/                # Specialist model implementations (BaseSpecialist)
│   │   └── base.py            # Abstract BaseSpecialist interface
│   ├── evidence/              # Evidence fusion, consistency & confidence
│   ├── evaluation/            # Reproducible benchmark harnesses
│   └── reports/               # Report generation (PDF & JSON)
│
├── docs/                      # Architectural & Engineering Specifications
│   ├── ARCHITECTURE.md        # System architecture & hybrid pipeline
│   ├── REQUIREMENTS_TRACEABILITY.md # SIH26167 requirement mapping
│   ├── DATASET_PLAN.md        # Curated datasets & zero-leakage strategy
│   ├── MODEL_PLAN.md          # Specialist registry & adaptation plan
│   ├── API_CONTRACT.md        # RESTful API specifications
│   └── IMPLEMENTATION_PLAN.md # Incremental milestones (M0 to M14)
│
├── scripts/
│   └── generate_fixtures.py   # Deterministic test fixture generator
│
└── tests/                     # Automated test suite (<3s execution)
    ├── conftest.py            # Shared pytest fixtures
    ├── fixtures/              # In-repo lightweight GeoTIFFs (< 1.5 MB)
    ├── test_env.py            # Environment, CUDA & Rasterio verification
    ├── test_schemas.py        # Contract serialization & validation tests
    ├── test_registry.py       # ModelRegistry & BaseSpecialist unit tests
    ├── test_fixtures.py       # GeoTIFF CRS, transform, & band tests
    └── test_api.py            # FastAPI endpoint integration tests
```

---

## 3. Quickstart & Local Setup

### Prerequisites
- Windows 11 / Linux
- Python 3.11
- NVIDIA GPU (RTX 4070 12 GB VRAM or equivalent with CUDA 12.x)

### Environment Setup

```powershell
# 1. Clone repository
git clone https://github.com/shravan-dev16/satquery-ai.git
cd satquery-ai

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Generate lightweight test fixtures (< 1.5 MB)
python scripts/generate_fixtures.py

# 6. Run the automated test suite
pytest
```

---

## 4. Current Milestone Status

- **M0 (Scaffolding & Environment Layer):** **COMPLETED**
  - Production directory skeleton created.
  - Typed Pydantic schemas implemented (`StandardResultContract`, `EvidenceBundle`, `SpecialistInput/Output`, `ConfidenceBreakdown`).
  - BaseSpecialist abstract interface and central ModelRegistry implemented.
  - Synthetic lightweight GeoTIFF fixtures generated (1.32 MB total, strictly $<5\text{ MB}$).
  - Full automated test suite passing.
- **M1 (Single-Image Ingestion & Real VQA Smoke Test):** **COMPLETED**
  - **M1-A (Ingestion & Validation):** `RasterValidator`, `GeoTIFFReader`, and `ModalityDetector` implemented, distinguishing verified sensor metadata from heuristic inferences. Pre-flight endpoint `POST /api/v1/validate` operational.
  - **M1-B (VQA Specialist Smoke Test):** `RemoteSensingVQASpecialist` (initial candidate using zero-shot `Qwen2-VL-2B-Instruct`, Apache 2.0) integrated behind `BaseSpecialist` and registered in `ModelRegistry`.
  - **Endpoint:** `POST /api/v1/analyze` operational, executing full ingestion $\to$ validation $\to$ registry selection $\to$ VLM inference $\to$ trace generation $\to$ `StandardResultContract`.
  - **Real Satellite Smoke Test:** Verified on real satellite test imagery (`tests/fixtures/real_rs_sample.tif`, Rasterio repository GeoTIFF test image used for geospatial pipeline smoke testing, BSD 3-Clause). Peak VRAM ~4.45 GB on RTX 4070 SUPER, clean post-inference deallocation.
  - **Notice:** This is a smoke test, not benchmark validation or remote-sensing adaptation.
- **M2 (Text-Guided Grounding & Spatial Projection):** **COMPLETED**
  - **M2-A (Spatial Region Grounding):** `RemoteSensingGroundingSpecialist` (`RS_GROUND`) implemented and registered in `ModelRegistry`. Decoupled behind `BaseSpecialist`.
  - **Strict Coordinate Convention:** Origin top-left $(0, 0)$, $x$ right, $y$ down, ordering strictly `[xmin, ymin, xmax, ymax]`.
  - **Deterministic Geospatial Projection:** `SpatialTransformer` in `backend/evidence/spatial.py` transforms pixel bounding boxes into projected GeoJSON Polygons via the GeoTIFF affine transform matrix and preserves CRS (`EPSG:32618`).
  - **Grounding Output Contract:** Returns `evidence.boxes` and `evidence.regions` adhering to `{region_name, label, bbox_pixel, confidence, polygon_pixel: null}`. Derived geometries are transparently labeled as axis-aligned polygons, never misrepresented as semantic segmentation.
  - **Empty Result Warning:** Returns explicit structured warnings when target features are absent.
  - **M2-B (Captioning Preparation):** `RemoteSensingCaptionSpecialist` (`RS_CAPTION`) interface registered in `ModelRegistry`.
- **M2.1 (Grounding Recovery & Model Selection):** **COMPLETED**
  - **Empirical Evaluation:** Evaluated `Qwen/Qwen2-VL-2B-Instruct` (baseline) against `IDEA-Research/grounding-dino-base` (dedicated detector) across 35 verified VRSBench validation samples.
  - **Measured Results:** Grounding DINO achieves **0.2469 Mean IoU** (vs 0.0728 for Qwen), **22.86% success@0.5** (vs 0.0% for Qwen), **189.9 ms mean latency** (9.2x faster), and **1,732 MB peak VRAM** (2.5x lower).
  - **Degenerate Box Detection:** Implemented rule flagging boxes covering $\ge 98\%$ image area as degenerate (`is_degenerate=True`), excluding them from valid spatial evidence.
  - **Query Normalizer:** Deterministic natural-language to detector entity conversion (`QueryNormalizer.normalize`).
  - **Architectural Selection:** Grounding DINO Base selected as active `RS_GROUND` specialist; Qwen2-VL retained exclusively for `RS_VQA`. Both remain dynamically decoupled.
- **M3 (Bi-Temporal Change Analysis Foundation):** **COMPLETED**
  - **11-Point Bi-Temporal Validation:** `BiTemporalValidator` verifies CRS, geotransforms, resolutions, dimensions, real spatial overlap percentage, and acquisition dates ($t_1 < t_2$). Emits explicit warnings when timestamps are missing.
  - **Deterministic Geospatial Alignment:** `BiTemporalAligner` handles cross-CRS reprojection, configurable bilinear/nearest resampling, and common spatial intersection cropping while preserving affine geotransforms.
  - **Empirical Model Evaluation (LEVIR-CD Held-Out 20-Sample Subset):**
    - **TinyCD Specialist (~316k params):** **0.8347 Mean IoU**, **0.9091 F1 Score**, **0.9216 Precision**, **0.8985 Recall**, **0.0052 False-Positive Ratio**, **92.3 ms Latency**, **219.4 MB Peak VRAM**.
    - **Deterministic CVA Baseline:** **0.0327 Mean IoU**, **0.0619 F1 Score**, **44.6 ms Latency**, **0.0 MB VRAM**.
    - **Selection:** TinyCD selected as primary learned `CHANGE_DETECT` specialist; CVA retained as fallback and zero-parameter baseline.
  - **Physical Area Computation:** Physical area in $m^2$ and hectares calculated strictly when CRS uses linear meters ($640,000\text{ m}^2$ / $64.0\text{ ha}$ on UTM fixture); pixel statistics with explicit warnings otherwise.
  - **Visual Evidence Artifacts:** Generates aligned $T_1$, $T_2$, binary mask, and semi-transparent red change overlay servable via `/api/v1/static/previews/`.
  - **Auditable 9-Step Trace:** Complete execution trace returned in `StandardResultContract` via FastAPI `POST /api/v1/analyze`.
  - **Automated Tests:** 77 fast unit/integration tests passing in 3.60s.
- **M4 (Dedicated Change Detection & Benchmark Evaluation):** **COMPLETED**
  - **Automated Benchmark Harness:** Production evaluation runner in `backend/evaluation/change_benchmark.py` supporting `TinyCD`, `DeterministicCVASpecialist`, and extensible specialists.
  - **LEVIR-CD Reproduction:** Reproduced M3 baseline on 20-sample validation split (`docs/evaluation/levir_cd_subset.json`): TinyCD achieved **0.8347 Macro Mean IoU**, **0.9091 Macro F1**, **0.9195 Pixel-Global F1**, **95.5 ms Latency**, and **219.4 MB Peak VRAM** on RTX 4070. CVA scored **0.0327 Mean IoU** and **0.0619 F1**.
  - **Diverse Remote-Sensing Generalization:** Benchmarked 12 samples across 8 operational categories (`docs/evaluation/diverse_rs_manifest.json`): confirmed high urban/infrastructure accuracy, documented lower natural canopy/agriculture recall due to pre-training domain shift, and verified 0 false alarm pixels on illumination shifts and registration jitter.
  - **Structured Failure Taxonomy:** Categorized empirical error modes in `backend/evaluation/failure_taxonomy.py` (`seasonal_variation_false_positive`, `small_change_missed`, non-urban domain shift) to guide subsequent M10 remote-sensing adaptation.
  - **Machine-Readable Reports:** Full results saved in `docs/evaluation/m4_levir_cd_benchmark_report.json`, `docs/evaluation/m4_diverse_rs_evaluation_report.json`, and `docs/evaluation/M4_BENCHMARK_REPORT.md`.
  - **Automated Tests:** All 95 unit/integration tests passing in 17.5s.
- **M5 (Change Semantic Interpretation & Change VQA):** **COMPLETED**
  - **Specialist Implementation:** `ChangeVQASpecialist` (`CHANGE_VQA`) integrated behind `BaseSpecialist` and registered in `ModelRegistry` (now 5 active specialists registered: `RS_VQA`, `RS_GROUND`, `RS_CAPTION`, `CHANGE_DETECT`, `CHANGE_VQA`).
  - **Conditioned Visual Reasoning:** Evaluates an evidence-conditioned 3-panel composite preview (`Time 1 | Time 2 | Change Overlay`) with injected spatial clusters, bounding boxes, and pixel statistics.
  - **Scientific Honesty & Anti-Hallucination:** Zero-change detector output (`changed_pixels == 0`) immediately short-circuits VLM generation ($0\text{ ms}$ latency), mathematically precluding hallucination on identical observations. Ambiguous land-cover classes default defensively to `"unknown"`.
  - **Auditable 10-Step Execution Trace:** Semantic change queries insert Step 9 (`SemanticInterpretation`) to produce an auditable 10-step trace, while pure geometric change detection preserves the exact 9-step M3/M4 trace.
  - **Defensible Confidence Separation:** Semantic model uncertainty (`semantic_uncertainty`) is strictly separated from system evidence confidence scores.
  - **Interactive UI Integration:** Frontend at `/ui/` updated with 10-step pipeline visual stepper and rich `.semantic-card` displaying temporal direction, predominant transitions, and structured region links.
  - **Automated Tests & Evaluation:** 105 tests passing in ~59s across the repository; empirical smoke test verified on 4 real scenes in `docs/evaluation/M5_SEMANTIC_EVALUATION.md`.
- **M6 (Optical + SAR Cross-Modal Analysis):** **NEXT** (Verifiable C-band microwave cloud penetration and roughness contrast).


