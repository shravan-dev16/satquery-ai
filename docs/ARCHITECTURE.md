# SatQuery AI — System Architecture

**Document Version:** 2.0.0 (Post-Audit Revision)  
**Project:** SatQuery AI — Interactive Vision-Language Assistant for Multimodal Remote Sensing  
**Problem Statement:** Smart India Hackathon — SIH26167 (ISRO / Space Technology)  
**Host Hardware Profile:** NVIDIA GeForce RTX 4070 Laptop/Desktop GPU (12,282 MiB VRAM), 32 GB RAM, Windows 11  

---

## 1. Executive Summary & Design Philosophy

SatQuery AI is an **agentic remote-sensing analysis system**, not a generic conversational chatbot with image attachments. Earth observation data possesses physical dimensions, geographic projections, multi-spectral band configurations, distinct physical imaging principles (optical solar reflectance vs. SAR microwave backscatter), and strict spatial alignment requirements.

Generic Vision-Language Models (VLMs) fail when deployed directly on raw satellite data because they:
1. Cannot interpret geographic coordinate reference systems (CRS) or geotransforms.
2. Lack awareness of calibrated radiometric physical units (reflectance vs. decibel backscatter $\sigma^0$).
3. Cannot perform deterministic pixel-level change detection or accurate area calculations ($m^2$, hectares).
4. Suffer from terrestrial hallucinations when answering queries over aerial/satellite overhead views.

To satisfy **SIH26167**, SatQuery AI implements a **hybrid agentic architecture**:
- **Geospatial & Radiometric Ingestion Layer:** Deterministically validates, reprojects, and normalizes GeoTIFFs.
- **Agentic Controller & Intent Parser:** Translates natural language queries and metadata into an auditable execution Directed Acyclic Graph (DAG).
- **Specialist Model Registry:** Decouples task execution into domain-specific modules (`RS_VQA`, `RS_GROUND`, `RS_CAPTION`, `CHANGE_DETECT`, `CHANGE_VQA`, `OPTICAL_SAR_ANALYSIS`).
- **Hybrid Evidence Fusion Layer:** Combines deterministic computer vision (pixel change masks, connected components, zonal statistics, GeoJSON vectorization) with semantic reasoning from remote-sensing adapted vision-language models.
- **Consistency & Defensible Confidence Engine:** Audits agreement across independent models, flags contradictions, and calculates an unmanufactured, signal-grounded confidence score.
- **Presentation & Export Layer:** Provides interactive geospatial visualization, observable execution tracing, and downloadable PDF/JSON mission reports.

```
+-------------------------------------------------------------------------------+
|                                  USER / CLIENT                                |
+-------------------------------------------------------------------------------+
                                        |
                   HTTP POST /api/v1/analyze (Images + Query)
                                        v
+-------------------------------------------------------------------------------+
|                           INPUT VALIDATION LAYER                              |
| - Format Check (GeoTIFF, TIFF, PNG, JPG)                                      |
| - Geospatial Metadata Extractor (CRS, Transform, Resolution, Bounds, Nodata)   |
| - Modality & Band Verifier (Optical RGB/NIR, SAR VV/VH, Grayscale)            |
| - Bi-temporal Co-registration & Spatial Overlap Evaluator (>20% overlap)      |
+-------------------------------------------------------------------------------+
                                        | Validated Imagery & Metadata
                                        v
+-------------------------------------------------------------------------------+
|                    AGENTIC CONTROLLER & TASK ROUTER                           |
| - Natural Language Query & Intent Parser (Rule-based + Intent Classifier)     |
| - Multi-image Relationship Analyzer (Single, Bi-temporal, Optical+SAR)        |
| - Dynamic DAG / Execution Plan Generator                                      |
| - Observable Step Tracer (Logs every decision, tool call, parameter)          |
+-------------------------------------------------------------------------------+
                                        |
                         Dispatches Structured Tool Calls
                                        v
+-------------------------------------------------------------------------------+
|                         SPECIALIST MODEL & TOOL REGISTRY                      |
|                                                                               |
|  [GEO_PREPROCESSOR]      [RS_VQA]                  [RS_GROUND / CAPTION]      |
|  - Percentile Scaling    - RS-Adapted VLM          - Florence-2 RS / SAM      |
|  - Reprojection / Crop   - RSVQAxBEN / VRSBench    - Bounding Boxes & Masks   |
|                                                                               |
|  [CHANGE_DETECT]         [CHANGE_VQA]              [OPTICAL_SAR_ANALYSIS]     |
|  - TinyCD / BIT / CVA    - Mask-Conditioned VLM    - Dual-stream Backscatter  |
|  - Pixel Change Masks    - Semantic Narrative      - Cloud-Penetrating Fusion |
+-------------------------------------------------------------------------------+
                                        |
                         Raw Predictions, Masks, Vectors
                                        v
+-------------------------------------------------------------------------------+
|                       EVIDENCE FUSION & AUDIT LAYER                           |
| - Mask Vectorization & GeoJSON Generation (rasterio.features)                 |
| - Zonal Statistics (Exact changed area in m^2 and %, cluster counts)          |
| - Cross-Model Consistency Checker (e.g. Change Detector vs. VLM Description)  |
| - Defensible Confidence Estimation (Signal-weighted, not fabricated)          |
| - Audit Trail Assembly (Full execution trace, warnings, latency)              |
+-------------------------------------------------------------------------------+
                                        |
                         Structured Result Contract JSON
                                        v
+-------------------------------------------------------------------------------+
|                         PRESENTATION & EXPORT LAYER                           |
| - Interactive Dashboard (Map views, split slider, mask overlays, bbox drawer) |
| - Trace Inspector (Observable step-by-step pipeline audit)                    |
| - Downloadable Report Generator (Structured PDF & JSON export)                 |
+-------------------------------------------------------------------------------+
```

---

## 2. Architecture Status & Decision Classifications

### 2.1 CONFIRMED Decisions
- **Decoupled Specialist Registry:** Specialist models are isolated behind standard abstract Python interfaces (`BaseSpecialist`) and declared in `backend/agent/registry.py`. No model calls are hardcoded into web controllers.
- **Hybrid Change Detection:** Change detection uses a deterministic/neural spatial model (`TinyCD` or `BIT` or classical difference) to generate pixel-exact masks and compute real area metrics ($m^2$), followed by a vision-language specialist (`CHANGE_VQA`) conditioned on the mask for semantic interpretation. Generic VLMs will NOT be relied upon for raw pixel change masking.
- **Physical Optical-SAR Complementarity:** Optical and SAR are treated as physically distinct modalities. Optical provides spectral reflectance; SAR provides active microwave backscatter (roughness, double bounce, cloud/shadow penetration). Cross-modal analysis must explicitly output cloud-penetration and structural contrast layers rather than opaque tensor concatenation.
- **Strict Input Validation Gatekeeper:** All inputs must pass `rasterio` validation (CRS, transform, resolution, bounds, nodata, temporal ordering). Missing spatial overlap ($<20\%$) triggers immediate structured rejection (`NO_SPATIAL_OVERLAP`).
- **Defensible Confidence Scoring:** Confidence is computed from observable quality signals (input quality, alignment error, model probability, consistency penalty) and never randomly generated.
- **Unified Coordinate Convention & Spatial Transformation (M2):** The system enforces ONE single internal coordinate convention across all modules: origin at top-left $(0, 0)$, $x$ right, $y$ down, ordering strictly `[xmin, ymin, xmax, ymax]`. The `SpatialTransformer` in `backend/evidence/spatial.py` converts pixel bounding boxes to projected GeoJSON Polygons via the GeoTIFF affine matrix and native CRS. Derived geometries are transparently labeled as axis-aligned bounding box polygons, never misrepresented as semantic segmentation.

### 2.2 ASSUMPTIONS
- Input imagery will be predominantly standard GeoTIFF (Sentinel-1, Sentinel-2, Landsat, or commercial aerial imagery like LEVIR), with PNG/JPEG accepted for benchmark validation.
- Smoke testing uses verified open test rasters (`tests/fixtures/real_rs_sample.tif`, designated as *"Rasterio repository GeoTIFF test image used for geospatial pipeline smoke testing"* from `https://raw.githubusercontent.com/rasterio/rasterio/master/tests/data/RGB.byte.tif`, BSD 3-Clause).
- Bi-temporal pairs possess valid geospatial georeferencing allowing `rasterio` to calculate coordinate intersections, or are pre-aligned spatial crops.
- The host GPU (RTX 4070 12 GB) has approximately 8.8 GB available VRAM when system display processes are active.

### 2.3 TO VERIFY Items
- **Windows Binary Wheels:** Verified: `rasterio` and `gdal` wheels run cleanly under Windows 11 with Python 3.11.
- **PyTorch CUDA 12.x Binding:** Verified: `torch.cuda.is_available()` binds to the RTX 4070 SUPER and utilizes tensor cores.
- **Actual Peak Memory under Sequential Execution:** Empirically verified: VQA and Grounding require ~4.37 GB peak VRAM in bfloat16, leaving >7.6 GB headroom.

### 2.4 RISKS
- **Sub-pixel Misregistration:** Even small 1-2 pixel registration shifts between bi-temporal scenes cause false positive edge changes.  
  *Mitigation:* Apply morphological noise filtering (erosion/dilation) and minimum area thresholds on change clusters.
- **VLM Overconfidence / Hallucination:** Generic vision models asserting false features.  
  *Mitigation:* Verifiable remote-sensing domain adaptation using LoRA on open RS data, coupled with `ConsistencyChecker` cross-verification.

### 2.5 SPECIALIST CANDIDATE STATUS (Post-M2.1)
- **RS_GROUND:** `IDEA-Research/grounding-dino-base` is the selected primary candidate (zero-shot, unadapted). Selected after rigorous M2.1 evaluation against `Qwen2-VL-2B-Instruct` on a deterministic 35-sample VRSBench validation referring-expression subset. Grounding DINO delivers 0.2469 Mean IoU (vs 0.0728 for Qwen), 22.86% success@0.5 (vs 0.0% for Qwen), 189.9 ms mean latency, and 1.73 GB peak VRAM. Overlapping candidates are suppressed with NMS (threshold 0.70) and degenerate full-image boxes ($\ge 98\%$ area) are excluded from API output.
- **RS_VQA:** `Qwen2-VL-2B-Instruct` is retained as the active specialist for vision-language semantic question answering and spatial reasoning (zero-shot, unadapted). Decoupled behind `BaseSpecialist`.
- **RS_CAPTION:** Interface prepared in `backend/models/caption.py` (M2-B) without premature implementation.
- **CHANGE_DETECT:** Bi-temporal alignment foundation and change detection vertical slice (TinyCD vs deterministic CVA baseline) implemented in M3/M4.
- **CHANGE_VQA:** Semantic change interpretation and change-based VQA planned for M5.
- **OPTICAL_SAR_ANALYSIS:** Dual-stream physics-informed extractor planned for M6.

---

## 3. Practical Hardware & Inference Constraints (RTX 4070 12 GB)

Parameter count is **not** a rigid architectural constraint. A 7B parameter model quantized to 4-bit (AWQ/NF4) requires ~3.5 GB of weight VRAM, which easily runs alongside specialized heads. Instead, we enforce **practical runtime inference constraints**:

1. **Peak Working VRAM Ceiling:** $\le 7.5\text{ GB}$ maximum allocated VRAM across any pipeline step, leaving $\ge 1.3\text{ GB}$ headroom against our 8.8 GB available ceiling.
2. **Precision Policy:**
   - Vision & Change Detection specialists: `torch.float16` or `torch.bfloat16`.
   - Vision-Language Models: `torch.float16` for models $\le 3\text{B}$; 4-bit (`bitsandbytes` NF4 or AWQ) for models $> 3\text{B}$.
3. **Sequential Pipeline Execution:** In multi-stage workflows (e.g. Bi-temporal Change Masking $\to$ Change VQA), models execute strictly sequentially. Intermediate heavy tensors are deleted and `torch.cuda.empty_cache()` is called between stages.
4. **Tiling & Dynamic Resolution:** Images larger than $1024 \times 1024$ are processed via sliding-window tiles ($512 \times 512$ with $64\text{ px}$ overlap) rather than single high-resolution VRAM allocations, avoiding $O(N^2)$ attention memory spikes.
5. **Interactive Batch Size:** Strictly $B=1$ for all real-time inference endpoints.

---

## 4. Evidence-First Fusion & Consistency Pipeline

### 4.1 Hybrid Evidence Flow
```
[Raw GeoTIFF Pair (t1, t2)]
            │
            ▼
┌─────────────────────────┐
│ Alignment & Normalization│ (Reprojection, Resampling, Percentile Scaling)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ CHANGE_DETECT (TinyCD)  │ (Deterministic / Neural Pixel Masking)
└───────────┬─────────────┘
            │
            ├──► Binary Change Mask [H x W]
            ├──► Soft Probability Heatmap [H x W]
            │
            ▼
┌─────────────────────────┐
│ Evidence Vectorizer     │ (Polygonize clusters, compute area in m^2 and %)
└───────────┬─────────────┘
            │
            ├──► GeoJSON Polygons & Bounding Boxes
            ├──► Zonal Change Metrics (Area, Centroids, Count)
            │
            ▼
┌─────────────────────────┐
│ CHANGE_VQA (RS-VLM)     │ (Conditioned on T1, T2, and Verified Change Mask)
└───────────┬─────────────┘
            │
            ├──► Semantic Transition Description
            │
            ▼
┌─────────────────────────┐
│ Consistency Checker     │ (Validates detector stats vs VLM narrative)
└───────────┬─────────────┘
            │
            ▼
[Standard Result Contract JSON]
```

### 4.2 Cross-Model Consistency Checking
The `ConsistencyChecker` validates outputs across independent modules before generating the final answer:
- **Change Detection vs VLM Consistency:** If the change detector finds $0.05\%$ change, but the VLM claims *"Extensive urban expansion occurred across the entire area"*, a `HIGH_CONTRADICTION_WARNING` is attached, the confidence score is penalized by $-0.30$, and the discrepancy is made transparent to the user.
- **Optical vs SAR Complementary Validation:** When optical detects water (low reflectance) and SAR detects high backscatter (rough or double-bounce surface), the system records an explicit physical divergence note (e.g. *"Potential emergent vegetation or submerged structures detected in SAR"*).

---

## 5. Defensible Confidence Estimation Formula

Under `AGENTS.md` Rule 11, confidence is never fabricated. It is computed as:

$$\text{Confidence} = 0.20 \cdot C_{\text{input}} + 0.20 \cdot C_{\text{registration}} + 0.60 \cdot C_{\text{model}} - P_{\text{conflict}}$$

Where:
- $C_{\text{input}} \in [0.0, 1.0]$: Fraction of valid, non-nodata, well-exposed pixels.
- $C_{\text{registration}} \in [0.0, 1.0]$: Spatial overlap ratio and sub-pixel alignment score ($1.0$ for single images).
- $C_{\text{model}} \in [0.0, 1.0]$: Specialist model output probability / softmax margin.
- $P_{\text{conflict}} \in [0.0, 0.5]$: Consistency penalty imposed when independent specialists disagree.
- Final confidence is clipped strictly to $[0.05, 0.99]$.
