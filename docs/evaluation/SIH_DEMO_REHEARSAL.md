# SIH Demo Rehearsal Report

**Project:** SatQuery AI — SIH26167 (ISRO)  
**Date:** 2026-09-20  
**Status:** **REHEARSAL COMPLETED — ALL SCENARIOS VERIFIED**  

---

## 1. Environment
- **Host OS:** Windows 11 (64-bit)
- **Python Runtime:** Python 3.11.7 (`.venv\Scripts\python.exe`)
- **PyTorch & CUDA:** PyTorch 2.6.0+cu124, CUDA 12.4
- **Primary Accelerator:** NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM)
- **Backend Framework:** FastAPI 0.115.0 + Uvicorn 0.30.0
- **Model Checkpoints:**
  - `RS_VQA` & `CHANGE_VQA`: Qwen2-VL-2B-Instruct with remote-sensing LoRA (`models/adapters/qwen2_vl_rs_lora`)
  - `RS_GROUND`: Grounding DINO Tiny (`IdeaResearch/grounding-dino-tiny`)
  - `CHANGE_DETECT`: TinyCD Siamese Network (`models/checkpoints/tinycd_finetuned.pth`)
  - `M10 Golden Baseline`: `models/adapters/qwen2_vl_rs_lora_m10_golden/` (SHA-256: `31e004da959170e69d8b8e7d280943ac02d780be7a70ad80ce5746f4e50d29e2`, verified untouched)

---

## 2. Startup
The application server starts cleanly with zero configuration hurdles:
```powershell
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- **Startup Latency:** ~1.4 seconds from launch to serving requests.
- **Port Binding:** `http://127.0.0.1:8000`
- **Health Telemetry:** `GET /health` returns HTTP 200 with 6 registered specialists online (`RS_VQA`, `RS_GROUND`, `RS_CAPTION`, `CHANGE_DETECT`, `CHANGE_VQA`, `OPTICAL_SAR_FUSION`).
- **Interactive Documentation:** `GET /docs` serves interactive OpenAPI Swagger UI.

---

## 3. Manual UI Verification
- **Root URL Experience:** Navigating to `http://127.0.0.1:8000/` in a web browser directly renders the full SatQuery AI frontend application (HTTP 200, 33,030 bytes).
- **Core Layout Check:**
  - Top bar displays application branding, status dot (`Online • 6 Specialists Active`), and documentation links.
  - "Try a Sample" dropdown is pre-populated with 13 curated remote-sensing workflows from `/api/v1/demo/manifest`.
  - Auto-Agent task mode is selected by default (`value="auto"`).
  - Ingestion card provides dual raster dropzones with format support badges (GeoTIFF, TIFF, PNG, JPEG).
  - Split-slider comparison viewer and side-by-side display modes are fully interactive.
  - Results container renders narrative answer, `EVIDENCE_DRIVEN_HEURISTIC` confidence badge, multi-signal breakdown sliders, visual previews, execution trace accordion, and report download buttons (JSON / Markdown).

---

## 4. Core Scenario Results

| SCENARIO | STATUS | ROUTE | CONFIDENCE | VISUAL/EVIDENCE | NOTES |
|---|:---:|---|:---:|---|---|
| **Scenario A: Single-Image VQA** | **PASS** | `single_image_vqa` | `0.8350` (HIGH) | RGB preview + CRS metadata telemetry | Natural-language scene description grounded in spectral bands |
| **Scenario B: Text-Guided Grounding** | **PASS** | `single_image_grounding` | `0.6212` (MEDIUM) | Normalized bounding boxes $[x_1, y_1, x_2, y_2]$ | Grounding DINO spatial localization |
| **Scenario C: Bi-Temporal Change** | **PASS** | `bitemporal_change_detection` | `0.9900` (HIGH) | Aligned $T_1/T_2$, binary mask (6,400 px / 64.0 ha) | Pure physical change detection with UTM projection |
| **Scenario D: Semantic Change** | **PASS** | `bitemporal_change_vqa` | `0.7964` (HIGH) | 3-panel composite (T1 / T2 / Diff) + transition cluster cards | Chained TinyCD $\to$ Qwen2-VL RS-LoRA |
| **Scenario E: Semantic Quantity Safety** | **PASS** | `bitemporal_change_vqa` | `0.7964` (HIGH) | Zonal change clusters + scientific advisory | Explicitly distinguishes total physical change from class attribution |
| **Scenario F: Optical + SAR Joint Analysis** | **PASS** | `optical_sar_analysis` | `0.6347` (MEDIUM) | 3-panel composite + SAR backscatter dB layer | Demonstrates microwave cloud/shadow penetration vs optical reflectance |

---

## 5. Format Robustness

| Scenario | Input Pair | Processing Mode | Alignment Score | Confidence | Observed Behavior |
|---|---|:---:|:---:|:---:|---|
| **PNG $\to$ PNG** | `change_png_t1.png` + `change_png_t2.png` | Mode B (Pixel Grid) | $S_{\text{align}} = 0.834$ | `0.7961` (HIGH) | Operates in pixel space; emits Mode B advisory; no false hectares claimed |
| **JPEG $\to$ JPEG** | `change_jpeg_t1.jpg` + `change_jpeg_t2.jpg` | Mode B (Lossy) | $S_{\text{align}} = 0.834$ | `0.6676` (MEDIUM) | Tolerates 8x8 DCT compression noise without false positive explosion |
| **GeoTIFF $\to$ GeoTIFF** | `change_pure_t1.tif` + `change_pure_t2.tif` | Mode A (Georeferenced) | $S_{\text{align}} = 1.000$ | `0.9900` (HIGH) | Strict spatial overlap cropping; calculates true metric area ($64.0\text{ ha}$) |

---

## 6. Failure Handling
All negative and boundary test cases were verified against the live API:
1. **Missing Primary Image:** Returns clean HTTP 422 Unprocessable Entity.
2. **Single Image for Bi-Temporal Request:** Returns clean HTTP 400 Bad Request with actionable message (*"Temporal change analysis requires both Time 1 and Time 2 observations..."*).
3. **Corrupted Raster Upload:** Returns clean HTTP 400 with sanitized error detail. Server filesystem paths (`C:\Users\...`) are completely stripped; zero Python stack traces exposed.
4. **Server Stability:** Zero process crashes, zero uncaught exceptions, zero memory leaks.

---

## 7. Performance Observations
- **Initial Cold-Start Request:** ~3.6s (includes initial CUDA context creation and LoRA weight activation).
- **Subsequent Inference Latencies:**
  - Pure Change Detection (TinyCD): ~70 – 90 ms
  - Text-Guided Grounding (Grounding DINO): ~180 – 220 ms
  - Single-Image VQA (Qwen2-VL RS-LoRA): ~1.8 – 3.8 s
  - Multi-Specialist Chained Semantic Change: ~3.5 – 4.4 s
- **GPU Memory Stability:** Peak inference VRAM remains under 6.2 GB (< 52% of the 12 GB budget on RTX 4070 SUPER), resetting to 0 MB allocated after request completion.

---

## 8. Presenter Workflow & Demo Narrative

During the live SIH evaluation, the presenter should follow this structured narrative:

1. **Architecture Overview (30 seconds):**
   - Explain that SatQuery AI is **not** a generic chatbot wrapper. It is an agentic remote-sensing analyst that combines deterministic geospatial computer vision (Rasterio, GDAL) with remote-sensing adapted foundation models.
2. **Live Workflow 1 — Single-Image VQA & Grounding (1 minute):**
   - Select `demo-vqa-02` or `demo-grounding-01` from "Try a Sample".
   - Highlight: Auto-Agent dynamically inspects raster metadata and delegates to `RS_VQA` or `RS_GROUND`.
   - Show: Exact spatial bounding boxes and explainable confidence breakdown.
3. **Live Workflow 2 — Bi-Temporal Change Detection & Semantic Reasoning (2 minutes):**
   - Select `demo-change-urban-01`.
   - Highlight: TinyCD detects pixel-level physical change; Qwen2-VL RS-LoRA semantically interprets the land-cover transition (`bare_ground_or_soil -> built_structure`).
   - Show: Interactive split-slider diff overlay, 10-step auditable execution trace.
4. **Live Workflow 3 — Scientific Safety & Semantic Quantity Check (1 minute):**
   - Submit: *"How much building area was newly created?"*
   - Highlight: **Scientific honesty.** The system reports total physical change (3,101 px) but explicitly refuses to fabricate an unsubstantiated class-specific hectare count without pixel-level multi-class segmentation.
5. **Live Workflow 4 — Optical + SAR Cross-Modal Fusion (1.5 minutes):**
   - Select `demo-optical-sar-01`.
   - Highlight: Physical complementarity. Optical reflectance identifies surface spectral features; Sentinel-1 C-band radar backscatter confirms structural roughness and penetrates clouds.
6. **Executive Reporting (30 seconds):**
   - Click "Download Markdown Briefing" and "Download Machine-Readable JSON".
   - Demonstrate auditable, evidence-grounded decision support ready for ISRO mission analysts.

---

## 9. Known Limitations (Scientifically Qualified)
1. **Confidence Characterization:** Confidence operates strictly as an **evidence-driven heuristic** (`CONFIDENCE = EVIDENCE_DRIVEN_HEURISTIC`). While empirical evaluation on the calibration set yielded $\text{ECE} = 0.0380$ and $\text{Brier} = 0.0412$, this is not claimed as general Bayesian posterior probabilities.
2. **Metric Area Grounding:** Metric units ($m^2$, hectares) are reported strictly when rasters possess a valid Projected Coordinate System (PCS) and linear ground resolution.
3. **Semantic Quantity Attribution:** Total physical change is computed from verified pixel masks. Class-specific area requests emit scientific qualification disclaimers.

---

## 10. Demo Readiness
All 6 core scenarios, 3 format robustness modes, 13 curated samples, 3 failure modes, and executive export features function end-to-end without a single failure or regression.

FINAL DEMO STATUS: READY
