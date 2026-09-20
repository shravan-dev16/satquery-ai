# SatQuery AI — M12 UI Implementation & Architecture Guide

**Milestone:** M12 Full UI + Product Integration  
**Authority:** Smart India Hackathon (SIH26167) / Indian Space Research Organisation (ISRO)  
**Status:** Complete & Frozen Baseline Verified  
**Audience:** Judges, Technical Evaluators, Frontend/UX Engineers  

---

## 1. Overview & Architectural Philosophy

SatQuery AI is not a generic multimodal chat interface. It is an **autonomous remote-sensing vision-language assistant** purpose-built for Earth Observation intelligence across single-image, bi-temporal, and cross-modal sensor imagery.

Milestone M12 represents the final user-facing productization phase. It wraps the verified, frozen backend (M0–M11) into a clean, modern, technical, and judge-ready web application defaulting to **Auto-Agent orchestration**.

```
[ User Imagery + Query ]
           │
           ▼
[ Auto-Agent Router ] ────────► Model Selection (VQA / Grounding / TinyCD / Qwen2-VL / Optical-SAR)
           │
           ▼
[ Specialist Execution DAG ] ──► Preprocessing & Alignment (FFT / PSR)
           │
           ▼
[ Evidence Fusion & Gating ] ──► M8 Consistency Gating + M9 Defensible Confidence
           │
           ▼
[ Judge-Facing Product UI ] ───► Answer + Adaptive Visual Hub + M11 Executive Report Export
```

---

## 2. Core UI Architecture & Design Standards

The user interface is implemented as a high-performance, single-page application (SPA) adhering to the following design principles:

1. **Auto-Agent Default:** The primary user interface operates autonomously. Non-technical judges and analysts never need to select internal Python classes, specialist model names, or pipeline routes.
2. **Visual Hierarchy (5-Second Comprehension):**
   - **Step 1: Input Ingestion & "Try a Sample" Toolbar:** Dropzones with Drag-and-Drop support for primary and secondary observations, coupled with a 1-click sample loader.
   - **Step 2: Natural-Language Query Bar:** Query input with contextual prompt suggestions and an active Auto-Agent telemetry indicator.
   - **Step 3: Analytical Findings & Executive Answer Card:** Prominent display of the verbatim AI remote-sensing analyst answer, with zero client-side hallucination or metric fabrication.
   - **Step 4: Adaptive Visual Hub:** Contextually switches between:
     - *Bi-Temporal Mode:* Interactive Before/After Split Comparison Slider + 4-panel Side-by-Side Gallery (T1, T2, Binary Mask, Change Overlay).
     - *Grounding Mode:* Target Scene + Visual Grounding Overlay with projected bounding boxes.
     - *Single-Image Mode:* Scene overview and multispectral metadata.
     - *Optical-SAR Mode:* 3-Panel Multimodal Composite (Optical RGB + SAR Backscatter + Joint Fused Land-Cover).
   - **Step 5: Defensible System Confidence (M9):** Tier badge (HIGH, MEDIUM, LOW, UNSUPPORTED), numeric confidence score, subfactor telemetry grid (Specialist, Evidence, Alignment, Input, Consistency Penalty), and explainable factor checkmarks.
   - **Step 6: Quantitative Telemetry Grid:** Surface pixel count, Change Ratio (% AOI), Connected Clusters, and Verified Metric Area ($m^2$ and ha; strictly restricted to georeferenced rasters).
   - **Step 7: M11 Structured Report Export:** Direct 1-click export for machine-readable JSON and human-readable Markdown executive briefing.
   - **Step 8: Expandable Technical Audits:** Deep-dive accordion panels for Geospatial Bounding Boxes, Evidence Consistency (M8), Auditable Execution Trace (Rule 12), and Developer/Diagnostics Console.

---

## 3. API Contract & Integration Specification

The frontend connects directly to the RESTful endpoints exposed by FastAPI (`backend/main.py`):

| Method | Endpoint | Description | Payload Format |
|---|---|---|---|
| `GET` | `/health` / `/api/v1/health` | Service liveness probe and registered specialist telemetry | JSON |
| `GET` | `/api/v1/demo/manifest` | Manifest listing all 12 curated demonstration workflows | JSON |
| `GET` | `/api/v1/demo/files/{filename}` | Binary access for demo sample imagery | Binary Raster |
| `POST` | `/api/v1/analyze` | **Unified Agentic Entrypoint** executing full pipeline | `multipart/form-data` |
| `GET` | `/api/v1/static/previews/{file}` | Web-renderable PNG previews, masks, and overlays | PNG Image |

### 3.1 Analysis Request Format (`POST /api/v1/analyze`)
- `query` (`string`, required): Natural-language analytical question.
- `image_primary` (`file`, required): Primary raster (T1 observation, single scene, or optical image).
- `image_secondary` (`file`, optional): Secondary raster (T2 observation or SAR microwave image).
- `task_hint` (`string`, optional): Diagnostic override (`auto`, `change_vqa`, `optical_sar`, `tinycd_raw`, `cva_raw`). Defaults to `auto`.

### 3.2 Response Contract (`StandardResultContract`)
Adheres strictly to the frozen schema:
- `task`: Canonical task string (e.g. `bitemporal_change_detection`, `single_image_grounding`).
- `status`: Execution status (`success`, `partial`, `failed`).
- `answer`: Grounded analyst response text.
- `confidence`: Heuristic multi-factor score ($0.0 \dots 1.0$).
- `confidence_level`: Categorical tier (`HIGH`, `MEDIUM`, `LOW`, `UNSUPPORTED`).
- `confidence_breakdown`: Sub-scores, penalties, supporting factors, and calculation details.
- `evidence`: Bundle containing images (`url`, `role`), masks, bounding boxes, statistics, semantic interpretation, and consistency reports.
- `warnings`: Preprocessing, alignment, and format limitation advisories.
- `execution_trace`: Step-by-step observable pipeline telemetry with timings.
- `report`: Pre-packaged M11 `AnalystReport` object.

---

## 4. Supported Operational Workflows & Demo Samples

All 12 operational workflows are pre-configured in `datasets/ui_demo/manifest.json` and accessible via the **"Try a Sample"** toolbar:

| Workflow ID | Workflow Name | Inputs | Sample Files | Expected Routing | Key Evidence |
|---|---|---|---|---|---|
| `demo-vqa-01` | Single-Image VQA (Aerial Scene) | 1 PNG | `vqa_sample_1.png` | `single_image_vqa` | Scene description, parking/road context |
| `demo-vqa-02` | Single-Image GeoTIFF VQA | 1 GeoTIFF | `vqa_sample_2.tif` | `single_image_vqa` | Sentinel-2 multispectral feature analysis |
| `demo-grounding-01` | Text-Guided Grounding | 1 PNG | `grounding_harbor.png` | `single_image_grounding` | Grounding overlay, harbor bounding box |
| `demo-change-pure-01` | Pure Bi-Temporal Change | 2 TIFFs | `change_pure_t1.tif` + `t2` | `bitemporal_change_detection` | 64.00 ha, 640,000 m², change mask |
| `demo-change-urban-01` | Semantic Built-Up Change | 2 TIFFs | `change_urban_t1.tif` + `t2` | `bitemporal_change_vqa` | Urban transition reasoning, 31.01 ha |
| `demo-change-buildings-01` | Semantic Quantity Attribution | 2 TIFFs | `change_urban_t1.tif` + `t2` | `bitemporal_change_vqa` | Qualified attribution without area fabrication |
| `demo-change-vegetation-01` | Vegetation Canopy Loss | 2 TIFFs | `change_forest_t1.tif` + `t2` | `bitemporal_change_vqa` | Forest canopy loss transition, 163.84 ha |
| `demo-change-water-01` | Water Reservoir Dynamics | 2 TIFFs | `change_water_t1.tif` + `t2` | `bitemporal_change_vqa` | Water body boundary expansion, 163.84 ha |
| `demo-optical-sar-01` | Optical + SAR Sensor Fusion | 1 Opt + 1 SAR | `optical_multimodal.tif` + `sar` | `optical_sar_analysis` | 3-Panel composite, backscatter contrast |
| `demo-bitemporal-png-01` | PNG-to-PNG Bi-Temporal | 2 PNGs | `change_png_t1.png` + `t2` | `bitemporal_change_detection` | Pixel-space statistics (396 px), no metric ha |
| `demo-bitemporal-jpeg-01` | JPEG Compressed Bi-Temporal | 2 JPEGs | `change_jpeg_t1.jpg` + `t2` | `bitemporal_change_detection` | 85 px change, format limitation warnings |
| `demo-risk-misaligned-01` | Misaligned / Risk Test Pair | 2 PNGs | `misaligned_t1.png` + `t2` | `bitemporal_change_detection` | Alignment advisory, reduced confidence |
| `demo-killer-workflow-01` | Full Multi-Specialist Flow | 2 TIFFs | `change_urban_t1.tif` + `t2` | `bitemporal_change_vqa` | End-to-end multi-model execution and audit |

---

## 5. Scientific & Legal Safeguards (Zero Fabricated Output)

1. **No Fabricated Geospatial Area:** PNG and JPEG images lacking spatial references (`CRS` and `geotransform`) display `Pixel space only (Metric area unmeasured)`. Frontend never converts pixels to hectares based solely on dimensions.
2. **Binary Change vs. Semantic Class Separation:** Total detected physical surface change is strictly distinguished from class-specific built-up or vegetation change unless pixel-level multi-class segmentation evidence exists.
3. **Defensible Confidence:** Stated explicitly as a multi-factor heuristic (weighted across specialist signal, alignment quality, evidence sufficiency, and consistency penalties), not an uncalibrated statistical probability.
4. **Path Sanitization:** Server filesystem paths (e.g. `C:\Users\...` or `/home/...`) are systematically sanitized to basename filenames before display in UI text, answers, or warnings.

---

## 6. M11 Structured Report Export

The interface provides instant download buttons for the complete `AnalystReport` packaged by M11:
- **Download JSON Report:** Downloads `satquery_report_<report_id>.json` containing full provenance ledgers, quantitative metrics, bounding boxes, and execution trace.
- **Download Markdown Report:** Downloads `satquery_report_<report_id>.md` structured as an executive intelligence briefing.
- **Toggle Executive Briefing:** Opens an in-app viewer displaying the markdown briefing directly.

---

## 7. Known Limitations & Environment Notes

- **Playwright Browser Automation:** In this local Windows development environment, automated headless Playwright driver initialization via `open_browser_url` fails due to external distribution CDN 404 responses for the specific Windows driver zip (`playwright-1.57.0-win32_x64.zip`). 
- **Verification Strategy:** Fully validated via comprehensive automated API test suite (`scripts/verify_m12_workflows.py`), full pytest regression suite (`222 passed`), static frontend asset tests (`tests/test_frontend_integration.py`), and confirmed live Uvicorn HTTP 200 server request logs for `/ui/`, `/ui/index.css`, `/ui/app.js`, `/api/v1/health`, and `/api/v1/demo/manifest`.

---

## 8. Manual Verification Instructions (Judging Demonstration)

To demonstrate SatQuery AI to judges or evaluators:

1. **Launch Backend Dev Server:**
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
2. **Open Browser:**
   Navigate to `http://127.0.0.1:8000/ui/`.
3. **Step 1: Test Bi-Temporal Change:**
   - From "Try a Sample", choose **Pure Bi-Temporal Change Detection**.
   - Observe both TIFF files loaded into Slot 1 and Slot 2, with query auto-filled.
   - Click **Analyze Imagery**.
   - Observe real-time progress stepper $\to$ Results displayed.
   - Inspect the interactive Before/After split slider, changed pixel count (6,400 px), verified metric area (64.00 ha / 640,000 $m^2$), and HIGH confidence badge (0.99).
4. **Step 2: Test Semantic Change Reasoning:**
   - From "Try a Sample", choose **Semantic Built-Up Change**.
   - Click **Analyze Imagery**.
   - Inspect the semantic transition card (`bare_land -> built_up`), the limitation notice, and the M8 consistency narrative.
5. **Step 3: Test Text-Guided Grounding:**
   - From "Try a Sample", choose **Text-Guided Region Grounding (Harbor)**.
   - Click **Analyze Imagery**.
   - Observe adaptive UI switching to Single Grounding View with bounding box overlay and pixel coordinates table.
6. **Step 4: Test Optical + SAR Cross-Modal Fusion:**
   - From "Try a Sample", choose **Optical + SAR Cross-Modal Fusion**.
   - Click **Analyze Imagery**.
   - Observe 3-panel composite (Optical RGB + SAR Microwave Backscatter + Joint Fused Land-Cover).
7. **Step 5: Test Format Robustness & Risk Safeguards:**
   - Choose **JPEG Compressed Bi-Temporal Change** or **Misaligned / Risk Test Pair**.
   - Observe system advisories and non-georeferenced raster status (`Pixel space only`).
8. **Step 6: Export Reports:**
   - Click **Download JSON Report** and **Download Markdown Report**.
   - Click **Toggle Executive Briefing** to review the rendered intelligence report.
