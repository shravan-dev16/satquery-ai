# SatQuery AI — M13 Final Release Audit & System Freeze Report

**Date:** 2026-09-20  
**Milestone:** M13 (Final Hardening, Deployment, Browser Validation, and Release Freeze)  
**Problem Statement:** SIH26167 — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries  
**Organization:** Indian Space Research Organisation (ISRO)  
**Domain:** Space Technology  
**Evaluation Recommendation:** `READY_FOR_SIH_DEMO`  

---

## 1. Executive Summary

SatQuery AI has achieved full implementation, rigorous validation, and final operational freeze under Milestone M13. The system represents an agentic remote-sensing analytical assistant that accepts natural-language queries over single, paired, bi-temporal, and multimodal (Optical + SAR) Earth observation imagery.

Unlike generic chatbot wrappers, SatQuery AI implements an autonomous orchestrator that interprets natural-language intent, inspects raster spatial metadata and band structures, validates co-registration and compatibility, and dynamically delegates sub-tasks to specialized remote-sensing models. Every analysis produces visual and spatial evidence, an auditable execution trace, evidence-driven heuristic confidence scoring, and downloadable intelligence reports.

All backend components, model weights, and fine-tuned checkpoints are frozen. The canonical user interface is served directly from the root URL `http://127.0.0.1:8000/`.

---

## 2. System Architecture & Information Flow

SatQuery AI follows an evidence-first, modular architecture designed around the SIH26167 specification:

```
                      +----------------------------------+
                      |         Natural Language         |
                      |        Query & Image(s)          |
                      +-----------------+----------------+
                                        |
                                        v
                      +----------------------------------+
                      |     Input Validator & Metadata   |
                      |  (Rasterio / GDAL / CRS / Bands) |
                      +-----------------+----------------+
                                        |
                                        v
                      +----------------------------------+
                      |     Query / Task Interpreter     |
                      |   (Intent & Modality Resolver)   |
                      +-----------------+----------------+
                                        |
                                        v
                      +----------------------------------+
                      |        Agentic Controller        |
                      |   (Dynamic Task Graph Routing)   |
                      +-----------------+----------------+
                                        |
      +-------------------+-------------+-------------+--------------------+
      |                   |                           |                    |
      v                   v                           v                    v
+-----------+    +-------------------+       +-----------------+   +---------------+
|  RS_VQA   |    |     RS_GROUND     |       |  CHANGE_DETECT  |   |  OPTICAL_SAR  |
| (Qwen2-VL |    |  (Grounding DINO  |       |    (TinyCD      |   | (Cross-Modal  |
|  RS-LoRA) |    |     Specialist)   |       |   Fine-Tuned)   |   |    Fusion)    |
+-----+-----+    +---------+---------+       +--------+--------+   +-------+-------+
      |                    |                          |                    |
      +--------------------+-------------+------------+--------------------+
                                         |
                                         v
                      +----------------------------------+
                      |      Evidence Fusion Engine      |
                      |  (Spatial Overlays, Masks, BBox) |
                      +-----------------+----------------+
                                        |
                                        v
                      +----------------------------------+
                      |  Consistency & Alignment Gating  |
                      | (Phase Correlation / PSR / Agree)|
                      +-----------------+----------------+
                                        |
                                        v
                      +----------------------------------+
                      |  Defensible Confidence Estimator |
                      | (Signal-Derived Multi-Factor)    |
                      +-----------------+----------------+
                                        |
                                        v
                      +----------------------------------+
                      |     Result Synthesizer & Trace   |
                      |  (Answer, Evidence, Warnings)    |
                      +-----------------+----------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
+----------------------------------+         +----------------------------------+
|      Interactive Web UI          |         |    Exportable Intelligence       |
|    (http://127.0.0.1:8000/)      |         |         JSON & Markdown          |
+----------------------------------+         +----------------------------------+
```

---

## 3. Canonical Production Endpoints

| Endpoint | Method | Client / Purpose | Output Format |
|---|---|---|---|
| `/` | `GET` | Browser navigation entrypoint | `text/html` (Polished SatQuery AI UI) |
| `/` | `GET` | API clients / Health check probes | `application/json` (`{"app": "SatQuery AI", ...}`) |
| `/ui/` | `GET` | Explicit UI route alias | `text/html` |
| `/test-ui/` | `GET` | Developer & debug test UI route | `text/html` |
| `/index.css` | `GET` | Root static stylesheet | `text/css` |
| `/app.js` | `GET` | Root frontend application script | `application/javascript` |
| `/favicon.ico` | `GET` | Browser favicon handler | `204 No Content` |
| `/api/v1/analyze` | `POST` | Core multimodal analysis pipeline | `application/json` (Standard Result Contract) |
| `/api/v1/health` | `GET` | Subsystem telemetry & model readiness | `application/json` |
| `/health` | `GET` | Root health status alias | `application/json` |
| `/api/v1/report/json` | `POST` | Downloadable audit report | `application/json` |
| `/api/v1/report/markdown`| `POST` | Downloadable executive summary | `text/markdown` |
| `/docs` | `GET` | Interactive Swagger API documentation | `text/html` |
| `/openapi.json` | `GET` | OpenAPI 3.0 specification | `application/json` |

---

## 4. Frozen Model & Checkpoint Inventory

All production model weights have been locked and verified against cryptographic hashes:

1. **Remote-Sensing Vision-Language Model (`RS_VQA` & `CHANGE_VQA`)**:
   - **Base Model:** Qwen2-VL-2B-Instruct
   - **Adaptation Technique:** Low-Rank Adaptation (LoRA, $r=16$, $\alpha=32$, dropout $0.05$) targeting all attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
   - **Training Data:** VRSBench remote-sensing visual question answering and scene description subset
   - **Adapter Directory:** `models/checkpoints/qwen2_vl_lora_extended`
   - **Adapter Checksum (SHA256):** `31e004da959170e69d8b8e7d280943ac02d780be7a70ad80ce5746f4e50d29e2` (`adapter_model.safetensors`)
   - **Integrity Status:** Verified unchanged against M10 Golden Baseline.

2. **Text-Guided Region Grounding Specialist (`RS_GROUND`)**:
   - **Architecture:** Grounding DINO (`IdeaResearch/grounding-dino-tiny`)
   - **Capability:** Natural-language zero-shot spatial bounding box localization with normalized bounding coordinates $[x_1, y_1, x_2, y_2]$ and per-detection confidence scores.

3. **Bi-Temporal Change Detection Specialist (`CHANGE_DETECT`)**:
   - **Architecture:** TinyCD Siamese Change Detection Network
   - **Checkpoint Path:** `models/checkpoints/tinycd_finetuned.pth`
   - **Size:** 1,280,183 bytes
   - **Checkpoint Checksum (SHA256):** `04eb7c032d23a67e1069eb87e91eb7ebbe876b5c3ea1dbca4b6bcf3f9c6d4ba4`
   - **Training Source:** LEVIR-CD bi-temporal building change dataset.

4. **Optical + SAR Cross-Modal Fusion Specialist (`OPTICAL_SAR_FUSION`)**:
   - **Capability:** Multi-spectral reflectance fusion with Sentinel-1 dual-polarization ($\text{VV}/\text{VH}$) backscatter analysis, cloud-robust feature extraction, and composite visual generation.

---

## 5. SIH26167 Requirement Traceability Matrix

| # | SIH26167 Mandatory Requirement | Implementing Module(s) | Verification Evidence | Status |
|---|---|---|---|---|
| 1 | Single optical/multispectral image analysis | `backend/preprocessing/geotiff.py`, `backend/models/vqa.py` | GeoTIFF CRS extraction, multispectral band extraction, demo sample `demo-vqa-02` | **VERIFIED** |
| 2 | Single SAR image analysis | `backend/preprocessing/modality.py`, `backend/models/optical_sar.py` | SAR single-channel backscatter detection, histogram scaling | **VERIFIED** |
| 3 | Single-image remote-sensing VQA | `backend/models/vqa.py`, Qwen2-VL RS-LoRA adapter | Evaluated on VRSBench; demo sample `demo-vqa-01` | **VERIFIED** |
| 4 | Additional task: Grounding / Captioning | `backend/models/grounding.py` (Grounding DINO) | Real spatial bounding boxes; demo sample `demo-grounding-01` | **VERIFIED** |
| 5 | Bi-temporal image-pair change analysis | `backend/models/change.py` (TinyCD checkpoint) | Binary change masks, pixel-count change stats; demo sample `demo-change-pure-01` | **VERIFIED** |
| 6 | Change description / Change VQA | `backend/models/change.py` -> `backend/models/vqa.py` | Chained multi-specialist semantic interpretation; demo sample `demo-change-urban-01` | **VERIFIED** |
| 7 | Optical + SAR cross-modal analysis | `backend/models/optical_sar.py` | 3-panel composite, complementary reasoning; demo sample `demo-optical-sar-01` | **VERIFIED** |
| 8 | Remote-sensing adaptation / fine-tuning | LoRA adapter on Qwen2-VL; fine-tuned TinyCD | Documented training runs, loss convergence, and frozen checkpoint hashes | **VERIFIED** |
| 9 | Agentic task interpretation & tool routing | `backend/agent/router.py`, `backend/agent/planner.py` | Structured intent graph, zero keyword-only shortcuts; 12/12 workflows verified | **VERIFIED** |
| 10| Input compatibility validation | `backend/preprocessing/validation.py`, `backend/preprocessing/alignment.py` | Sub-pixel phase correlation, dimension/CRS checks, misalignment warning gating | **VERIFIED** |
| 11| Evidence-grounded output | `backend/evidence/fusion.py`, `frontend/app.js` | Side-by-side viewer, change mask overlay, bounding box viewer, spatial metrics | **VERIFIED** |
| 12| Evidence-driven confidence heuristic | `backend/evidence/confidence.py` | Multi-factor signal estimation, adverse condition penalty, no arbitrary values | **VERIFIED** |
| 13| Auditable execution trace | `backend/agent/trace.py`, UI trace drawer | Step-by-step observable timing, specialist selection, and advisories exposed | **VERIFIED** |
| 14| Downloadable reports | `backend/main.py` (`/report/json`, `/report/markdown`) | UI one-click JSON audit report and Markdown briefing download | **VERIFIED** |

---

## 6. Verification & Test Results

### 6.1 Backend Test Suite (Pytest)
- **Execution Command:** `pytest -q`
- **Total Tests:** 225
- **Passed:** 225
- **Failed:** 0
- **Execution Time:** ~160s
- **Coverage Areas:** GeoTIFF raster parsing, CRS geotransform validation, format matrix robustness (TIFF, GeoTIFF, PNG, JPEG), agent routing logic, specialist registries, evidence fusion, confidence scoring, report generation, and frontend HTTP endpoints.

### 6.2 Format Robustness Matrix
- **Execution Script:** `python scripts/audit_format_matrix.py`
- **Scenarios Evaluated:** 10 multi-format bi-temporal scenarios
- **Overall Scenario Accuracy:** 90.0% (9/10 passing)
- **False Positive Change Rate:** 0.0% (No false alarms on identical pairs)
- **False Alignment Failures:** 0.0% (Sub-pixel phase correlation PSR resolves spectral attenuation)

### 6.3 End-to-End Operational Workflows
- **Execution Script:** `python scripts/verify_m12_workflows.py`
- **Workflows Verified:** 12/12 passing with 100% operational success
- **Telemetry Check:**
  - `Single-image VQA`: High confidence (0.79), CRS telemetry present.
  - `Region Grounding`: High confidence (0.83), valid bounding boxes rendered.
  - `Pure Bi-Temporal`: High confidence (0.85), TinyCD change mask generated.
  - `Semantic Urban Change`: High confidence (0.80), multi-specialist chaining confirmed.
  - `Semantic Quantity Safeguard`: Limitation advisory rendered, no fabricated metric area.
  - `Optical + SAR`: High confidence (0.78), cross-modal composite generated.
  - `Misaligned Pair`: Confidence penalized (0.45), alignment warning displayed.

### 6.4 Failure Mode & Negative Input Testing
- **Execution Script:** `python scripts/test_failure_modes.py`
- **Test Cases:**
  1. Missing file inputs $\to$ Clean HTTP 422 validation response.
  2. Single image with bi-temporal prompt $\to$ Routed gracefully or rejected with clear user guidance.
  3. Corrupted / zero-byte rasters $\to$ HTTP 400 with human-readable error.
  4. Non-georeferenced images requesting hectare quantities $\to$ Handled with scientific advisory; no hallucinated measurements.
  5. Internal error handling $\to$ Absolute server paths (`C:\Users\...`) sanitized from client-facing responses; zero raw exception traces exposed.

---

## 7. Security, Privacy & Data Hygiene Audit

1. **Credentials & Secrets:**
   - Git repository inspected: Zero API keys, private tokens, passwords, or credentials committed.
   - `.env.example` provided for local configuration.
2. **Server Path Sanitization:**
   - `_sanitize_error_message()` active in `backend/main.py`. Local temporary file paths (e.g., `AppData/Local/Temp/...`) are stripped and replaced with base filenames in all HTTP responses.
3. **Data Integrity:**
   - Datasets are partitioned clearly: Synthetic robustness fixtures, public benchmark subsets (LEVIR-CD, VRSBench), and UI demonstration samples are explicitly cataloged in `datasets/ui_demo/manifest.json`.

---

## 8. Hardware Profile & Runtime Efficiency

- **Development & Verification Host:** Local workstation with NVIDIA GeForce RTX 4070 (12 GB VRAM), 32 GB RAM, Windows 11.
- **VRAM Utilization:**
  - Idle server: ~0.8 GB VRAM.
  - Single-image VQA (Qwen2-VL RS-LoRA fp16): ~4.2 GB VRAM.
  - Region Grounding (Grounding DINO tiny): ~1.2 GB VRAM.
  - Change Detection (TinyCD): ~0.6 GB VRAM.
  - Peak inference load: < 6.5 GB VRAM (well within 12 GB ceiling).
- **Latency:**
  - Change detection inference: ~180 ms.
  - Region grounding inference: ~450 ms.
  - VQA / Scene understanding inference: ~1,800 ms.
  - End-to-end HTTP response cycle: < 2.5 s for single image, < 3.5 s for multi-specialist bi-temporal change.

---

## 9. Known Technical Limitations & Boundaries

In accordance with scientific integrity guidelines:
1. **Confidence Characterization (`CONFIDENCE = EVIDENCE_DRIVEN_HEURISTIC`):** Confidence estimation in M9/M13 operates as an **evidence-driven heuristic** combining model certainty, input raster quality, co-registration alignment PSR, and cross-specialist consistency gating. M9 is strictly not a generally calibrated posterior probability mechanism. Empirical evaluation on the stated calibration evaluation set ($N=10$) produced $\text{ECE} = 0.0380$ and $\text{Brier} = 0.0412$, but these empirical results do not claim to establish general calibrated posterior probabilities across uncalibrated sensors or arbitrary out-of-distribution imagery.
2. **Metric Area Grounding:** SatQuery AI only calculates metric surface areas (hectares, square meters) when rasters possess a valid Projected Coordinate System (PCS) and pixel ground resolution. On un-georeferenced web rasters (PNG/JPEG), only normalized pixel counts and percentages are reported.
3. **Synthetic Boundary Sensitivity:** On synthetic, non-georeferenced pairs with artificial sharp line shifts (e.g., `01_building`), TinyCD's convolutional attention may exhibit subtle boundary sensitivity under heavy JPEG compression artifacts. This is documented and handled via confidence penalties.
4. **Single Grounding Query Scope:** Grounding DINO detects spatial bounding boxes based on natural language expressions. Multiple conflicting entities in a single prompt are separated into distinct candidate boxes, but complex relational grounding is constrained by the underlying detector vocabulary.

---

## 10. Final Release Recommendation

All 14 mandatory functional requirements of Smart India Hackathon PS26167 are implemented, integrated, documented, and verified.

**Verdict: `READY_FOR_SIH_DEMO`**
