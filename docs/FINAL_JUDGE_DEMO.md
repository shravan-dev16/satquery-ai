# SatQuery AI — Final Judge Demonstration Script (SIH26167)

**Milestone:** M13 Release Freeze  
**Target URL:** `http://127.0.0.1:8000/`  
**API Documentation:** `http://127.0.0.1:8000/docs`  
**Health Endpoint:** `http://127.0.0.1:8000/health`  

---

## Overview

SatQuery AI is an **agentic remote-sensing analysis system** engineered for SIH26167 (ISRO). It is not a generic image chatbot. It dynamically routes queries across specialized vision-language models, physical change detectors, cross-modal optical/SAR fusers, and geospatial validators to produce evidence-grounded, defensible intelligence.

This document provides the exact script and sequence for presenting SatQuery AI to the evaluation committee.

---

## Pre-Demo Checklist (1 Minute Before Judging)

1. Ensure the backend server is running in the virtual environment:
   ```bash
   .\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
2. Open your web browser and navigate directly to:
   ```
   http://127.0.0.1:8000/
   ```
3. Verify that the SatQuery AI interface loads cleanly:
   - Header shows: `SatQuery AI — Agentic Multimodal Remote Sensing Analyst`
   - Mode selector shows `Auto-Agent (Autonomous Routing)` selected by default.
   - Dual upload boxes (`Primary Image / T1 / Optical`, `Secondary Image / T2 / SAR`) are ready.
   - Sample tray button `Try a Sample` is available.

---

## DEMO 1: Single-Image Scene Understanding & GeoTIFF Metadata

**Objective:** Demonstrate remote-sensing visual question answering, multispectral GeoTIFF ingestion, CRS validation, and spatial telemetry extraction without generic chatbot hallucinations.

### Presenter Action:
1. Click **`Try a Sample`**.
2. Select **`Single-Image GeoTIFF VQA (Sentinel-2)`** (`demo-vqa-02`).
   - The sample auto-populates `datasets/ui_demo/vqa_sample_2.tif`.
   - Query auto-populates: `"Describe the land-cover and major features visible in this remote sensing image."`
3. Click **`Analyze`**.

### What to Point Out to Judges:
- **Agentic Routing:** Show the `Execution Trace` drawer: `Auto-Agent` identified a single input and selected the `RS_VQA` specialist.
- **Remote-Sensing Fine-Tuned Model:** The model active is Qwen2-VL-2B adapted with RS LoRA (trained on VRSBench remote sensing scenes).
- **Geospatial Integrity:** Show the `Geospatial & Image Metadata` card:
  - Driver: `GTiff`
  - Dimensions: `512 x 512`
  - Native Coordinate Reference System (CRS) and geotransform are reported.
- **Defensible Confidence:** High confidence score backed by valid band calibration and sharp imagery.

---

## DEMO 2: Text-Guided Spatial Region Grounding

**Objective:** Demonstrate precise spatial bounding box grounding driven by natural language, without manual polygon drawing.

### Presenter Action:
1. Click **`Try a Sample`**.
2. Select **`Text-Guided Region Grounding (Harbor)`** (`demo-grounding-01`).
   - The sample auto-populates `datasets/ui_demo/grounding_harbor.png`.
   - Query: `"Where is the harbor?"`
3. Click **`Analyze`**.

### What to Point Out to Judges:
- **Specialist Routing:** Routed autonomously to `RS_GROUND` (Grounding DINO specialist).
- **Visual Spatial Evidence:** The result does not just say "there is a harbor". It renders interactive bounding box overlays with normalized pixel coordinates and confidence scores directly on the high-resolution aerial scene.
- **Auditable Trace:** Points to bounding box detection count, thresholding, and execution latency.

---

## DEMO 3: Bi-Temporal Physical Change Detection

**Objective:** Demonstrate bi-temporal image pair ingestion, sub-pixel phase correlation alignment, and physical surface change segmentation using the fine-tuned TinyCD network.

### Presenter Action:
1. Click **`Try a Sample`**.
2. Select **`Pure Bi-Temporal Change Detection`** (`demo-change-pure-01`).
   - Populates T1: `datasets/ui_demo/change_pure_t1.tif`
   - Populates T2: `datasets/ui_demo/change_pure_t2.tif`
   - Query: `"What changed between these two dates?"`
3. Click **`Analyze`**.

### What to Point Out to Judges:
- **Automatic Multi-Image Detection:** Agent detects two temporal rasters, validates pair compatibility (dimensions, channels, alignment), and engages `CHANGE_DETECT` (TinyCD checkpoint).
- **Side-by-Side Comparison:** The UI displays:
  - `Image T1 (Before)`
  - `Image T2 (After)`
  - `Change Detection Overlay / Binary Mask`
- **Quantitative Metrics:** Shows physical change percentage (e.g., ~12.4% surface alteration) and pixel-count statistics without fabricating semantic land-cover figures.

---

## DEMO 4: Semantic Change Analysis & Safeguards

**Objective:** Demonstrate multi-specialist chaining (`CHANGE_DETECT -> CHANGE_VQA`), semantic land-cover interpretation, and mandatory SIH safeguards preventing fabricated area figures.

### Presenter Action:
1. Click **`Try a Sample`**.
2. Select **`Semantic Built-Up Change`** (`demo-change-urban-01`).
   - Populates T1: `datasets/ui_demo/change_urban_t1.tif`
   - Populates T2: `datasets/ui_demo/change_urban_t2.tif`
   - Query: `"Did the built-up area increase?"`
3. Click **`Analyze`**.

### What to Point Out to Judges:
- **Multi-Specialist Fusion:** The agentic trace shows a 2-stage pipeline:
  1. `CHANGE_DETECT` isolates physical pixel differences.
  2. `CHANGE_VQA` interprets the semantic context of the altered regions.
- **Physical vs Semantic Distinction:** The system explicitly states the physical change area and provides the semantic interpretation of urban construction.
- **Semantic Quantity Safeguard (Crucial):**
  - Switch the query to: `"How much did the buildings increase in hectares?"`
  - Click **`Analyze`**.
  - Show judges the **Scientific Advisory**: SatQuery AI warns that un-georeferenced images lack metric ground sample distance (GSD), and pixel-level semantic classification cannot fabricate ungrounded hectare figures.

---

## DEMO 5: Optical + SAR Cross-Modal Complementary Analysis

**Objective:** Demonstrate cross-modal analysis combining cloud/lighting-sensitive Optical imagery with all-weather structural SAR imagery.

### Presenter Action:
1. Click **`Try a Sample`**.
2. Select **`Optical + SAR Cross-Modal Fusion`** (`demo-optical-sar-01`).
   - Populates Optical: `datasets/ui_demo/optical_multimodal.tif`
   - Populates SAR: `datasets/ui_demo/sar_multimodal.tif`
   - Query: `"Use the optical and SAR images together to identify built-up and water-covered regions."`
3. Click **`Analyze`**.

### What to Point Out to Judges:
- **Specialist Routing:** Routed to `OPTICAL_SAR_FUSION`.
- **Complementarity Breakdown:**
  - Optical reveals surface spectral reflectance, vegetation indices, and color boundaries.
  - SAR reveals double-bounce dielectric backscatter (built-up structures) and specular dark reflections (calm water).
- **3-Panel Composite:** The evidence panel generates a side-by-side view with cross-modal fusion overlays.

---

## DEMO 6: Reliability Under Adverse Conditions (Misalignment Gating)

**Objective:** Prove system robustness. When fed misaligned or invalid imagery, SatQuery AI does not output hallucinated certainty; it reduces confidence and issues an actionable advisory.

### Presenter Action:
1. Click **`Try a Sample`**.
2. Select **`Misaligned / Risk Test (Reduced Confidence & Advisory)`** (`demo-risk-misaligned-01`).
   - Populates T1: `datasets/ui_demo/misaligned_t1.png`
   - Populates T2: `datasets/ui_demo/misaligned_t2.png` (synthetically shifted)
   - Query: `"What changed between these two dates?"`
3. Click **`Analyze`**.

### What to Point Out to Judges:
- **Automated Co-Registration Check:** The spatial alignment validator computes normalized cross-correlation and phase correlation PSR.
- **Warning Advisory:** System displays `[ADVISORY: Spatial misalignment detected between inputs; confidence degraded]`.
- **Evidence-Driven Confidence:** System confidence heuristic automatically penalizes down from `HIGH` to `LOW/MODERATE` (~0.40–0.50).
- **Defense:** SatQuery AI protects decision-makers from acting on false change artifacts created by registration drift.

---

## DEMO 7: Downloadable Intelligence Reports

**Objective:** Demonstrate operational compliance with SIH reporting requirements.

### Presenter Action:
1. With any previous analysis completed, scroll to the top of the **Analysis Results** card.
2. Click **`Download JSON`**. Show that a structured, machine-readable audit report downloads immediately.
3. Click **`Download Markdown`**. Show that a clean, human-readable executive briefing downloads immediately.

---

## Summary Script for Judges

> *"Distinguished Judges, SatQuery AI solves the core challenge of remote sensing intelligence: turning massive multimodal Earth observation data into verifiable, evidence-grounded answers. Unlike generic chatbots, SatQuery AI executes true agentic orchestration—routing queries to specialized remote-sensing vision-language adapters, physical change detectors, and cross-modal fusers. Every result is grounded by interactive spatial evidence, coordinate telemetry, an auditable execution trace, and evidence-driven confidence heuristics that refuse to hallucinate."*
