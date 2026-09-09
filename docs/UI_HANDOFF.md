# SatQuery AI — Frontend / UI Integration & Handoff Contract (Milestone M12 Guide)

**Document Version:** 1.0.0 (Backend Frozen Baseline)  
**Target Audience:** Person B (Frontend Developer / UX Engineer)  
**Standard Compliance:** `AGENTS.md` Rules 7, 8, 9, 10, 11, 12, 18, 19, 29, 37  
**Base URL:** `http://localhost:8000` (or reverse-proxy origin)  

---

## 1. Overview & Architectural Boundaries

This document specifies the exact, frozen RESTful HTTP interface between the **SatQuery AI backend** and the **frontend web application** (`/ui`).

### 1.1 Team Ownership Division (`AGENTS.md` Rule 19)
- **Person A (Backend Owner):** Core AI models, specialist registry, GeoTIFF processing, agentic routing (`AgentRouter`), execution DAG, consistency gating (M8), defensible confidence (M9), and report packaging (M11).
- **Person B (Frontend Owner):** Upload cards, query input, split-slider / comparison views, bounding box overlays, M9 confidence cards, M8 evidence reliability status, M11 report preview / download widgets, and overall visual polish.

### 1.2 Core Product Principles
- **Autonomous Orchestration by Default:** The user inputs imagery + a natural-language query. The backend automatically determines the task, selects models, validates inputs, and executes the sequence. Do not force the user to manually select AI models.
- **Evidence-First Presentation:** Never show text without corresponding spatial or statistical evidence when spatial evidence exists.
- **Separation of Total Physical Change vs. Semantic Class Area:** If a user asks *"How much buildings changed?"*, the backend returns total detected physical change ($px$ and $m^2$). The class-specific area is `null` with status `unmeasured_from_spatial_evidence`. **Do NOT fabricate class-specific hectares on the frontend.**

---

## 2. API Endpoints Reference

| Method | Endpoint | Description | Content-Type |
|---|---|---|---|
| `GET` | `/health` / `/api/v1/health` | Health and readiness check. Returns status and online specialist count. | None |
| `POST` | `/api/v1/validate` | Pre-flight validation. Checks raster headers, CRS, bounds, and pair compatibility without running models. | `multipart/form-data` |
| `POST` | `/api/v1/analyze` | **Primary Unified Entrypoint.** Submits query and rasters; executes full agentic pipeline. | `multipart/form-data` |
| `GET` | `/api/v1/models` | Status of registered specialist models and hardware capabilities. | None |
| `GET` | `/api/v1/static/previews/{filename}` | Direct static file access for web-renderable PNG previews, masks, and composites. | Static image |

---

## 3. Primary Analysis Endpoint (`POST /api/v1/analyze`)

### 3.1 Request Specification (`multipart/form-data`)

| Form Field | Type | Required | Description |
|---|---|---|---|
| `query` | `string` | **Yes** | Natural language question or instruction (e.g. *"What changed between these two dates?"*, *"Where is the harbor?"*, *"Describe the land-cover."*). |
| `image_primary` | `File` (Binary) | **Yes** | Primary remote-sensing raster (T1 GeoTIFF, TIFF, or PNG/JPEG benchmark image). |
| `image_secondary` | `File` (Binary) | No | Secondary observation (T2 for bi-temporal change, or SAR image for optical+SAR fusion). |
| `task_hint` | `string` | No | **Optional developer override only.** Defaults to `"auto"`. Accepted values: `"change_vqa"`, `"optical_sar"`, `"tinycd_raw"`, `"cva_raw"`. Omit or set to `null`/`"auto"` for normal autonomous operation. |

### 3.2 Standard Result Contract (`StandardResultContract`)

Every successful analysis returns an HTTP 200 JSON payload adhering to `StandardResultContract`:

```json
{
  "task": "bitemporal_change_vqa",
  "status": "success",
  "answer": "Total detected physical surface change is 1,662 pixels across 3 spatial clusters. Semantic interpretation indicates The built-up area increased. (predominant transition: built_structure -> built_structure). Note: Specific built-up / building surface area is not directly measurable from the binary change mask without pixel-level multi-class segmentation; total detected change encompasses all verified physical surface transitions.",
  "confidence": 0.8075,
  "confidence_level": "HIGH",
  "confidence_breakdown": {
    "heuristic_name": "evidence-weighted confidence heuristic",
    "overall_confidence": 0.8075,
    "specialist_confidence": 0.85,
    "input_quality_score": 1.0,
    "spatial_alignment_score": 1.0,
    "model_confidence_score": 0.85,
    "consistency_penalty": 0.0,
    "is_calibrated_probability": false,
    "evidence_quality_score": 1.0,
    "confidence_level": "HIGH",
    "confidence_factors": [
      "optimal_input_resolution_and_crs",
      "verified_spatial_co_registration",
      "specialist_confidence_calibrated",
      "evidence_consistency_corroborated"
    ],
    "confidence_warnings": [],
    "calculation_details": {
      "weights": { "specialist": 0.5, "evidence": 0.3, "alignment": 0.1, "input": 0.1 },
      "effective_penalty": 0.0
    }
  },
  "evidence": {
    "images": [
      {
        "role": "primary",
        "url": "/api/v1/static/previews/t1_preview_cd_36761.png",
        "width": 256,
        "height": 256,
        "crs": "EPSG:32650",
        "bounds": [500000.0, 4200000.0, 502560.0, 4202560.0]
      },
      {
        "role": "secondary",
        "url": "/api/v1/static/previews/t2_preview_cd_36761.png",
        "width": 256,
        "height": 256,
        "crs": "EPSG:32650",
        "bounds": [500000.0, 4200000.0, 502560.0, 4202560.0]
      },
      {
        "role": "change_overlay",
        "url": "/api/v1/static/previews/change_overlay_cd_36761.png",
        "width": 256,
        "height": 256,
        "crs": "EPSG:32650",
        "bounds": null
      },
      {
        "role": "semantic_composite",
        "url": "/api/v1/static/previews/semantic_composite_cvqa_36795.png",
        "width": 1344,
        "height": 448,
        "crs": null,
        "bounds": null
      }
    ],
    "masks": [
      {
        "mask_id": "change_mask",
        "label": "Detected Physical Change",
        "url": "/api/v1/static/previews/change_mask_cd_36761.png",
        "format": "image/png",
        "palette": { "0": "#000000", "255": "#FFFFFF" }
      }
    ],
    "boxes": [
      {
        "box_id": "cluster_001",
        "label": "Physical Change Cluster 1",
        "confidence": 0.90,
        "model_score": 0.90,
        "is_degenerate": false,
        "degenerate_reason": null,
        "coordinates_normalized": [0.12, 0.25, 0.45, 0.60],
        "coordinates_pixel": [30, 64, 115, 153],
        "geojson": {
          "type": "Polygon",
          "coordinates": [[[500300.0, 4201920.0], [501150.0, 4201920.0], [501150.0, 4201030.0], [500300.0, 4201030.0], [500300.0, 4201920.0]]]
        }
      }
    ],
    "statistics": [
      { "metric_name": "changed_pixels", "display_name": "Changed Pixels", "value": 1662.0, "unit": "px" },
      { "metric_name": "change_ratio_pct", "display_name": "Change Ratio", "value": 2.536, "unit": "%" },
      { "metric_name": "total_clusters", "display_name": "Change Clusters", "value": 3.0, "unit": "regions" }
    ],
    "regions": [],
    "complementarity_report": null,
    "semantic_interpretation": {
      "summary": "The built-up area increased.",
      "temporal_direction": "increased",
      "predominant_transition": "built_structure -> built_structure",
      "transitions": [
        {
          "transition_id": "trans_001",
          "from_class": "built_structure",
          "to_class": "built_structure",
          "description": "The built-up area increased.",
          "region_id": "cluster_001",
          "bbox_pixel": [30, 64, 115, 153],
          "semantic_confidence": 0.85,
          "is_uncertain": false,
          "evidence_support": "visual_crop_comparison"
        }
      ],
      "supporting_regions": ["cluster_001"],
      "semantic_uncertainty": 0.15,
      "warnings": []
    },
    "fused_items": [],
    "consistency_report": {
      "status": "CONSISTENT",
      "is_gated": false,
      "gating_action": "allow",
      "conflicts": [],
      "supporting_evidence_ids": [],
      "warnings": [],
      "specialist_confidences": { "CHANGE_DETECT": 0.90, "CHANGE_VQA": 0.85 },
      "evidence_quality_score": 1.0,
      "summary_narrative": "Evidence is consistent across specialists."
    }
  },
  "evidence_status": "CONSISTENT",
  "models": [
    { "identifier": "CHANGE_DETECT", "model_name": "TinyCD", "version": "1.0.0", "execution_time_ms": 115 },
    { "identifier": "CHANGE_VQA", "model_name": "Qwen2-VL-RS", "version": "1.0.0", "execution_time_ms": 842 }
  ],
  "parameters": {
    "changed_pixels": 1662,
    "change_ratio_pct": 2.536,
    "total_clusters": 3,
    "total_changed_pixels": 1662,
    "total_changed_area_ha": null,
    "semantic_class_detected": "built-up / building",
    "semantic_changed_area_ha": null,
    "semantic_area_status": "unmeasured_from_spatial_evidence",
    "semantic_area_limitation": "Class-specific surface area is not directly measurable from binary change detection without pixel-level multi-class semantic segmentation."
  },
  "warnings": [],
  "execution_trace": [
    { "step_number": 1, "step_name": "InputValidation", "status": "completed", "details": "Validated 2 GeoTIFF files...", "duration_ms": 12 },
    { "step_number": 2, "step_name": "TemporalValidation", "status": "completed", "details": "Temporal sequence confirmed...", "duration_ms": 5 },
    { "step_number": 3, "step_name": "SpatialCompatibility", "status": "completed", "details": "100% spatial overlap...", "duration_ms": 8 },
    { "step_number": 4, "step_name": "Alignment", "status": "completed", "details": "Raster grids aligned...", "duration_ms": 14 }
  ],
  "execution_time_ms": 985,
  "report": {
    "report_id": "rpt_9a8b7c6d",
    "report_title": "SatQuery Earth Observation Intelligence Report",
    "timestamp_utc": "2026-09-09T10:05:40Z",
    "task_type": "bitemporal_change_vqa",
    "status": "success",
    "executive_summary": "...",
    "quantitative_statistics": {
      "changed_pixels": 1662,
      "metrics": {
        "total_changed_pixels": 1662,
        "semantic_area_status": "unmeasured_from_spatial_evidence"
      }
    },
    "visual_artifacts": [],
    "provenance_records": []
  }
}
```

---

## 4. Frontend Component Mapping

### 4.1 Visual Hub & Slider Frame
- **Before Image (Left):** `contract.evidence.images.find(img => img.role === 'primary').url`
- **After / Overlay Image (Right):**
  - For Change Analysis: `contract.evidence.images.find(img => img.role === 'change_overlay').url`
  - For Grounding: `contract.evidence.images.find(img => img.role === 'grounding_preview').url`
  - For Optical+SAR: `contract.evidence.images.find(img => img.role === 'semantic_composite').url`
- **Binary Change Mask:** `contract.evidence.masks[0].url`

### 4.2 Defensible System Confidence Card (M9)
- **Overall Score:** `contract.confidence` (e.g., `0.8075` $\to$ `80.8%`)
- **Tier Badge (`confidence_level`):** `HIGH` (Emerald), `MEDIUM` (Amber), `LOW` (Crimson), `UNSUPPORTED` (Gray)
- **Subfactor Progress Bars:**
  - Specialist Confidence: `contract.confidence_breakdown.specialist_confidence`
  - Evidence Quality: `contract.confidence_breakdown.evidence_quality_score`
  - Spatial Alignment: `contract.confidence_breakdown.spatial_alignment_score`
  - Input Quality: `contract.confidence_breakdown.input_quality_score`
  - Consistency Penalty: `contract.confidence_breakdown.consistency_penalty`
- **Explainable Factors List:** `contract.confidence_breakdown.confidence_factors` (Array of strings, e.g. `"evidence_consistency_corroborated"`)

### 4.3 Evidence Consistency & Reliability Card (M8)
- **Consistency Status Badge:** `contract.evidence_status` or `contract.evidence.consistency_report.status`
  - Values: `CONSISTENT`, `PARTIALLY_CONSISTENT`, `UNCERTAIN`, `CONTRADICTORY`, `INSUFFICIENT_EVIDENCE`.
- **Conflicts List:** `contract.evidence.consistency_report.conflicts`
- **Gated Answer Warning:** If `contract.evidence.consistency_report.is_gated` is true, the answer starts with `[CONTRADICTION DETECTED]` or `[INSUFFICIENT EVIDENCE]`. Highlight this visually in amber/crimson.

### 4.4 Quantitative Metrics Cards
- **Changed Pixels:** `contract.parameters.changed_pixels`
- **Physical Area:** Find metric in `contract.evidence.statistics` (`changed_area_m2` and `changed_area_hectares`).
  - If null, display `"Pixel space only"` or `"Class-specific area unmeasured"`.
- **Change Ratio:** `contract.parameters.change_ratio_pct` + `"%"`
- **Total Clusters:** `contract.parameters.total_clusters`

### 4.5 M11 Downloadable Report Integration (Teammate Task)
- `contract.report` contains the full pre-built structured `AnalystReport`.
- A dedicated button e.g. `Download Report (JSON)` can serialize `JSON.stringify(contract.report, null, 2)` directly to a local file download.
- An optional modal or collapsible panel can render the report's `executive_summary`, `provenance_records`, and `audit_trail`.

---

## 5. Error Responses Reference

The backend strictly conforms to RFC 7807 problem details:

```json
{
  "detail": "Input validation failed: Unreadable file: Rasterio I/O error..."
}
```

- **HTTP 400 Bad Request:** User validation failure (missing T2 for bi-temporal query, incompatible dual optical images for SAR query, corrupted GeoTIFF).
- **HTTP 422 Unprocessable Entity:** Missing required form fields (`image_primary` or `query`).
- **HTTP 500 Internal Server Error:** Unexpected model execution crash (system logs audit event; client receives clean JSON).

**Frontend Rule:** Always display `data.detail` inside the `#error-banner` without crashing the application.

---

## 6. Sample UI Demo Dataset Reference

A reproducible demo dataset has been prepared in `datasets/ui_demo/` with manifest `datasets/ui_demo/manifest.json`:

| File(s) | Suggested Query | Workflow |
|---|---|---|
| `vqa_sample_1.png` | *"What is present in this image?"* | Single-Image VQA |
| `grounding_harbor.png` | *"Where is the harbor?"* | Text-Guided Grounding |
| `change_pure_t1.tif` + `change_pure_t2.tif` | *"What changed between these two dates?"* | Pure Bi-Temporal Change (TinyCD only) |
| `change_urban_t1.tif` + `change_urban_t2.tif` | *"Did the built-up area increase?"* | Semantic Built-Up Change |
| `change_urban_t1.tif` + `change_urban_t2.tif` | *"How much buildings were newly created?"* | Semantic Quantity (Attribution check) |
| `change_forest_t1.tif` + `change_forest_t2.tif` | *"How much vegetation was lost?"* | Semantic Vegetation Change |
| `change_water_t1.tif` + `change_water_t2.tif` | *"How much water area changed?"* | Semantic Water Change |
| `optical_multimodal.tif` + `sar_multimodal.tif` | *"Use the optical and SAR images together to identify built-up and water-covered regions."* | Cross-Modal Optical+SAR Fusion |
