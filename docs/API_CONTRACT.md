# SatQuery AI — API Contract & Schema Specification

**Document Version:** 2.0.0 (Post-Audit Revision)  
**Framework:** FastAPI / OpenAPI 3.1  
**Base URL:** `/api/v1`  
**Standard Compliance:** `AGENTS.md` Rule 8 (Standard Result Contract) and Rule 7 (Input Validation).

---

## 1. API Strategy Status & Decision Classifications

### 1.1 CONFIRMED Decisions
- **Unified Standard Result Contract:** All analysis requests return a uniform JSON schema containing: `task`, `status`, `answer`, `confidence`, `confidence_breakdown`, `evidence` (images, masks, boxes, statistics, regions), `models`, `parameters`, `warnings`, `execution_trace`, `execution_time_ms`.
- **Decoupled Architecture:** The frontend (Person B) interacts exclusively via RESTful HTTP endpoints (`/api/v1/analyze`, `/api/v1/validate`, `/api/v1/models`, `/api/v1/report/generate`) and has zero direct dependency on PyTorch or specialist model internals.
- **Physical Evidence Anchoring:** Spatial evidence includes both raster masks (PNG/GeoTIFF overlays) and vectorized GeoJSON features with real-world CRS coordinates and ground area metrics ($m^2$, ha).

### 1.2 ASSUMPTIONS
- Client applications will upload imagery as standard `multipart/form-data`.
- Intermediate preview PNGs and mask overlays are served via FastAPI static file mounts under `/api/v1/static/`.

### 1.3 TO VERIFY Items
- **Large Multipart File Upload Overhead:** Benchmark upload latency and memory buffering when transmitting $50\text{ MB}$ GeoTIFFs to FastAPI.
- **ReportLab PDF Generation Performance:** Verify PDF generation latency (< 500 ms) with embedded PNG masks on Windows.

### 1.4 RISKS
- **Temporary File Accumulation:** Uncleaned uploaded files filling up disk space over long interactive sessions.  
  *Mitigation:* Store uploaded files in temporary directory with scheduled TTL cleanup.

### 1.5 RECOMMENDED Design
- Maintain a single unified endpoint `POST /api/v1/analyze` for all agentic workflows, with optional `task_hint` for explicit testing overrides.
- Provide a dedicated pre-flight endpoint `POST /api/v1/validate` for fast client-side input checking.

---

## 2. API Endpoints Overview

| Method | Endpoint | Description | Consumes | Produces |
|---|---|---|---|---|
| `POST` | `/api/v1/analyze` | Unified agentic entrypoint. Accepts 1 or 2 images + natural language query, validates, routes, and returns evidence-grounded result. | `multipart/form-data` | `application/json` (Standard Result Contract) |
| `POST` | `/api/v1/validate` | Pre-flight validation endpoint. Inspects GeoTIFF headers, CRS, bounds, bands, and pair compatibility without running AI inference. | `multipart/form-data` | `application/json` (Validation Report) |
| `GET` | `/api/v1/models` | Registry health check and status. Returns loaded specialists, hardware device, and VRAM utilization. | None | `application/json` (Registry Status) |
| `POST` | `/api/v1/report/generate` | Generates a downloadable PDF mission summary report from a previous analysis result. | `application/json` | `application/pdf` (Binary File) |
| `GET` | `/api/v1/health` | Service liveness and readiness probe. | None | `application/json` |

---

## 3. Standard Result Contract (Rule 8)

```json
{
  "task": "bitemporal_change_analysis",
  "status": "success",
  "answer": "Between 2021-04-12 and 2023-05-18, significant new industrial construction was detected in the north-eastern sector. A total of 42,350 square meters of previously agricultural land was converted into built-up infrastructure.",
  "confidence": 0.88,
  "confidence_breakdown": {
    "input_quality_score": 0.95,
    "spatial_alignment_score": 0.92,
    "model_confidence_score": 0.89,
    "consistency_penalty": 0.00
  },
  "evidence": {
    "images": [
      {
        "role": "before_image",
        "url": "/api/v1/static/previews/t1_rgb.png",
        "width": 1024,
        "height": 1024,
        "crs": "EPSG:32643",
        "bounds": [725000.0, 3120000.0, 735240.0, 3130240.0]
      },
      {
        "role": "after_image",
        "url": "/api/v1/static/previews/t2_rgb.png",
        "width": 1024,
        "height": 1024,
        "crs": "EPSG:32643",
        "bounds": [725000.0, 3120000.0, 735240.0, 3130240.0]
      }
    ],
    "masks": [
      {
        "mask_id": "change_mask_binary",
        "label": "Detected Physical Change",
        "url": "/api/v1/static/masks/change_mask_001.png",
        "format": "image/png",
        "palette": {
          "0": "#00000000",
          "1": "#FF0033CC"
        }
      }
    ],
    "boxes": [
      {
        "box_id": "box_001",
        "label": "water body",
        "confidence": 0.88,
        "model_score": 0.88,
        "is_degenerate": false,
        "degenerate_reason": null,
        "coordinates_normalized": [0.12, 0.65, 0.45, 0.92],
        "coordinates_pixel": [123, 665, 461, 942],
        "geojson": {
          "type": "Polygon",
          "coordinates": [[[731650.0, 3128970.0], [734420.0, 3128970.0], [734420.0, 3125510.0], [731650.0, 3125510.0], [731650.0, 3128970.0]]]
        }
      }
    ],
    "statistics": [
      {
        "metric_name": "total_changed_area_m2",
        "display_name": "Total Changed Area",
        "value": 42350.0,
        "unit": "m^2"
      },
      {
        "metric_name": "changed_area_percentage",
        "display_name": "AOI Change Fraction",
        "value": 4.04,
        "unit": "%"
      }
    ],
    "regions": [
      {
        "region_name": "water body",
        "label": "water body",
        "bbox_pixel": [123.0, 665.0, 461.0, 942.0],
        "confidence": 0.88,
        "model_score": 0.88,
        "is_degenerate": false,
        "degenerate_reason": null,
        "polygon_pixel": null
      }
    ],
    "complementarity_report": null
  },
  "models": [
    {
      "identifier": "CHANGE_DETECT",
      "model_name": "TinyCD-Siamese-v1",
      "version": "1.0.0",
      "execution_time_ms": 142
    },
    {
      "identifier": "CHANGE_VQA",
      "model_name": "Qwen2-VL-2B-RS-LoRA",
      "version": "0.9.1",
      "execution_time_ms": 820
    }
  ],
  "parameters": {
    "change_threshold": 0.50,
    "min_patch_size": 25,
    "temperature": 0.2
  },
  "warnings": [],
  "execution_trace": [
    {
      "step_number": 1,
      "step_name": "InputValidation",
      "timestamp": "2026-09-07T14:22:01.102Z",
      "details": "Validated 2 GeoTIFF files. Both EPSG:32643, 10.0m resolution, spatial overlap 100%.",
      "duration_ms": 18
    },
    {
      "step_number": 2,
      "step_name": "AgenticRouting",
      "timestamp": "2026-09-07T14:22:01.120Z",
      "details": "Identified Intent: BITEMPORAL_CHANGE_ANALYSIS. Planned DAG: [GEO_PREPROCESSOR -> CHANGE_DETECT -> EVIDENCE_FUSER -> CHANGE_VQA -> CONSISTENCY_CHECKER].",
      "duration_ms": 12
    },
    {
      "step_number": 3,
      "step_name": "ModelExecution_CHANGE_DETECT",
      "timestamp": "2026-09-07T14:22:01.132Z",
      "details": "Executed TinyCD model on GPU (RTX 4070). Generated binary mask and soft probability map.",
      "duration_ms": 142
    },
    {
      "step_number": 4,
      "step_name": "EvidenceVectorization",
      "timestamp": "2026-09-07T14:22:01.274Z",
      "details": "Extracted 1 change cluster polygon and computed zonal change statistics.",
      "duration_ms": 45
    },
    {
      "step_number": 5,
      "step_name": "ModelExecution_CHANGE_VQA",
      "timestamp": "2026-09-07T14:22:01.319Z",
      "details": "Executed RS-adapted VLM conditioned on change mask. Generated semantic interpretation.",
      "duration_ms": 820
    },
    {
      "step_number": 6,
      "step_name": "ConsistencyCheck",
      "timestamp": "2026-09-07T14:22:02.139Z",
      "details": "Confirmed agreement between change detector (4.04% area) and VLM narrative (industrial development). No conflicts.",
      "duration_ms": 6
    }
  ],
  "execution_time_ms": 1043
}
```

### 3.1 Coordinate Conventions & Spatial Transformation (Milestone M2)

To guarantee zero coordinate inversion errors across models and visualization frontends, SatQuery AI strictly enforces:

- **Origin $(0, 0)$:** Top-Left corner of the raster image.
- **X direction:** Horizontal, increasing to the right: $0 \le x \le \text{width}$.
- **Y direction:** Vertical, increasing downwards: $0 \le y \le \text{height}$.
- **Bounding Box Ordering:** Strictly **`[xmin, ymin, xmax, ymax]`** across all schemas (`coordinates_pixel`, `coordinates_normalized`, `bbox_pixel`).
- **Normalized Scale:** $[0.0, 1.0]$, calculated as $(x_{\text{min}}/W, y_{\text{min}}/H, x_{\text{max}}/W, y_{\text{max}}/H)$.
- **Geospatial GeoJSON Feature:** Generated via deterministic affine forward projection:
  $$\begin{bmatrix} X_{\text{proj}} \\ Y_{\text{proj}} \end{bmatrix} = \begin{bmatrix} a & b & c \\ d & e & f \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$
  Yields a 5-point closed polygon: `[[TL], [TR], [BR], [BL], [TL]]`.
- **Honesty Contract:** Axis-aligned bounding box polygons are documented with `derivation="axis_aligned_bounding_box"`. The system never outputs bounding boxes masquerading as semantic segmentation masks.

---

## 4. Optical-SAR Complementarity Schema Addition

When executing cross-modal analysis (`OPTICAL_SAR_ANALYSIS`), the `evidence` object populates the `complementarity_report` field:

```json
"complementarity_report": {
  "optical_limitations": "Dense cloud cover obscuring 34.2% of the central region.",
  "sar_penetration": "C-band microwave radar successfully penetrated cloud cover, resolving runways and buildings.",
  "structural_contrast": "SAR double-bounce backscatter identified high-density concrete structures indistinguishable in low-contrast optical shadows.",
  "layers": [
    {
      "layer_id": "sar_cloud_penetrated_features",
      "label": "Features Detected Through Cloud",
      "url": "/api/v1/static/masks/sar_penetration_001.png"
    },
    {
      "layer_id": "sar_roughness_contrast",
      "label": "Microwave Roughness & Double-Bounce",
      "url": "/api/v1/static/masks/sar_roughness_001.png"
    }
  ]
}
```

---

## 5. Standard Error Registry
- `INVALID_FILE_FORMAT`: File is not a readable TIFF, GeoTIFF, PNG, or JPEG.
- `CORRUPTED_METADATA`: GeoTIFF headers are unparseable or corrupted.
- `CRS_UNRECOGNIZED`: Coordinate Reference System cannot be resolved into an EPSG code or WKT string.
- `CRS_MISMATCH_NON_REPROJECTABLE`: Images have different coordinate systems and lack transform bounds.
- `NO_SPATIAL_OVERLAP`: Overlap between pairs is $< 20\%$.
- `TEMPORAL_ORDER_INVALID`: Image $t_2$ acquired before $t_1$ for forward change analysis.
- `MODALITY_INCOMPATIBLE`: Selected task requires SAR but received two optical images (or vice versa).
- `GPU_OUT_OF_MEMORY`: System VRAM exceeded; operation fell back to CPU or failed.
- `SPECIALIST_UNAVAILABLE`: Model specialist failed health check and no fallback exists.
