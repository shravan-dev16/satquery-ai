# Milestone M11 — Reporting & Evidence Packaging

> **Standard:** Canonical Architecture Document  
> **Milestone Status:** COMPLETE ✅  
> **Verification Suite:** 19 dedicated unit/integration tests passing; 195 repository tests passing (100% green); 5 real end-to-end workflows validated.  
> **Governing Problem Statement:** Smart India Hackathon 2026 — SIH26167 / Indian Space Research Organisation (ISRO).

---

## 1. Milestone Purpose & Scope Boundary

Milestone **M11 (Reporting & Evidence Packaging)** serves as the canonical audit and presentation layer of SatQuery AI. It translates the disparate outputs produced across the specialist pipeline:

- **M7 Dynamic Orchestrator:** Task intent, target specialist selection, and multi-stage DAGs.
- **Specialist Models:** Physical masks, bounding boxes, class segmentations, and semantic narratives (`RS_VQA`, `RS_GROUND`, `CHANGE_DETECT`, `CHANGE_VQA`, `OPTICAL_SAR_FUSION`).
- **M8 Evidence Normalizer, Fuser & Consistency Checker:** Standardized evidence items, provenance metadata, cross-model conflict detection (Rules C1–C8), and reliability gating.
- **M9 Defensible System Confidence Engine:** Multi-factor heuristic scoring (input quality, alignment, model confidence, evidence completeness, contradiction caps).
- **Physical Preview Artifacts:** Static visual assets in `/api/v1/static/previews/`.

### Strict Architectural Boundaries
1. **Packaging, Not Inference:** M11 executes **ZERO new model inference passes** and performs **ZERO model selection**.
2. **Preservation, Not Recomputation:** M11 preserves M9 system confidence and M8 consistency statuses exactly as evaluated upstream.
3. **No Uncalibrated Probability Claims (Rule 11):** M9 produces a heuristic system confidence score. It is never labeled as a calibrated probability or probability of correctness.
4. **Backend Focus:** M11 packages structured data consumed seamlessly by both the REST API and future frontend interfaces without forcing internal model logic onto the UI layer.

---

## 2. Canonical Architecture

```
User Query + Input Imagery (GeoTIFF / TIFF)
                     │
                     ▼
          Input Pre-Flight Validation
                     │
                     ▼
       M7 Agentic Router & Planner (DAG)
                     │
                     ▼
          Specialist Model Execution
     (VQA / Grounding / TinyCD / Optical-SAR)
                     │
                     ▼
     M8 Evidence Normalization & Fusion
                     │
                     ▼
       M8 Cross-Model Consistency Checking
                     │
                     ▼
        M9 Defensible System Confidence
                     │
                     ▼
       StandardResultContract Assembly
                     │
                     ▼
      ╔═══════════════════════════════════╗
      ║      M11 REPORT BUILDER           ║
      ║  (backend/reports/builder.py)     ║
      ║                                   ║
      ║  - Input Summary Extraction       ║
      ║  - Visual Evidence Indexing       ║
      ║  - Quantitative Metrics Synthesis ║
      ║  - M9 Confidence Audit Summary    ║
      ║  - M8 Consistency & Gating Audit  ║
      ║  - Specialist Provenance Ledger   ║
      ║  - Chronological Trace Assembly   ║
      ╚═══════════════════════════════════╝
                     │
                     ▼
              AnalystReport
      (JSON Schema / Markdown Briefing)
                     │
                     ▼
        StandardResultContract.report
                     │
                     ▼
             POST /api/v1/analyze
```

---

## 3. Data Contract & Schema Architecture

The reporting contract is defined in `backend/reports/schema.py` and exported via `backend/reports/__init__.py`.

### 3.1 `AnalystReport` Root Schema

```json
{
  "metadata": {
    "report_id": "REP_single_image_vqa_1788944822_90d0ca",
    "timestamp": "2026-09-09T09:07:02.123456Z",
    "mission": "Smart India Hackathon 2026 / SIH26167",
    "organization": "Indian Space Research Organisation (ISRO)",
    "app_version": "0.3.0",
    "execution_time_ms": 135
  },
  "task": "single_image_vqa",
  "answer": "The image depicts a combination of land cover types. The dominant feature is a large green area...",
  "input_summary": {
    "query": "What type of land cover and structures are visible in this scene?",
    "input_count": 1,
    "images": [
      {
        "filename": "optical_s2_sample.tif",
        "role": "primary",
        "dimensions": [256, 256],
        "band_count": 4,
        "crs": "EPSG:32643",
        "resolution": [10.0, 10.0],
        "modality": "optical",
        "bounds": [500000.0, 1400000.0, 502560.0, 1402560.0]
      }
    ],
    "temporal_ordering": null,
    "spatial_overlap_percentage": null
  },
  "visual_evidence": [
    {
      "id": "vis_src_1",
      "type": "source_image",
      "label": "Input Image 1 (primary)",
      "path_or_url": "/api/v1/static/previews/optical_s2_sample.tif",
      "description": "Ingested raster imagery (EPSG:32643)",
      "format": "image/png"
    }
  ],
  "statistics": {
    "metrics": {},
    "zonal_statistics": [],
    "summary_notes": []
  },
  "confidence": {
    "score": 0.83,
    "level": "HIGH",
    "is_calibrated_probability": false,
    "method": "evidence-weighted confidence heuristic",
    "supporting_factors": [
      "Standard projected CRS verified (EPSG:32643)",
      "Optimal ground resolution (10.0m)",
      "High specialist certainty (1.00)"
    ],
    "warnings": [],
    "breakdown": { ... }
  },
  "consistency": {
    "status": "CONSISTENT",
    "is_gated": false,
    "gating_action": "allow",
    "conflicts_count": 0,
    "conflicts": [],
    "summary_narrative": "Single-specialist output verified; no multi-source discrepancy detected."
  },
  "provenance": [
    {
      "evidence_id": "prov_rs_vqa_1",
      "source_specialist": "RS_VQA",
      "model_name": "Remote-Sensing VQA (Qwen/Qwen2-VL-2B-Instruct)",
      "version": "1.0.0",
      "claim": "The image depicts a combination of land cover types...",
      "evidence_type": "specialist_inference",
      "input_files": ["optical_s2_sample.tif"],
      "execution_stage": "specialist_execution",
      "raw_confidence": 1.0,
      "provenance_details": { "execution_time_ms": 120 }
    }
  ],
  "execution_trace": [
    {
      "step_number": 1,
      "step_name": "InputValidation",
      "status": "completed",
      "details": "Validated GeoTIFF headers: 256x256, 4 bands, EPSG:32643.",
      "duration_ms": 8
    }
  ],
  "warnings": []
}
```

---

## 4. Sub-Component Breakdown

### 4.1 Input Summary & Pre-Flight Validation (`InputSummary`)
Captures all physical dimensions, spatial resolution, CRS, bounding boxes, sensor modalities, and (for multi-temporal tasks) temporal ordering strings and spatial co-registration overlap percentages.

### 4.2 Visual Evidence Index (`VisualEvidenceReference`)
Normalizes all visual artifacts generated by specialists:
- `source_image`: Ingested rasters or normalized RGB visualizations.
- `mask`: Binary physical change masks or multi-class semantic segmentation masks.
- `overlay`: Transparent composite masks overlaid on post-change imagery.
- `composite`: Multi-panel split comparison frames or Optical-SAR false-color composites.

All URLs resolve to static endpoints mounted at `/api/v1/static/previews/{filename}`.

### 4.3 Quantitative Statistics (`QuantitativeStatistics`)
Extracts grounded physical and spatial telemetry:
- **Change Detection:** `changed_pixels`, `physical_area_m2`, `physical_area_ha`, `change_ratio_pct`.
- **Text-Guided Grounding:** `detected_boxes_count`, `valid_boxes_count`, `max_box_confidence`.
- **Semantic Change:** `predominant_transition`, `temporal_direction`, `semantic_uncertainty`.
- **Optical-SAR Fusion:** `complementary_layers_count`, class distribution breakdown.

### 4.4 Defensible Confidence Summary (`ConfidenceSummary`)
Transparent disclosure adhering to AGENTS.md Rule 11:
- Explicit field `is_calibrated_probability = False`.
- Discloses categorical tier: `HIGH`, `MEDIUM`, `LOW`, or `UNSUPPORTED`.
- Itemizes supporting factors (e.g. valid CRS, high overlap) and dampening penalties (e.g. poor resolution, missing metadata, conflict penalties).

### 4.5 Multi-Source Consistency Summary (`ConsistencySummary`)
Exposes M8 consistency checker telemetry:
- Observable status: `CONSISTENT`, `PARTIALLY_CONSISTENT`, `UNCERTAIN`, `CONTRADICTORY`, `INSUFFICIENT_EVIDENCE`.
- Itemizes detected `EvidenceConflict` records, including `rule_violated`, `severity`, and `conflicting_sources`.
- Discloses `gating_action` (`allow`, `qualify`, `flag_contradiction`, `state_insufficient`).

### 4.6 Specialist Provenance Ledger (`SpecialistProvenanceEntry`)
Ensures complete auditability for every claim:
- Maps each finding to its `source_specialist` (`CHANGE_DETECT`, `RS_GROUND`, `OPTICAL_SAR_FUSION`, etc.).
- Discloses underlying `model_name` and `version`.
- Retains `raw_confidence` reported by the model prior to system heuristic synthesis.

---

## 5. Supported Workflows

| Workflow | Specialists Invoked | Report Key Outputs |
|---|---|---|
| **A. Single-Image VQA** | `RS_VQA` (LoRA-adapted Qwen2-VL) | Analytical narrative, input CRS/resolution metadata, model provenance, trace. |
| **B. Text-Guided Grounding** | `RS_GROUND` (Grounding DINO) | Bounding box coordinates `[xmin, ymin, xmax, ymax]`, box count, valid count, preview link. |
| **C. Bi-Temporal Change Detection** | `CHANGE_DETECT` (TinyCD / CVA) | Changed pixels, area (ha), binary change mask, fused overlay, temporal order, overlap %. |
| **D. Optical + SAR Joint Analysis** | `OPTICAL_SAR_FUSION` | Microwave penetration layers, roughness layers, joint class distribution, dual-sensor provenance. |
| **E. Combined Killer Workflow** | `CHANGE_DETECT` + `CHANGE_VQA` | Physical change mask + semantic transition narrative (`built_structure -> built_structure`), unified trace. |

---

## 6. Failure Handling & Anti-Hallucination Integrity

1. **Contradictory Evidence:** When physical detection conflicts with language interpretation (Rule C1), the report preserves the contradiction, retains the conflict record in `consistency.conflicts`, reduces confidence, and highlights the gating action.
2. **Insufficient Evidence:** When image overlap is insufficient (<30%) or images lack projection, the report records `INSUFFICIENT_EVIDENCE` and suppresses confident claims.
3. **Zero-Change Gating:** When TinyCD detects zero surface change, semantic interpretation is suppressed by the zero-change gate, and the report clearly notes that no change was detected exceeding operational thresholds.
4. **Fallback Handler:** If malformed specialist data is encountered, `ReportBuilder._build_fallback_report` produces a safe, structured report with diagnostic warnings rather than raising an unhandled 500 error.

---

## 7. Multi-Format Serialization

### 7.1 Machine-Readable JSON
Available via `AnalystReport.to_json(indent=2)` or directly serialized via FastAPI response model in `POST /api/v1/analyze`.

### 7.2 Human-Readable Markdown Briefing
Available via `AnalystReport.to_markdown()` (implemented in `backend/reports/markdown.py`):
- Executive summary with question and primary finding.
- Defensible confidence and consistency badges.
- Ingested raster validation table.
- Quantitative measurements and zonal indicators.
- Visual evidence links to preview assets.
- Specialist provenance ledger table.
- Chronological execution audit trace table.
- Operational limitations and warnings.

---

## 8. Verification Results

### Unit & Integration Test Suite (`tests/test_reporting.py`)
- `test_single_image_vqa_report`: **PASSED**
- `test_grounding_report`: **PASSED**
- `test_bitemporal_change_report`: **PASSED**
- `test_optical_sar_report`: **PASSED**
- `test_combined_multi_specialist_report`: **PASSED**
- `test_confidence_packaging`: **PASSED**
- `test_consistency_packaging`: **PASSED**
- `test_evidence_provenance_ledger`: **PASSED**
- `test_visual_artifact_references`: **PASSED**
- `test_execution_trace_preservation`: **PASSED**
- `test_contradictory_evidence_handling`: **PASSED**
- `test_insufficient_evidence_handling`: **PASSED**
- `test_missing_optional_evidence_robustness`: **PASSED**
- `test_specialist_failure_handling`: **PASSED**
- `test_malformed_contract_fallback_safety`: **PASSED**
- `test_json_serialization_roundtrip`: **PASSED**
- `test_markdown_export`: **PASSED**
- `test_api_analyze_returns_m11_report`: **PASSED**
- `test_report_determinism`: **PASSED**

**Full Repository Test Suite:**
```
================ 195 passed, 285 warnings in 191.09s (0:03:11) ================
```
Zero failures. Zero regressions.
