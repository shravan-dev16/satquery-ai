# SatQuery AI — System Failure Taxonomy

**Document Version:** 1.0.0 (Post-M7 Architecture Lock)  
**Standard:** Compliance with `AGENTS.md` Rule 10 (Consistency Checking), Rule 11 (Confidence Principle), Rule 24 (Graceful Fallbacks), Rule 31 (Failure-Handling Principle), and SIH26167 Requirements.

---

## 1. Executive Purpose & Principles

SatQuery AI does not conceal errors, manufacture certainty, or fabricate results when evidence is insufficient or contradictory. 

In accordance with **Rule 31** (*"A qualified answer is better than a confidently wrong answer"*), this document establishes a standardized, 12-category failure taxonomy spanning the entire end-to-end pipeline:
1. Every failure must be classified deterministically rather than swallowed silently.
2. Incomplete or degraded states must be explicitly signaled to downstream modules (`warnings`, `trace`, `confidence_breakdown`).
3. Future milestone evaluations (M8 consistency, M9 confidence calibration, M10 domain adaptation) must classify observed test discrepancies against these codes.

---

## 2. Canonical Failure Taxonomy Matrix (F1 – F12)

| Code | Failure Classification | Pipeline Stage | Primary Trigger / Condition | Handling Strategy | Downstream Impact |
|---|---|---|---|---|---|
| **F1** | **Input Invalidity** | `InputValidator` / Ingestion | Unreadable file, corrupt header, unsupported container format, missing bands, invalid dimensions ($\le 0$). | Reject immediately with HTTP 400/422; return structured validation error. | Pipeline terminates before inference; zero GPU compute wasted. |
| **F2** | **Geospatial Incompatibility** | `BiTemporalValidator`, `CrossModalValidator` | Missing CRS, invalid affine geotransform, disjoint spatial coverage, non-overlapping bounding boxes, $<20\%$ spatial intersection. | Reject pair analysis deterministically; abort alignment. | Prevents invalid co-registration; terminates bitemporal or cross-modal pipeline. |
| **F3** | **Modality Mismatch** | `ModalityDetector`, Alignment | Sensor modality does not match specialist requirements (e.g., dual-optical submitted to optical-SAR specialist; SAR submitted to optical VQA). | Auto-route if resolvable (swap pair order with trace note); reject with explicit error if unsupported. | Pipeline aborted or routed to single-modality fallback with audit warning. |
| **F4** | **Model Execution Failure** | `ModelRegistry`, Specialist Runners | Out-of-memory (OOM), CUDA runtime error, checkpoint loading failure, hardware timeout. | Catch exception, log trace, trigger Rule 24 fallback if registered (e.g. CVA fallback for TinyCD), or return graceful failure contract. | Marks specialist as failed in `ExecutionTrace`; applies confidence penalty. |
| **F5** | **Spatial Detection Failure** | `CHANGE_DETECT`, `RS_GROUND` | Full-image collapse ($\ge 98\%$ mask coverage), empty detection when referring expression is unambiguous, degenerate bounding boxes. | Apply degeneracy filter; replace degenerate box/mask with empty evidence bundle and warning. | Warning appended to `StandardResultContract.warnings`; confidence reduced. |
| **F6** | **Semantic Interpretation Failure** | `CHANGE_VQA`, `RS_VQA` | VLM hallucination on zero-change scene, unparseable markdown/JSON output, vague generic response. | Enforce Zero-Change Short-Circuit gate; JSON schema repair parser; default ambiguous classes to `"unknown"`. | Returns structured fallback with `semantic_uncertainty="high"`. |
| **F7** | **Optical/SAR Disagreement** | `OPTICAL_SAR_FUSION` | Optical spectrum indicates water/vegetation but SAR backscatter indicates dense structural roughness (or vice versa); cloud occlusion. | Tag spatial region with `is_cross_modal_conflict=True`; document discrepancy in `ComplementarityReport`. | Discrepancy documented in narrative; region marked as conflicted. |
| **F8** | **Evidence Contradiction** | M8 Consistency Engine | Physical change mask indicates massive change ($>30\%$) but VLM asserts no change occurred; or detector says 0% while text claims new construction. | Flag contradiction in `EvidenceBundle`; append `ConsistencyWarning`; trigger confidence penalty. | System surfaces contradiction explicitly; prevents confident false consensus. |
| **F9** | **Confidence Uncertainty** | M9 Confidence Engine | Low radiometric contrast, severe cloud cover, extreme off-nadir angle, uncalibrated probability distribution. | Emit `system_confidence < 0.40`; append explicit disclaimer advising analyst manual verification. | Report marks findings as *Provisional / Analyst Review Required*. |
| **F10** | **Unsupported Sensor / Domain** | Ingestion & Modality | Imagery originating from unvalidated satellite platforms (hyperspectral, thermal, non-standard polarimetry) or unvalidated radiometric ranges. | Issue domain warning (`"Uncalibrated sensor profile; inference quality may degrade"`). | Retains execution but flags domain divergence in execution trace. |
| **F11** | **Benchmark Failure** | Evaluation Harness | Specialist performance drops below minimum required metric thresholds on standard benchmark splits (e.g., IoU $< 0.50$, F1 $< 0.60$). | Halt automated promotion; log per-sample confusion matrix to benchmark failure report. | Triggers model recalibration or domain fine-tuning (M10). |
| **F12** | **Generalization Failure** | Evaluation Harness | Model verified on one domain/sensor fails drastically on unseen geographic biome or land-cover category (e.g. LEVIR-CD model failing on agricultural crop phenology). | Classify under `FailureCategory` (e.g. `AGRICULTURE_ERROR`, `SEASONAL_VARIATION`); log in benchmark report. | Documents boundary of operational deployment; sets adaptation target for M10. |

---

## 3. Failure Mode Diagnostic Procedures

### 3.1 Input & Preprocessing Diagnostics (F1 – F3)
When an input failure occurs:
- Verify GeoTIFF metadata via `backend/preprocessing/metadata.py`.
- Confirm affine transform via `rasterio.transform.Affine`.
- Check EPSG code validity using `pyproj.CRS.from_user_input`.
- Check intersection calculation via `backend/preprocessing/alignment.py`.

### 3.2 Specialist Runtime Diagnostics (F4 – F7)
When specialist execution or spatial evidence exhibits anomalies:
- Check GPU VRAM allocation via `torch.cuda.memory_allocated()`.
- Inspect raw prediction mask coverage: if `changed_pixels / total_pixels >= 0.98`, trigger degenerate mask filter.
- In cross-modal analysis, cross-check optical `ExG` index against SAR local standard deviation texture $\sigma_{\text{local}}$.

### 3.3 Multi-Model Fusion Diagnostics (F8 – F9)
- For bi-temporal pipelines, verify alignment between `change_statistics.changed_area_ha` and `semantic_interpretation.temporal_direction`.
- Any directional divergence (`changed_pixels > 1000` paired with `temporal_direction == "no_change"`) is classified as **F8 Evidence Contradiction**.

---

## 4. Architectural Rules on Failure Transparency

1. **No Silent Swallowing:** Never wrap specialist execution in a bare `except: pass` that returns a fake empty result with high confidence.
2. **Explicit User Warnings:** All failures and fallbacks must be appended to the `warnings` array of the `StandardResultContract`.
3. **Trace Auditability:** Every step that encounters a failure or activates a fallback must record `status="warning"` or `status="fallback"` with explicit rationale in the `ExecutionTrace`.
4. **Permanent Record:** All benchmark and evaluation runs must dump failure classifications into structured JSON reports (`docs/evaluation/*_report.json`).
