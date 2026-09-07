# SatQuery AI — Requirements Traceability Matrix (RTM)

**Document Version:** 2.0.0 (Post-Audit Revision)  
**Problem Statement:** Smart India Hackathon — SIH26167  
**Organization:** Indian Space Research Organisation (ISRO)  
**Standard:** Every mandatory SIH requirement maps strictly to:
`REQUIREMENT -> MODULE -> MODEL/TOOL -> DATASET -> TEST -> DEMO -> EVIDENCE`

---

## 1. Traceability Matrix Overview

| Req ID | Mandatory SIH26167 Requirement | Backend Module | Specialist Model / Tool | Reference Dataset | Automated Test Suite | Target Demo | Grounded Evidence Produced |
|---|---|---|---|---|---|---|---|
| **REQ-01** | Single optical/multispectral image analysis | `backend/preprocessing/geotiff.py`, `normalization.py` | `GeoPreprocessor` (radiometric scaling, NDVI/NDWI, band extraction) | BigEarthNet-S2, Sentinel-2 L2A | `tests/test_validation.py::test_optical_bands` | DEMO 1 & 2 | Band histograms, true/false color rendering, spectral index maps |
| **REQ-02** | Single SAR image analysis | `backend/preprocessing/geotiff.py`, `modality.py` | `SARPreprocessor` (speckle filter, dB scaling, VV/VH ratio) | BigEarthNet-S1, Sentinel-1 GRD | `tests/test_validation.py::test_sar_polarization` | DEMO 4 | Calibrated dB backscatter map, polarization ratio layer ($VV/VH$) |
| **REQ-03** | Single-image Visual Question Answering (VQA) | `backend/models/vqa.py`, `agent/router.py` | `RS_VQA` (`Qwen2-VL-2B` / `Florence-2` + RS LoRA) | RSVQAxBEN (BigEarthNet), VRSBench | `tests/test_models.py::test_rs_vqa_inference` | DEMO 1 | Natural language answer, token likelihood, confidence score |
| **REQ-04** | Text-guided Region Grounding & Captioning | `backend/models/grounding.py`, `models/caption.py` | `RS_GROUND` & `RS_CAPTION` (`Florence-2-base`) | VRSBench (grounding & captions) | `tests/test_models.py::test_rs_grounding` | DEMO 2 | Bounding boxes `[ymin, xmin, ymax, xmax]`, GeoJSON polygon boundaries, descriptive text |
| **REQ-05** | Bi-temporal image-pair change analysis | `backend/preprocessing/alignment.py`, `models/change.py` | `CHANGE_DETECT` (`TinyCD` / `BIT` Siamese CNN) | LEVIR-CD (building change) | `tests/test_models.py::test_change_detection` | DEMO 3 | Pixel change probability mask, difference overlay, exact change area ($m^2$, %) |
| **REQ-06** | Change description or change-based VQA | `backend/models/change_vqa.py`, `evidence/fusion.py` | `CHANGE_VQA` (Mask-conditioned RS-VLM) | LEVIR-CC, CDVQA | `tests/test_models.py::test_change_vqa` | DEMO 3 | Natural language narrative of what changed, affected land classes, change metrics |
| **REQ-07** | Optical + SAR cross-modal analysis | `backend/models/optical_sar.py`, `alignment.py` | `OPTICAL_SAR_ANALYSIS` (Physics-grounded complementary fusion) | Sen1-2, BigEarthNet-MM | `tests/test_models.py::test_optical_sar_fusion` | DEMO 4 | Cloud-penetrated surface layer, structural contrast map, complementarity report |
| **REQ-08** | Remote-Sensing adaptation / fine-tuning | `backend/evaluation/`, `scripts/train_adapter.py` | LoRA adapter on RSVQAxBEN (BigEarthNet) / VRSBench | RSVQAxBEN, BigEarthNet-S2, LEVIR-CD | `tests/test_models.py::test_adapted_weights_loaded` | DEMO 1 & 2 | Adapter checkpoint, verifiable training log, loss curve, evaluation accuracy |
| **REQ-09** | Agentic task interpretation & tool selection | `backend/agent/parser.py`, `router.py`, `executor.py` | `QueryIntentParser` + `AgenticRouter` DAG | Query intent test corpus | `tests/test_router.py::test_task_routing` | DEMO 1, 2, 3, 4 | Structured Intent object, dynamic tool execution plan, parameter bindings |
| **REQ-10** | Input compatibility validation | `backend/preprocessing/geotiff.py`, `alignment.py` | `InputValidator` (CRS, transform, resolution, overlap) | Multi-sensor test GeoTIFF fixtures (valid/invalid) | `tests/test_validation.py::test_pair_compatibility` | DEMO 1, 2, 3, 4 | Detailed validation report, CRS reprojection status, spatial overlap % |
| **REQ-11** | Evidence-grounded output | `backend/evidence/fusion.py` | `EvidenceFuser` (raster-to-vector, zonal metrics) | LEVIR-CD, VRSBench fixtures | `tests/test_consistency.py::test_evidence_generation` | DEMO 1, 2, 3, 4 | Visual masks, bounding boxes, numeric area ($m^2$ / ha), GeoJSON coordinates |
| **REQ-12** | Confidence information | `backend/evidence/confidence.py` | `ConfidenceEstimator` (defensible multi-factor formula) | Multi-modal test scenarios (clean vs degraded) | `tests/test_consistency.py::test_confidence_scoring` | DEMO 1, 2, 3, 4 | Calibrated confidence score $[0.0, 1.0]$ broken down by factor (input, alignment, model) |
| **REQ-13** | Auditable execution trace | `backend/agent/trace.py` | `ExecutionTraceRecorder` (observable summary logger) | Pipeline execution runs | `tests/test_router.py::test_execution_trace_audit` | DEMO 1, 2, 3, 4 | Step-by-step observable JSON trace with timestamps, tool names, parameters, warnings |
| **REQ-14** | Downloadable reports | `backend/reports/pdf_generator.py`, `json_export.py` | `ReportGenerator` (ReportLab PDF & JSON exporter) | Completed analysis pipelines | `tests/test_api.py::test_report_download` | DEMO 1, 2, 3, 4 | Downloadable PDF document with metadata, images, masks, trace, and findings; standalone JSON bundle |

---

## 2. SIH Demo Scenarios Traceability

| Demo | Title | Key Tasks Invoked | Inputs | Expected Observable Evidence in UI |
|---|---|---|---|---|
| **DEMO 1** | Single-Image RS VQA | Input Validation $\to$ Modality Check $\to$ `RS_VQA` $\to$ Confidence Estimation | 1 Optical GeoTIFF + Query: *"What type of infrastructure is visible in the center?"* | Answer text, model confidence breakdown, spectral band viewer, execution trace. |
| **DEMO 2** | Text-Guided Region Grounding | Input Validation $\to$ `RS_GROUND` $\to$ Vectorize to GeoJSON $\to$ Zonal Metrics | 1 Optical/SAR GeoTIFF + Query: *"Ground all aircraft on the tarmac"* | Bounding boxes around aircraft, detected object count, GeoJSON coordinate list, execution trace. |
| **DEMO 3** | Bi-Temporal Change Analysis | Validation $\to$ Alignment $\to$ `CHANGE_DETECT` $\to$ `CHANGE_VQA` $\to$ Consistency Check | Pair of GeoTIFFs ($t_1, t_2$) + Query: *"What changed between these two dates?"* | Before/after slider, binary change mask, change area in $m^2$ and %, change caption narrative, consistency check status, PDF report download. |
| **DEMO 4** | Optical + SAR Joint Analysis | Validation $\to$ Co-registration $\to$ `OPTICAL_SAR_ANALYSIS` $\to$ Evidence Fusion | 1 Optical GeoTIFF + 1 SAR GeoTIFF + Query: *"Identify built-up regions using optical and SAR together"* | Optical RGB view, SAR dB backscatter view, complementary fusion map highlighting cloud-penetrated structures, execution trace. |
