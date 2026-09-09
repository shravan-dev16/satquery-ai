# SatQuery AI — Permanent Architecture Lock

**Standard:** Canonical Architecture Lock  
**Milestone Completion:** M7 Complete (137 passed tests, 0 failures, 0 regressions)  
**Next Milestone:** M8 (Evidence Fusion & Consistency Engine)  
**Governing Authority:** Smart India Hackathon — SIH26167 / Indian Space Research Organisation (ISRO)  

---

## 1. Project Purpose & System Identity

SatQuery AI is an **agentic multimodal remote-sensing analysis system** designed for Smart India Hackathon 2026 Problem Statement 26167.

> [!IMPORTANT]
> **SatQuery AI is NOT a generic chatbot with an image attached.**  
> It is an evidence-grounded remote-sensing AI analyst that ingests single, paired, multimodal, and bi-temporal Earth-observation imagery, interprets natural-language queries, dynamically sequences specialist remote-sensing models, and outputs verifiable visual, spatial, and numerical evidence.

---

## 2. Canonical System Architecture

```
                         USER
                          │
                          ▼
              Natural Language Query
                          +
                    Image Inputs
                          │
                          ▼
                 INPUT INSPECTION
            (Modality, CRS, Dimensions)
                          │
                          ▼
                 M7 ORCHESTRATOR
         (AgentRouter -> AgentPlanner -> AgentExecutor)
                          │
                 ┌────────┼────────┐
                 │        │        │
                 ▼        ▼        ▼
              SINGLE   BI-TEMP   OPTICAL+SAR
               IMAGE    ORAL        │
                 │        │         │
                 ▼        ▼         ▼
             RS_VQA /  CHANGE     M6
             RS_GROUND DETECT    FUSION
                        │
                        ▼
                    CHANGE_VQA
                        │
                 SPECIALIST OUTPUTS
                        │
                        ▼
                M8 EVIDENCE FUSION
                + CONSISTENCY
                        │
                        ▼
                M9 CONFIDENCE
                        │
                        ▼
                  FINAL ANSWER
              + visual evidence
              + spatial evidence
              + confidence
              + trace
                        │
                        ▼
                 M11 REPORTS
```

- **M10 Remote-Sensing Adaptation** improves the underlying visual specialists via domain-specific fine-tuning (BigEarthNet / VRSBench).
- The system remains **specialist-based**. It will **never** be collapsed into an unconstrained single multimodal VLM prompt.

---

## 3. Permanent Milestone State

- **M0 — Scaffolding & Contracts:** **COMPLETE** ✅ (BaseSpecialist, ModelRegistry, StandardResultContract, schemas)
- **M1 — Ingestion + VQA:** **COMPLETE** ✅ (GeoTIFF validation, metadata, baseline VQA)
- **M2 — Grounding:** **COMPLETE** ✅ (Text-guided spatial bounding boxes, coordinate projection)
- **M2.1 — Grounding Evaluation:** **COMPLETE** ✅ (Grounding DINO recovery, VRSBench subset evaluation)
- **M3 — Bi-Temporal Change Foundation:** **COMPLETE** ✅ (11-point validator, sub-pixel aligner, TinyCD + CVA)
- **M4 — Change Detection Benchmark:** **COMPLETE** ✅ (LEVIR-CD evaluation, diverse-scene generalization)
- **M5 — Semantic Change Interpretation:** **COMPLETE** ✅ (ChangeVQASpecialist, 3-panel composite, zero-change gate)
- **M6 — Optical-SAR Joint Analysis:** **COMPLETE** ✅ (CrossModalValidator, CrossModalAligner, 6-class physics fusion)
- **M7 — Agentic Orchestration & Dynamic Routing:** **COMPLETE** ✅ (Router, Planner, Executor, 16-case matrix)
- **M8 — Evidence Fusion & Consistency Engine:** **COMPLETE** ✅ (Rules C1–C8, EvidenceGater, 5-state reliability)
- **M9 — Defensible System Confidence Engine:** **COMPLETE** ✅ (Multi-factor heuristic, contradiction caps)
- **M10 — Remote-Sensing Adaptation (LoRA):** **COMPLETE** ✅ (Qwen2-VL PEFT LoRA, +6.25 pp overall, +28.57 pp change VQA)
- **M11 — Report Generation & Evidence Packaging:** **COMPLETE** ✅ (Canonical AnalystReport, JSON/Markdown export, 195 tests)

#### **NEXT MILESTONE:**
- **M12 — Final UI Polish (Teammate branch)**
- **M13 — End-to-End Hardening & Demo Rehearsal**

---

## 4. Canonical Specialist Inventory

The repository contains the following registered specialists in `backend/agent/registry.py`:

| Identifier | Class Name | Location | Operational Status | Technical Role |
|---|---|---|---|---|
| `RS_VQA` | `RemoteSensingVQASpecialist` | `backend/models/vqa.py` | **Active Production** | Single-image visual question answering (Florence-2 / BLIP-2) |
| `RS_GROUND` | `RemoteSensingGroundingSpecialist` | `backend/models/grounding.py` | **Active Production** | Text-guided region grounding with pixel & geo-polygons (Grounding DINO) |
| `CHANGE_DETECT` | `ChangeDetectionSpecialist` | `backend/models/change.py` | **Active Production** | Binary physical surface change detection (TinyCD neural + CVA fallback) |
| `CHANGE_VQA` | `ChangeVQASpecialist` | `backend/models/change_vqa.py` | **Active Production** | Conditioned semantic change interpretation across verified clusters |
| `OPTICAL_SAR_FUSION` | `OpticalSARSpecialist` | `backend/models/optical_sar.py` | **Active Production** | Cross-modal feature extraction and joint land-cover classification |
| `RS_CAPTION` | `RemoteSensingCaptionSpecialist` | `backend/models/caption.py` | **Interface Stub (M2-B)** | Architectural placeholder registering capability interface; full captioning implementation scheduled for future milestone |

> [!NOTE]
> `RS_CAPTION` is an architectural interface stub implemented during Milestone M2-B to establish schema compliance. It is not currently claimed as an evaluated production model.

---

## 5. Non-Negotiable Architectural Principles

1. **Evidence-First Architecture:** Text answers must be grounded in observable spatial evidence (masks, bounding boxes, zonal statistics, preview overlays).
2. **Specialist Separation:** Tasks requiring different inductive biases (detection vs reasoning vs fusion) must use specialized tools.
3. **Deterministic Validation Before Inference:** Raster readability, CRS, dimensions, resolution, and temporal ordering must be verified before model invocation.
4. **Explicit Modality Awareness:** Optical reflectance and SAR backscatter are physically distinct and must never be treated interchangeably.
5. **Temporal Detection Separated from Semantic Interpretation:** TinyCD determines *where* physical change happened; ChangeVQA determines *what* transformed.
6. **Optical and SAR Evidence Remain Distinguishable:** Joint analysis outputs must retain separate optical and SAR contributions alongside fused results.
7. **Specialist Confidence $\neq$ Final System Confidence:** Raw model scores are evidence signals; the system confidence engine (M9) synthesizes alignment, model, and consistency factors.
8. **Unsupported Claims Must Not Be Presented as Fact:** If data resolution is insufficient or models disagree, the system must emit explicit warnings.
9. **Observable Execution Trace:** Every response returns an auditable timeline of operational steps with timestamps, durations, and statuses.
10. **ModelRegistry is Canonical:** All tools register in `ModelRegistry`; hardcoded model class instantiations across controllers are forbidden.
11. **StandardResultContract is Canonical:** All specialist outputs conform strictly to the invariant Pydantic contract schema.
12. **Graceful Failure:** Incompatible inputs emit actionable HTTP 400 explanations without crashing or silent reinterpretation.
13. **Remote-Sensing Adaptation Must Be Measurable:** Adaptation claims must report base model, dataset, subset, hyperparameters, and held-out metrics.
14. **Synthetic Tests Do Not Establish Real-World Accuracy:** Synthetic fixtures validate pipeline logic only; real RS imagery is required for performance claims.
15. **Engineering Benchmarks Separated from Prescribed SIH Benchmarks:** Development datasets (LEVIR-CD cuts) are strictly distinguished from official benchmarks (VRSBench, RSVQA, CDVQA).
16. **Query-First User Experience:** Normal user operation is Query + Imagery $\rightarrow$ Auto-Orchestration. Model selection is purely an optional debug override.
17. **Explicit Uncertainty:** The system must quantify uncertainty (e.g. semantic uncertainty, sensor discordance) rather than fabricating false certainty.
18. **Evidence Provenance Preserved:** Coordinates, bounding boxes, and zonal statistics must retain traceability back to source pixels and geographic coordinates.
19. **M6 Deterministic Baseline is Interpretable:** The M6 rule-grounded optical-SAR baseline serves as an interpretable physics benchmark for future learned fusion.
20. **Future Learned Models Must Fit Existing Interfaces:** Subsequent fine-tuned or deep neural checkpoints must slot behind existing `BaseSpecialist` contracts.

---

## 6. Change Detection vs Change Understanding Boundary

- **`CHANGE_DETECT` exclusively owns:**
  - Spatial change localization (`WHERE`)
  - Binary change mask generation (`WHETHER`)
  - Changed pixel counts, percentages, and hectare area (`HOW MUCH`)
  - Connected component spatial clustering
- **`CHANGE_VQA` exclusively owns:**
  - Semantic land-cover transition identification (`WHAT KIND`)
  - Direction of change (`increased`, `decreased`, `modified`, `no_change`)
  - Multi-panel visual interpretation conditioned on TinyCD regions
- **Boundary Lock:** `CHANGE_VQA` must **never** fabricate change where `CHANGE_DETECT` found zero physical change. `CHANGE_DETECT` must **never** generate natural-language transition narratives.

---

## 7. Optical-SAR Joint Analysis Boundary

- M6 provides the cross-modal joint analysis specialist (`OPTICAL_SAR_FUSION`).
- The analytical result **must genuinely depend on both sensors**:
  - Optical visible/NIR bands contribute surface reflectance, vegetation greenness, and contextual texture.
  - SAR microwave backscatter contributes structural roughness, dielectric moisture response, and all-weather physical contrast.
- Outputs retain:
  - Optical evidence layer
  - SAR evidence layer
  - Joint fused classification
  - Sensor complementarity telemetry report

---

## 8. Confidence Boundary

- **Specialist Model Confidence:** Local metric emitted by individual specialist (e.g., softmax score, IoU overlap).
- **M8 Evidence Fusion:** Evaluates cross-model agreement and contradiction penalties. M8 does **not** compute final calibrated system confidence.
- **M9 Defensible Confidence Engine:** Exclusively owns final system confidence calculation, combining input data quality, spatial co-registration precision, model confidence, and consistency penalties.

---

## 9. Dataset / Sensor / Domain Abstraction

SatQuery AI's architecture explicitly models remote-sensing profiles:
- **Sensors:** Sentinel-2 (Multispectral Optical), Sentinel-1 (C-band SAR), Landsat-8/9, PlanetScope, RISAT, Cartosat.
- **Sensor Parameters:** Band counts, wavelength profiles, pixel resolution (e.g. 10m S2, 10m S1 GRD), radiometric ranges (uint16 reflectance vs float32 dB backscatter), and EPSG geotransforms.
- **Rule on Sensor Claims:** Never claim support for a sensor platform unless end-to-end reading, preprocessing, and inference tests have been executed on real imagery.

---

## 10. Synthetic Data Policy

- **Allowed Use:** Rapid zero-download unit testing, geometric projection assertions, edge-case validation (e.g. 0% change, 100% overlap, corrupt headers), and CI regression testing.
- **Strictly Prohibited:** Claiming real-world accuracy, F1 scores, or SIH milestone compliance based on synthetic fixtures.

---

## 11. Target Flagship Workflow

The eventual capstone demonstration of SatQuery AI:
```
Inputs: Time 1 Optical + Time 2 Optical + SAR + Natural-Language Query
Query: "Has the built-up area increased, where did it increase, and use SAR evidence to validate the result?"
Pipeline:
  1. Input Validation & Modality Inspection
  2. M7 Orchestration (Multi-Sensor Planning)
  3. CHANGE_DETECT (TinyCD: localize spatial change clusters)
  4. CHANGE_VQA (Interpret urban expansion transition)
  5. OPTICAL_SAR_FUSION (Verify building structure using SAR backscatter roughness)
  6. M8 Evidence Consistency (Confirm change detector and SAR backscatter agree)
  7. M9 Defensible Confidence (Compute calibrated probability)
  8. Final Result + Visual Evidence + Auditable Trace + Downloadable Report
```

---

## 12. Reproducibility & Environment Profile

- **Python:** 3.11.7
- **GPU Acceleration:** NVIDIA CUDA 12.4, PyTorch 2.5+, mixed-precision enabled.
- **Primary Hardware Constraint:** RTX 4070 (12 GB VRAM).
- **Geospatial Stack:** Rasterio 1.4+, GDAL 3.9+, Shapely, PyProj.
- **Execution Entrypoint:** `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
- **Frontend Entrypoint:** Available at `http://localhost:8000/ui`
- **Test Command:** `.venv\Scripts\pytest.exe tests -v`
