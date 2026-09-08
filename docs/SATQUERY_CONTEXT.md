# SatQuery AI — Master Context File

> **Primary Audience:** Future AI coding assistants and developers.  
> **Mandate:** Read this document FIRST before writing code or proposing architectural modifications. Do not deviate from the locked architecture.

---

## 1. Project Identity & Purpose

- **Project Name:** SatQuery AI
- **Official Title:** An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries
- **Hackathon:** Smart India Hackathon 2026 — Problem Statement **SIH26167**
- **Domain:** Space Technology
- **Organization:** Indian Space Research Organisation (ISRO)
- **Primary Objective:** Build an agentic remote-sensing analysis system that accepts natural-language queries over single, bi-temporal, and cross-modal Earth observation imagery, automatically routing requests to specialist tools, providing auditable execution traces and spatial evidence.

> [!CRITICAL]
> **SatQuery AI is NOT a generic chatbot with an image attached.**  
> It is an evidence-first remote-sensing intelligence platform. Never replace specialist modular pipelines with an unconstrained single multimodal LLM/VLM prompt.

---

## 2. Milestone State

| Milestone | Scope | Status | Verification Summary |
|---|---|---|---|
| **M0** | Scaffolding, BaseSpecialist, ModelRegistry, StandardResultContract | **COMPLETE** ✅ | 20 unit tests, contract round-trips |
| **M1** | GeoTIFF validation, metadata extraction, baseline VQA | **COMPLETE** ✅ | CRS/transform inspection, Florence-2/BLIP-2 |
| **M2** | Text-guided region grounding & geospatial bounding boxes | **COMPLETE** ✅ | Grounding DINO integration |
| **M2.1** | Grounding recovery & empirical subset evaluation | **COMPLETE** ✅ | VRSBench subset validation, IoU benchmarking |
| **M3** | Bi-temporal change foundation (alignment, CVA, TinyCD) | **COMPLETE** ✅ | 11-point validation, sub-pixel intersection crop |
| **M4** | Dedicated change detection benchmark & diverse RS evaluation | **COMPLETE** ✅ | LEVIR-CD benchmark, failure diagnostics |
| **M5** | Semantic change interpretation & Change VQA | **COMPLETE** ✅ | 3-panel composite evidence, zero-change gate |
| **M6** | Optical + SAR cross-modal joint analysis | **COMPLETE** ✅ | 6-class physics fusion, dual-modality sensitivity |
| **M7** | Agentic orchestration & dynamic routing | **COMPLETE** ✅ | Router, Planner, Executor DAGs, 16-case matrix |
| **M8** | Evidence Fusion & Consistency Engine | **COMPLETE** ✅ | Rules C1–C8, immutable evidence units, gating |
| **M9** | Defensible System Confidence Engine | **COMPLETE** ✅ | Multi-factor heuristic, contradiction dominance caps |
| **M10** | Remote-Sensing Adaptation (VRSBench / LoRA) | **PLANNED / AUDITED** 📋 | Target: Unified VLM LoRA for `RS_VQA` & `CHANGE_VQA` (`Qwen2-VL-2B-Instruct`) |
| **M11** | Report Generation (PDF & JSON) | *Scheduled* | Exportable audit-ready analyst reports |
| **M12** | Interactive UI Polish | *Scheduled* | Unified web analyst workspace |
| **M13** | End-to-End Hardening & Demo Rehearsal | *Scheduled* | Sub-3s response reliability for judging |

---

## 3. Specialist Inventory

| Identifier | Capability Class | Module | Status | Role |
|---|---|---|---|---|
| `RS_VQA` | `RemoteSensingVQASpecialist` | `backend/models/vqa.py` | **Production Specialist** | Visual question answering on single optical/SAR imagery |
| `RS_GROUND` | `RemoteSensingGroundingSpecialist` | `backend/models/grounding.py` | **Production Specialist** | Text-guided spatial object & feature bounding box localization |
| `CHANGE_DETECT` | `ChangeDetectionSpecialist` | `backend/models/change.py` | **Production Specialist** | Binary physical surface change detection (TinyCD / CVA) |
| `CHANGE_VQA` | `ChangeVQASpecialist` | `backend/models/change_vqa.py` | **Production Specialist** | Multi-stage conditioned semantic change reasoning |
| `OPTICAL_SAR_FUSION` | `OpticalSARSpecialist` | `backend/models/optical_sar.py` | **Production Specialist** | Cross-modal feature extraction and land-cover classification |
| `RS_CAPTION` | `RemoteSensingCaptionSpecialist` | `backend/models/caption.py` | *Interface Stub (M2-B)* | Architectural interface placeholder; full implementation scheduled |

---

## 4. Non-Negotiable Architectural Rules

1. **Evidence-First Answers:** Every spatial claim must be accompanied by visual masks, bounding boxes, or zonal statistics. Text narrative without evidence is forbidden.
2. **Specialist Separation:** Change detection (`WHERE / HOW MUCH`) is strictly separated from semantic change interpretation (`WHAT CHANGED`).
3. **Modalities Are Physics-Based:** Optical reflectance (surface chemistry/color) and SAR backscatter (dielectric constant/roughness/penetration) must remain distinguishable.
4. **No Certainty Fabrication (Scientific Honesty):** If imagery is misaligned, uncalibrated, or low-confidence, the system must return explicit warnings or structured failures. Never hallucinate metadata or certainty.
5. **Specialist Confidence $\neq$ System Confidence:** Model output confidence is raw telemetry; final system confidence is exclusively computed in M9.
6. **Auditable Execution Traces (Rule 12):** All API responses must return chronological, observable trace steps disclosing operations, timings, and statuses without hidden chain-of-thought.
7. **Query-First UX:** The normal user flow is Query + Imagery $\rightarrow$ Auto-Orchestration. Manual model selection (`task_hint`) is reserved for debug/developer overrides.

---

## 5. Development Guidelines for Future Milestones

- **Hardware Budget:** NVIDIA RTX 4070 (12 GB VRAM). Never load multiple large foundation models simultaneously without memory checks.
- **Testing Standard:** Every change must preserve 100% green test passes. Use `.venv\Scripts\pytest.exe tests -v`.
- **Modularity:** Always interact with specialists through `backend.agent.registry.registry` and return `backend.agent.schema.StandardResultContract`.
