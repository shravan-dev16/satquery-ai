# SatQuery AI — Milestone M7 Evaluation Report
## Agentic Orchestration & Dynamic Routing Engine

**Date:** September 8, 2026  
**Milestone:** M7 (Agentic Orchestration & Dynamic Routing)  
**Hackathon:** Smart India Hackathon — SIH26167  
**Organization:** Indian Space Research Organisation (ISRO)  
**Domain:** Space Technology  
**Baseline Test Count (Pre-M7):** 121 passed  
**Post-M7 Test Suite:** 137 passed (100% green, 0 failed, 0 regressions)  

---

### 1. Executive Summary

Milestone M7 introduces the **Agentic Orchestration & Dynamic Routing Layer** for SatQuery AI. In accordance with the SIH26167 problem statement:
> *"The system must automatically select, sequence, and execute the appropriate specialist models or tools according to the query and input configuration."*

SatQuery AI has eliminated mandatory manual model selection for standard operations. The system now accepts a natural-language query alongside single or dual remote-sensing observations (GeoTIFF / TIFF / PNG), deterministically analyzes user intent and physical input configurations, synthesizes an observable execution plan, dynamically resolves specialists from the central `ModelRegistry`, executes multi-stage specialist pipelines, and returns a unified `StandardResultContract` with full 10-step auditable execution traces.

> **System Scope Declaration:**  
> M7 is a **deterministic, rule-grounded agentic query and input configuration orchestration layer**. It does **NOT** claim unconstrained autonomous general intelligence or hallucinated agent loops. Every routing decision, plan step, and execution dispatch is auditable, repeatable, and grounded in remote-sensing physics and geospatial metadata.

---

### 2. Orchestration Architecture

The orchestration subsystem is housed under `backend/agent/` and strictly separates **Routing**, **Plan Creation**, and **Specialist Execution**:

```
USER QUERY + REMOTE SENSING OBSERVATIONS
                     │
                     ▼
          ┌─────────────────────┐
          │     AgentRouter     │
          └──────────┬──────────┘
                     │  - Inspects input configuration (single, dual, temporal, cross-modal)
                     │  - Inspects query semantic intent
                     │  - Detects modality (Optical vs SAR) via verified metadata
                     │  - Rejects impossible / incompatible combinations
                     ▼
             RoutingDecision
                     │
                     ▼
          ┌─────────────────────┐
          │    AgentPlanner     │
          └──────────┬──────────┘
                     │  - Builds ordered, observable PlanStep sequence
                     │  - Manages dependencies (e.g. TinyCD -> ChangeVQASpecialist)
                     │  - Flags single vs multi-stage pipelines
                     ▼
              ExecutionPlan
                     │
                     ▼
          ┌─────────────────────┐
          │    AgentExecutor    │ ◄─── ModelRegistry
          └──────────┬──────────┘      (Dynamic Specialist Discovery)
                     │  - Executes preprocessing & validation
                     │  - Sequentially invokes registered specialists
                     │  - Passes intermediate state & spatial masks forward
                     │  - Assembles evidence bundle & statistics
                     │  - Emits 10-step auditable execution trace
                     ▼
          StandardResultContract
```

---

### 3. Supported Task Classes and Routing Logic

| Task Class | Input Configuration | Primary Intent Triggers | Selected Specialists | Execution Type |
|---|---|---|---|---|
| `VQA` (`single_image_vqa`) | 1 Optical/SAR image | General land cover, terrain, scene inquiry (`what is`, `describe`, `is there`) | `RS_VQA` (Florence-2 / BLIP-2) | Single-stage (5 steps) |
| `GROUNDING` (`single_image_grounding`) | 1 Optical image | Spatial localization (`where is`, `locate`, `highlight`, `find the`, `bounding box`, `segment`) | `RS_GROUND` (Grounding DINO) | Single-stage (5 steps) |
| `CAPTION` (`single_image_caption`) | 1 Optical image | Scene description (`generate caption`, `describe the scene`, `summarize`) | `RS_CAPTION` | Single-stage (5 steps) |
| `CHANGE_DETECTION` (`bitemporal_change_detection`) | 2 Optical images (co-registered, temporal pair) | Spatial change localization without semantic categorization (`where did changes occur`, `detect change`, `difference map`) | `CHANGE_DETECT` (TinyCD / CVA Baseline) | Single-stage (9 steps) |
| `CHANGE_VQA` (`bitemporal_change_vqa`) | 2 Optical images (co-registered, temporal pair) | Semantic change interpretation (`what type of change`, `has built-up increased`, `describe change`, `land-cover transition`) | `CHANGE_DETECT` $\rightarrow$ `CHANGE_VQA` | **Multi-stage** (10 steps) |
| `OPTICAL_SAR_ANALYSIS` (`optical_sar_analysis`) | 1 Optical + 1 SAR image | Cross-sensor joint analysis (`built-up and water`, `optical and SAR`, `radar penetration`) | `OPTICAL_SAR_FUSION` (OpticalSARSpecialist) | Single-stage (10 steps) |

---

### 4. Input Configuration Constraints & Anti-Reinterpretation Rules

The router actively enforces scientific consistency rules to prevent invalid or misleading AI inferences:

1. **Single Image + Temporal Query Rejection:**  
   If a single observation is uploaded with a query such as `"What changed between these two dates?"`, the router immediately flags an invalid combination:
   `"Temporal change analysis requires both Time 1 (earlier) and Time 2 (later) observations, but only a single image was uploaded."`  
   The system never fabricates temporal change from a single timestamp.

2. **Optical + SAR Pair + Temporal Query Rejection (Sensor Physics Mismatch):**  
   If an Optical + SAR multimodal pair is uploaded with a temporal query (`"What changed between 2020 and 2024?"`), the router rejects it:
   `"Incompatible input configuration: Temporal change detection requires same-sensor observations to prevent false change alarms caused by sensor physics mismatch."`

3. **Dual Optical Pair + Cross-Modal SAR Query Rejection:**  
   If two optical images are uploaded with a query requesting SAR backscatter analysis (`"Use SAR to detect structures"`), the system rejects it because no SAR microwave observations are available.

4. **Dual SAR Pair + Cross-Modal Optical Query Rejection:**  
   If two SAR images are uploaded with a query requesting RGB reflectance analysis, the system rejects it with an explicit explanation.

---

### 5. Multi-Stage Change Workflow (`CHANGE_DETECT` $\rightarrow$ `CHANGE_VQA`)

The multi-stage change analysis workflow enforces clean separation of concerns:
- **TinyCD (WHERE / HOW MUCH):** Detects physical surface change, outputs binary change mask, computes connected component bounding boxes, and calculates changed pixel area and hectare statistics.
- **ChangeVQASpecialist (WHAT KIND OF CHANGE):** Receives the verified change mask, cropped high-resolution regions of interest, and physical area statistics. Assembles a 3-panel composite evidence image (Time 1, Time 2, Change Overlay) and performs conditioned vision-language reasoning to identify land-cover transitions (e.g. `forest_or_trees` $\rightarrow$ `bare_ground_or_soil`).
- **AgentExecutor Sequencing:** Orchestrates the handover between TinyCD and ChangeVQASpecialist without user intervention, ensuring that the semantic model only reasons about verified physical change clusters.

---

### 6. Auditable Execution Trace

Every routed workflow returns an observable execution trace conforming to Rule 12:

```json
{
  "step_number": 1,
  "step_name": "InputValidation",
  "status": "completed",
  "details": "Validated two rasters: T1='time1.tif', T2='time2.tif'.",
  "duration_ms": 12
},
{
  "step_number": 2,
  "step_name": "TemporalValidation",
  "status": "completed",
  "details": "Temporal ordering verification: verified_chronological.",
  "duration_ms": 1
},
{
  "step_number": 3,
  "step_name": "SpatialCompatibility",
  "status": "completed",
  "details": "Geospatial overlap: 100.0%. CRS match: True.",
  "duration_ms": 1
},
{
  "step_number": 4,
  "step_name": "Alignment",
  "status": "completed",
  "details": "Aligned imagery to common spatial grid: 256x256 px.",
  "duration_ms": 8
},
{
  "step_number": 5,
  "step_name": "SpecialistSelection",
  "status": "completed",
  "details": "Selected 'CHANGE_DETECT' and 'CHANGE_VQA' from ModelRegistry.",
  "duration_ms": 1
},
{
  "step_number": 6,
  "step_name": "ModelExecution",
  "status": "completed",
  "details": "Executed change detection model in 38 ms.",
  "duration_ms": 38
},
{
  "step_number": 7,
  "step_name": "ChangeMaskValidation",
  "status": "completed",
  "details": "Mask validation: 13.89% scene change, 3 spatial clusters.",
  "duration_ms": 2
},
{
  "step_number": 8,
  "step_name": "Statistics",
  "status": "completed",
  "details": "Calculated physical zonal statistics: 9,102 changed pixels.",
  "duration_ms": 1
},
{
  "step_number": 9,
  "step_name": "SemanticInterpretation",
  "status": "completed",
  "details": "ChangeVQASpecialist interpreted: vegetation -> bare_ground (decreased).",
  "duration_ms": 420
},
{
  "step_number": 10,
  "step_name": "EvidenceAssembly",
  "status": "completed",
  "details": "Assembled StandardResultContract. Evidence confidence: 0.88.",
  "duration_ms": 2
}
```

---

### 7. Full Routing Test Matrix Verification

The dedicated test suite `tests/test_agent_orchestration.py` covers all 16 matrix cases:

| # | Test Case | Function | Status |
|---|---|---|---|
| 1 | Single optical image $\rightarrow$ VQA | `test_single_image_vqa_routing` | **PASSED** |
| 2 | Single optical image $\rightarrow$ Grounding | `test_single_image_grounding_routing` | **PASSED** |
| 3 | Bi-temporal pair $\rightarrow$ Change detection | `test_bitemporal_pair_change_detection_routing` | **PASSED** |
| 4 | Bi-temporal pair $\rightarrow$ Semantic change interpretation | `test_bitemporal_semantic_interpretation_routing` | **PASSED** |
| 5 | Optical + SAR pair $\rightarrow$ Optical-SAR fusion | `test_optical_sar_joint_routing` | **PASSED** |
| 6 | Two optical images $\rightarrow$ Cross-modal rejected | `test_two_optical_not_optical_sar` | **PASSED** |
| 7 | Two SAR images $\rightarrow$ Cross-modal rejected | `test_two_sar_not_optical_sar` | **PASSED** |
| 8 | Single image + temporal query $\rightarrow$ Validation failure | `test_single_image_temporal_question_validation_failure` | **PASSED** |
| 9 | Optical + SAR + temporal query $\rightarrow$ Physics mismatch rejected | `test_optical_sar_temporal_query_rejected` | **PASSED** |
| 10 | Query wording variations $\rightarrow$ Stable task classification | `test_query_wording_variations_same_task` | **PASSED** |
| 11 | Dynamic specialist dispatch from `ModelRegistry` | `test_specialist_lookup_from_model_registry` | **PASSED** |
| 12 | Multi-stage change plan sequencing order | `test_multi_stage_change_plan_ordered_correctly` | **PASSED** |
| 13 | Trace ordering and monotonically increasing steps | `test_trace_contains_actual_execution_order` | **PASSED** |
| 14 | API automatic routing without manual `task_hint` | `test_api_automatic_routing` | **PASSED** |
| 15 | Manual debug override backward compatibility (`cva_raw`, `tinycd_raw`) | `test_manual_candidate_override_does_not_break` | **PASSED** |
| 16 | Impossible task/input configurations fail cleanly (HTTP 400) | `test_ambiguous_or_impossible_combinations_fail_cleanly` | **PASSED** |

---

### 8. Known Limitations & Strict Boundaries

1. **Deterministic Lexical/Regex Routing:**  
   Routing uses regex intent pattern banks and physical input configuration inspection. It does not employ an uncalibrated general-purpose LLM router, ensuring predictable, zero-latency, and reproducible routing.
2. **Milestone Boundaries Preserved:**  
   - **M8 (Evidence Fusion):** M7 collects specialist evidence into `EvidenceBundle`, but multi-model cross-evidence fusion engine remains strictly scheduled for M8.
   - **M9 (Confidence Engine):** M7 passes model and heuristic confidence scores; full calibrated confidence modeling remains scheduled for M9.
   - **M10 (Fine-Tuning):** Model weights are loaded and evaluated as configured in M1–M6; remote-sensing LoRA fine-tuning remains scheduled for M10.
