# SatQuery AI — Architecture Decision Records (ADRs)

**Standard:** Lightweight Architecture Decision Records  
**Status:** Canonical & Locked  
**Governing Authority:** SIH26167 / Indian Space Research Organisation (ISRO)  

---

### AD-001: Modular Specialist Architecture over Monolithic Multimodal LLM
- **Context:** Early vision-language assistants often pass images directly into general-purpose VLMs with prompt text. For remote sensing, generic VLMs suffer from severe spatial hallucination, lack coordinate projection capability, ignore SAR microwave physics, and cannot perform rigorous change detection.
- **Decision:** SatQuery AI adopts an agentic modular specialist architecture. Analytical tasks are distributed across discrete, specialized models/tools (`RS_VQA`, `RS_GROUND`, `CHANGE_DETECT`, `CHANGE_VQA`, `OPTICAL_SAR_FUSION`).
- **Consequences:** Eliminates black-box hallucination, allows independent model fine-tuning and benchmarking, and respects hardware boundaries (12 GB RTX 4070).

---

### AD-002: ModelRegistry as the Canonical Specialist Discovery Surface
- **Context:** Hardcoding model instances or import calls across routing and API layers leads to tight coupling, circular imports, and brittle test scaffolding.
- **Decision:** All models must subclass `BaseSpecialist`, declare explicit `ModelCapability` manifests, and register in the central singleton `ModelRegistry`. The orchestrator discovers and invokes tools exclusively via `registry.get(ident)` or `registry.find_specialists()`.
- **Consequences:** New specialists can be added or mocked in unit tests without changing routing or executor code.

---

### AD-003: StandardResultContract as the Invariant Exchange Schema
- **Context:** Diverse analytical pipelines (binary masks, bounding boxes, text narratives, cross-modal tables) require a predictable representation for API clients, UI dashboards, and report generators.
- **Decision:** Every specialist and orchestration workflow must return a validated `StandardResultContract` (Rule 8) containing `task`, `status`, `answer`, `confidence`, `evidence` bundle (images, masks, boxes, statistics, regions), `models` record, `parameters`, `warnings`, `execution_trace`, and `execution_time_ms`.
- **Consequences:** The frontend and report layers never depend on model-specific internal schemas.

---

### AD-004: Strict Separation of Change Detection from Semantic Interpretation
- **Context:** Bi-temporal change analysis involves two distinct analytical steps: determining spatial location and physical area (`WHERE / HOW MUCH`), and classifying the land-cover transition (`WHAT CHANGED`). Asking a VLM to detect change directly produces fabricated boundaries and unreliable pixel statistics.
- **Decision:** `CHANGE_DETECT` (TinyCD / CVA) exclusively generates binary masks, connected component clusters, and physical area metrics. `CHANGE_VQA` (VLM) is conditioned on those verified spatial clusters to interpret semantic transitions (`forest` $\rightarrow$ `bare_ground`).
- **Consequences:** Changes cannot be hallucinated without physical spectral/feature difference.

---

### AD-005: Genuine Optical-SAR Cross-Modal Fusion
- **Context:** Optical reflectance (visible/NIR bands) and SAR backscatter (microwave roughness/dielectric properties) provide complementary Earth observation signals. Naive tensor concatenation or ignoring backscatter physics fails to provide true cross-modal utility.
- **Decision:** Cross-modal joint analysis requires co-registered optical and SAR imagery, evaluates optical indices alongside calibrated SAR amplitude and backscatter ratios, and verifies sensitivity to both sensors.
- **Consequences:** The system visibly demonstrates why both modalities are required (e.g. cloud penetration, structure identification).

---

### AD-006: Interpretable Physics-Grounded Baseline for Optical-SAR (M6)
- **Context:** Complex deep neural multimodal fusion models require heavy training and can behave as black boxes. An interpretable, deterministic baseline was required for Milestone M6.
- **Decision:** M6 implemented a deterministic 6-class physics-grounded fusion specialist evaluating ExG, NIR reflectance, SAR backscatter magnitude, and cross-channel contrast.
- **Consequences:** Establishes a verifiable, deterministic baseline against which future learned deep fusion architectures can be measured.

---

### AD-007: Specialist Model Confidence Decoupled from Final System Confidence
- **Context:** Individual neural networks produce uncalibrated softmax or sigmoid scores that do not reflect geospatial alignment quality, sensor noise, or cross-model contradictions.
- **Decision:** Specialist output confidence is treated as raw evidence telemetry. Final system confidence is exclusively synthesized by the confidence engine (M9), combining input quality, spatial registration score, model score, and cross-evidence consistency penalties.
- **Consequences:** Model uncertainty is transparently exposed rather than disguised as statistical certainty.

---

### AD-008: Synthetic Data Restricted to Pipeline Correctness (Not Generalization Proof)
- **Context:** Synthetic test fixtures allow rapid, zero-download CI testing for sub-pixel crops, corner cases, and zero-change conditions, but cannot prove real-world Earth observation performance.
- **Decision:** Synthetic fixtures are strictly restricted to unit/integration testing and regression validation. All model performance and benchmark claims must use verified open remote-sensing datasets (LEVIR-CD, VRSBench, RSVQA, BigEarthNet).
- **Consequences:** Prevents deceptive claims of benchmark accuracy based on synthetic data.

---

### AD-009: Separation of Engineering Benchmarks from Official SIH Benchmarks
- **Context:** During development, engineering subsets (e.g., LEVIR-CD small cuts, local validation rasters) are necessary for rapid iteration, while SIH26167 prescribes formal benchmarks (VRSBench, RSVQA, CDVQA).
- **Decision:** All evaluation documents must explicitly distinguish Engineering Development Benchmarks from Prescribed SIH Evaluation Benchmarks. Engineering metrics must never be reported as official SIH compliance results.
- **Consequences:** Maintains rigorous academic and technical honesty across all milestones.

---

### AD-010: Deterministic Query & Input Configuration Orchestration (M7)
- **Context:** Using an unconstrained general-purpose LLM to decide routing introduces non-deterministic latency, potential infinite loops, and hallucinated model selection.
- **Decision:** M7 orchestrator (`AgentRouter`, `AgentPlanner`, `AgentExecutor`) uses deterministic intent pattern banks combined with physical input configuration analysis (modality, dimensions, timestamps, spatial overlap).
- **Consequences:** Zero-latency routing, 100% reproducible execution plans, and deterministic rejection of impossible requests.

---

### AD-011: Strict Preservation of Evidence Provenance
- **Context:** When multiple specialists execute in sequence, intermediate bounding boxes, raw pixel coordinates, and sensor metadata can be lost or overwritten.
- **Decision:** Every evidence item retains its provenance: bounding boxes preserve normalized and pixel coordinates; raster masks link to generated preview URLs; statistics record metric units and computation source.
- **Consequences:** Every final claim can be audited back to the exact intermediate step and source pixels.

---

### AD-012: Explicit Uncertainty Exposure and Zero Certainty Fabrication
- **Context:** Real-world satellite imagery often contains cloud cover, poor co-registration, or ambiguous resolution. AI chatbots tend to generate confident answers regardless of data quality.
- **Decision:** When image quality is degraded, spatial overlap is insufficient, or model confidence is low, SatQuery AI must emit explicit warnings and downgrade confidence (Rule 31). A qualified or rejected answer is preferred over a confidently incorrect answer.
- **Consequences:** Establishes operational credibility for mission-critical defense and Earth observation analysts.

---

### AD-013: Grounded Remote-Sensing Domain Adaptation Requirement
- **Context:** SIH26167 problem statement mandates remote-sensing specialization. Relying solely on zero-shot weights from natural-image datasets (COCO/ImageNet) is insufficient.
- **Decision:** At least one visual foundation model must be adapted using remote-sensing datasets (e.g. BigEarthNet / VRSBench / LEVIR-CD) with documented hyperparameters, checkpoints, and evaluation results (M10).
- **Consequences:** Fulfills mandatory hackathon criteria with defensible domain fine-tuning.

---

### AD-014: Geographically Diverse Scene Generalization Evaluation
- **Context:** Change detection models trained exclusively on residential rooftops (e.g. standard LEVIR-CD) fail when evaluated on agricultural, desert, forested, or coastal scenes.
- **Decision:** M4 introduced a dedicated diverse remote-sensing benchmark suite evaluating models across urban, agricultural, forest, coastal, and desert scenes with distinct failure tracking.
- **Consequences:** Generalization limits are mapped and documented before final deployment.

---

### AD-015: Query-First User Experience
- **Context:** Forcing analysts to know whether to run TinyCD, Florence-2, Grounding DINO, or CVA creates unnecessary cognitive friction.
- **Decision:** The user flow is Query-First: upload imagery, type an analytical question, and click Analyze. The agent orchestrates specialist selection automatically. Manual model override is kept solely as an optional debug capability.
- **Consequences:** Simplifies judging demos while keeping full developer auditability intact.

---

### AD-016: Rigorous Separation of Total Spatial Change, Semantic Interpretation, and Semantic-Specific Area
- **Context:** When users ask semantic quantity queries on bi-temporal imagery (e.g. *"How much buildings were newly created?"* or *"How much vegetation was lost?"*), naive systems conflate total detected surface change (e.g. $149.36\text{ ha}$ across 23 clusters) with the requested semantic class, falsely claiming that all $149.36\text{ ha}$ were newly created buildings.
- **Decision:** SatQuery AI enforces strict scientific separation:
  1. **Total Spatial Change:** Measured directly from the verified binary change mask generated by `CHANGE_DETECT` (`changed_pixels`, `change_ratio_pct`, `physical_area_ha`).
  2. **Semantic Change Interpretation:** Inferred by `CHANGE_VQA` conditioned on the 3-panel composite and spatial clusters (`summary`, `temporal_direction`, `predominant_transition`).
  3. **Semantic-Specific Area:** Requires pixel-level multi-class semantic segmentation. When only binary change detection is active, class-specific area is NOT directly measurable. The backend records `semantic_changed_area_ha = None`, `semantic_area_status = "unmeasured_from_spatial_evidence"`, and explicitly qualifies this limitation in both the analytical answer and the M11 analyst report.
- **Consequences:** Eliminates scientific fabrication, upholds SIH26167 / ISRO technical credibility, and delivers the exact physical change and semantic qualitative transitions without false quantitative certainty.

