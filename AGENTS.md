# SatQuery AI — Agent Instructions

## 1. PROJECT IDENTITY

Project:
**SatQuery AI — An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

Hackathon:
**Smart India Hackathon — SIH26167**
Organization:
**Indian Space Research Organisation (ISRO)**
Domain:
**Space Technology**

SatQuery AI is NOT a generic image chatbot.

It is an **agentic remote-sensing analysis system** that accepts natural-language queries over single, paired, multimodal, and bi-temporal Earth-observation imagery and automatically selects suitable specialist models/tools to produce evidence-grounded results.

The official SIH26167 problem statement is the source of truth for mandatory functionality.

---

# 2. PRIMARY OBJECTIVE

Build a technically credible, reproducible, modular, evaluation-driven system that satisfies every mandatory requirement of SIH26167.

The system must support:

1. Single optical/multispectral image analysis.
2. Single SAR image analysis.
3. Single-image VQA.
4. At least one additional single-image capability:
   - captioning / scene description
   - OR text-guided region grounding.
5. Bi-temporal image-pair change analysis.
6. Change description or change-based VQA.
7. Optical + SAR cross-modal analysis.
8. Remote-sensing adaptation/fine-tuning using BigEarthNet or appropriate open remote-sensing data.
9. Agentic task interpretation and model/tool selection.
10. Input compatibility validation.
11. Evidence-grounded output.
12. Confidence information.
13. Auditable execution trace.
14. Downloadable reports.

The project must demonstrate these capabilities through a reliable interactive GUI/web application.

---

# 3. CORE PRODUCT PRINCIPLE

SatQuery must behave like a **remote-sensing AI analyst**, not like a chatbot with an image attached.

Conceptual architecture:

USER
  ↓
INPUT VALIDATOR
  ↓
QUERY / TASK INTERPRETER
  ↓
AGENTIC CONTROLLER
  ↓
MODEL / TOOL REGISTRY
  ├── Remote-sensing VQA
  ├── Captioning
  ├── Grounding
  ├── Change Detection
  ├── Change VQA / Captioning
  ├── Optical-SAR Analysis
  └── Geospatial Preprocessing
  ↓
EVIDENCE FUSION
  ↓
CONSISTENCY CHECKER
  ↓
CONFIDENCE ESTIMATION
  ↓
ANSWER GENERATOR
  ↓
VISUAL EVIDENCE + EXECUTION TRACE + REPORT

Do not replace this architecture with a generic LLM/VLM pipeline without strong technical justification.

---

# 4. MANDATORY REMOTE-SENSING SPECIALIZATION

A generic multimodal LLM/VLM alone is NOT sufficient.

At least one visual or vision-language component must be adapted using remote-sensing data.

Preferred directions include:

- BigEarthNet
- Prithvi
- TerraTorch
- TerraMind
- VRSBench
- RSVQA
- CDVQA
- LEVIR-CC
- ChangeChat or related open change-understanding resources

Do not claim that a generic VLM is remote-sensing adapted.

Record exactly:

- base model;
- dataset;
- subset size;
- training method;
- preprocessing;
- adapter/fine-tuning method;
- epochs;
- learning rate;
- hardware;
- evaluation results.

Never fabricate adaptation results or benchmark metrics.

---

# 5. SPECIALIST MODEL ARCHITECTURE

Use a central model/tool registry.

Possible specialist capabilities:

- `RS_VQA`
- `RS_CAPTION`
- `RS_GROUND`
- `CHANGE_DETECT`
- `CHANGE_VQA`
- `OPTICAL_SAR_ANALYSIS`
- `GEO_PREPROCESSOR`
- `EVIDENCE_FUSER`
- `CONSISTENCY_CHECKER`

Each specialist must have a clearly defined interface.

Every specialist should declare:

- identifier;
- task;
- supported modalities;
- supported input count;
- required preprocessing;
- output format;
- confidence availability;
- hardware requirements;
- fallback behavior.

Avoid hardcoding model calls throughout the application.

---

# 6. AGENTIC ORCHESTRATION

The controller must determine the workflow from:

- natural-language query;
- number of images;
- image modality;
- image metadata;
- acquisition dates;
- spatial compatibility;
- model/tool capabilities.

Examples:

Query:
"What changed between these two dates?"

Expected routing:

BITEMPORAL
→ alignment validation
→ change detection
→ change understanding
→ evidence fusion
→ final answer

Query:
"Where is the water body?"

Expected routing:

SINGLE IMAGE
→ grounding
→ spatial evidence
→ final answer

Query:
"Use the optical and SAR images together to identify built-up regions."

Expected routing:

OPTICAL + SAR
→ modality validation
→ co-registration validation
→ optical-SAR analysis
→ evidence fusion
→ final answer

Keyword-only routing is insufficient.

When practical, task selection should use a structured intent representation.

---

# 7. INPUT VALIDATION

All inference must pass through input validation.

Validate:

- file type;
- image readability;
- width/height;
- band count;
- CRS;
- geotransform;
- spatial resolution;
- nodata;
- acquisition date;
- sensor/modality;
- spatial coverage;
- pair compatibility;
- temporal ordering;
- co-registration where required.

Supported geospatial inputs:

- GeoTIFF
- TIFF

PNG/JPEG may be accepted for benchmark/demo datasets where appropriate.

Never silently invent metadata.

Never silently assume two images are compatible if compatibility cannot be established.

Return clear errors and warnings.

---

# 8. STANDARD RESULT CONTRACT

Specialists must return a common structured result.

Use a schema conceptually equivalent to:

{
  "task": "...",
  "answer": "...",
  "confidence": 0.0,
  "evidence": {
    "images": [],
    "masks": [],
    "boxes": [],
    "statistics": [],
    "regions": []
  },
  "models": [],
  "parameters": {},
  "warnings": [],
  "execution_time_ms": 0
}

The frontend must not depend on the internal implementation of individual models.

---

# 9. EVIDENCE-FIRST ANSWERS

Text alone is insufficient whenever spatial evidence can be generated.

Prefer returning:

- natural-language answer;
- confidence;
- source image;
- bounding boxes;
- masks;
- change overlays;
- detected regions;
- numerical statistics;
- relevant metadata.

Example:

BAD:

"Built-up area increased."

BETTER:

"Built-up area increased primarily in the northern section."

Evidence:

- change mask;
- affected region;
- estimated change statistic;
- model confidence;
- execution trace.

---

# 10. CONSISTENCY CHECKING

When multiple models produce independent evidence, compare their outputs.

Example:

Change detector:
"Built-up change detected: 18%."

Language model:
"No significant urban expansion."

The system must detect disagreement.

Do not hide contradictions.

Instead return something such as:

- consistency warning;
- reduced confidence;
- explanation of disagreement.

Confidence must be based on measurable signals whenever possible.

Do not generate arbitrary confidence percentages.

---

# 11. CONFIDENCE PRINCIPLE

Confidence should be derived from defensible signals such as:

- model confidence;
- evidence quality;
- input quality;
- registration quality;
- cross-model agreement;
- result completeness.

Document how confidence is calculated.

Distinguish:

- model confidence;
- estimated system confidence;
- calibrated confidence, if calibration is actually performed.

Never present an uncalibrated score as statistically validated probability.

---

# 12. AUDITABLE EXECUTION TRACE

The user must be able to see the observable execution summary.

Example:

Task:
Bi-temporal change analysis

Execution:

1. Input validation
2. Temporal pair detected
3. Spatial compatibility checked
4. Change detector selected
5. Change model executed
6. Change-VQA specialist executed
7. Evidence consistency checked
8. Final answer generated

Expose:

- selected task;
- selected models/tools;
- relevant parameters;
- outputs;
- warnings;
- confidence.

Do NOT expose hidden chain-of-thought or private model reasoning.

---

# 13. BI-TEMPORAL WORKFLOW

Bi-temporal analysis is a primary differentiator.

Required conceptual flow:

IMAGE A + IMAGE B
→ validation
→ temporal verification
→ alignment/co-registration validation
→ normalization
→ change detection
→ semantic change interpretation
→ change VQA / captioning
→ spatial evidence
→ consistency
→ confidence
→ answer

Whenever feasible, provide:

- before image;
- after image;
- change mask/overlay;
- changed regions;
- change description;
- relevant statistics.

Do not claim reliable change detection when the imagery is not sufficiently aligned.

---

# 14. OPTICAL + SAR WORKFLOW

Optical and SAR are complementary modalities.

The system must demonstrate why both are useful.

Required conceptual flow:

OPTICAL
+
SAR
→ compatibility validation
→ modality-specific preprocessing
→ specialist/fusion analysis
→ evidence fusion
→ answer

Do not simply concatenate tensors without technical justification.

Explain complementary information when appropriate.

Example:

Optical:
visible/spectral/contextual information.

SAR:
structural/backscatter information and cloud/night robustness.

---

# 15. DATASET POLICY

Use datasets purposefully.

Preferred mapping:

BigEarthNet:
remote-sensing adaptation and multispectral/SAR representation.

VRSBench:
single-image VQA, captioning, grounding.

RSVQA:
single-image remote-sensing VQA evaluation.

CDVQA:
change detection + VQA.

LEVIR-CC / related datasets:
bi-temporal change description/captioning.

ChangeChat / related resources:
interactive bitemporal change understanding.

Do not download every available dataset.

For every dataset document:

- source;
- license;
- purpose;
- task;
- modality;
- split;
- sample count;
- preprocessing;
- evaluation role.

Never mix train and evaluation data improperly.

Avoid benchmark leakage.

---

# 16. COMPUTE / HARDWARE CONSTRAINTS

Primary development machine:

NVIDIA RTX 4070 with approximately 12 GB VRAM.

Design local inference and experimentation around this constraint.

Prefer:

- small/medium models;
- quantization where appropriate;
- LoRA/PEFT;
- mixed precision;
- tiling;
- batch-size control;
- memory-efficient inference.

For heavier training, use cloud GPU resources when necessary.

Never assume that a large model fits into 12 GB VRAM without verifying it.

Before installing large ML dependencies or model checkpoints, estimate:

- VRAM;
- RAM;
- disk;
- inference latency.

Avoid unnecessary GPU dependencies.

---

# 17. SOFTWARE PRINCIPLES

Preferred stack unless a strong reason exists to change it:

Backend:
- Python
- FastAPI

ML:
- PyTorch
- Transformers where appropriate
- TerraTorch for geospatial foundation-model workflows where appropriate

Geospatial:
- Rasterio
- GDAL
- GeoPandas when required

Frontend:
- React/Next.js OR Streamlit depending on project complexity and implementation speed

Testing:
- pytest

Version control:
- Git

The final dependency stack must remain understandable and reproducible.

---

# 18. PROJECT STRUCTURE

Prefer a modular structure similar to:

satquery-ai/
│
├── AGENTS.md
├── README.md
├── backend/
│   ├── agent/
│   │   ├── planner.py
│   │   ├── router.py
│   │   ├── registry.py
│   │   ├── executor.py
│   │   └── trace.py
│   │
│   ├── preprocessing/
│   │   ├── geotiff.py
│   │   ├── metadata.py
│   │   ├── modality.py
│   │   ├── alignment.py
│   │   └── normalization.py
│   │
│   ├── models/
│   │   ├── vqa.py
│   │   ├── caption.py
│   │   ├── grounding.py
│   │   ├── change.py
│   │   └── optical_sar.py
│   │
│   ├── evidence/
│   │   ├── fusion.py
│   │   ├── confidence.py
│   │   └── consistency.py
│   │
│   ├── evaluation/
│   │
│   └── main.py
│
├── frontend/
│
├── configs/
├── datasets/
│   └── README.md
├── models/
│   └── README.md
├── scripts/
├── tests/
└── docs/

Adapt the structure when necessary, but preserve separation of concerns.

---

# 19. TWO-PERSON TEAM OWNERSHIP

PERSON A:
AI / remote sensing / backend / agent

Own:

- data ingestion;
- GeoTIFF processing;
- metadata;
- preprocessing;
- models;
- fine-tuning;
- model registry;
- agent;
- change analysis;
- optical-SAR analysis;
- evidence fusion;
- confidence;
- evaluation.

PERSON B:
Frontend / UX / visualization / application integration

Own:

- upload UI;
- query UI;
- metadata display;
- image visualization;
- temporal comparison;
- masks;
- bounding boxes;
- confidence display;
- execution trace;
- reports;
- API integration.

Shared ownership:

- API contracts;
- integration;
- tests;
- final demo;
- documentation;
- Git workflow.

Agents must clearly state ownership and dependencies when proposing work.

---

# 20. TASK REPORTING FORMAT

Whenever proposing an implementation task, use:

OWNER:
DEPENDENCIES:
OBJECTIVE:
INPUT:
OUTPUT:
FILES TO CHANGE:
TESTS:
DEFINITION OF DONE:

This prevents duplicated work between the two developers.

---

# 21. DEVELOPMENT MILESTONES

Implement incrementally in this approximate order:

M0:
Architecture + repository + shared contracts

M1:
Single image ingestion + VQA

M2:
Captioning or grounding

M3:
GeoTIFF validation + metadata

M4:
Bi-temporal ingestion + alignment

M5:
Change detection

M6:
Change VQA / description

M7:
Optical-SAR analysis

M8:
Agentic orchestration

M9:
Evidence fusion + consistency

M10:
Confidence

M11:
Benchmark evaluation

M12:
Report generation

M13:
UI polish

M14:
End-to-end judging/demo hardening

Do not jump randomly between milestones.

---

# 22. AGENT IMPLEMENTATION RULES

Before modifying the repository:

1. Inspect the existing files.
2. Understand current architecture.
3. Check for existing implementations.
4. Determine the smallest change that satisfies the requirement.
5. Identify dependencies.
6. Implement.
7. Run appropriate tests.
8. Verify the actual behavior.
9. Update documentation where necessary.

Do not blindly regenerate the entire project.

Do not create duplicate modules for functionality that already exists.

Do not rewrite stable code unnecessarily.

Do not delete working code without a clear reason.

---

# 23. NO FAKE IMPLEMENTATIONS

The following are prohibited:

- hardcoded fake AI answers presented as real inference;
- random confidence values;
- fake benchmark metrics;
- fake execution traces;
- fake masks/bounding boxes;
- claiming a model performed an operation when it did not;
- claiming fine-tuning occurred when only inference occurred;
- placeholder models left in the final judging path.

Mocks are permitted ONLY for:

- unit tests;
- temporary UI development;
- explicit development mode.

Mocks must be clearly labeled.

---

# 24. FALLBACKS

Experimental models can fail.

Implement graceful fallback behavior where practical.

If the primary specialist fails:

1. capture the failure;
2. attempt the approved fallback;
3. reduce confidence if appropriate;
4. expose a warning;
5. continue when safe.

Never silently substitute a generic model for a required specialist capability.

---

# 25. TESTING REQUIREMENTS

Write tests for:

- invalid files;
- unsupported formats;
- malformed GeoTIFFs;
- missing metadata;
- wrong modality;
- incompatible image pairs;
- temporal ordering;
- spatial compatibility;
- routing;
- model registry;
- specialist interfaces;
- evidence fusion;
- confidence;
- consistency checking;
- API contracts;
- report generation.

Create small test fixtures.

Do not require large public datasets to run basic tests.

---

# 26. EVALUATION REQUIREMENTS

Maintain reproducible evaluation scripts.

Where applicable track:

- VQA accuracy;
- grounding metrics;
- caption metrics;
- change detection metrics;
- change-VQA performance;
- optical-SAR performance;
- invalid-input robustness;
- latency;
- memory use.

Do not fabricate or selectively report metrics.

Document:

- dataset;
- split;
- preprocessing;
- checkpoint;
- metric;
- hardware;
- command used to reproduce the result.

---

# 27. SECURITY / DATA HYGIENE

Never commit:

- API keys;
- passwords;
- tokens;
- private credentials;
- large raw datasets;
- unnecessary model checkpoints.

Use:

- `.env`;
- `.env.example`;
- `.gitignore`.

Never print secrets to logs.

---

# 28. GIT RULES

Keep commits small and meaningful.

Preferred style:

feat:
fix:
refactor:
test:
docs:
chore:

Do not create giant commits containing unrelated functionality.

Before a commit:

1. inspect `git diff`;
2. run relevant tests;
3. verify no secrets or datasets are included;
4. commit only intentional changes.

Never force-push or rewrite shared history without explicit user approval.

---

# 29. FRONTEND RULES

The UI must communicate that SatQuery is a remote-sensing analytical system.

Prioritize:

- imagery;
- spatial evidence;
- temporal comparison;
- metadata;
- model execution;
- confidence;
- results.

Avoid unnecessary:

- animations;
- generic chatbot styling;
- decorative AI graphics;
- meaningless dashboards.

The primary judging flow should be obvious within seconds.

---

# 30. FINAL DEMO SCENARIOS

The final system must have reliable demonstrations for:

DEMO 1:
Single-image VQA.

DEMO 2:
Text-guided region grounding OR captioning.

DEMO 3:
Bi-temporal change analysis.

DEMO 4:
Optical-SAR joint analysis.

The strongest demonstration should visibly show:

1. uploaded imagery;
2. natural-language query;
3. automatic task selection;
4. selected specialist tools;
5. spatial evidence;
6. answer;
7. confidence;
8. execution trace.

---

# 31. FAILURE-HANDLING PRINCIPLE

When the system does not have enough evidence, it must say so.

Examples:

- low image quality;
- missing metadata;
- insufficient overlap;
- poor registration;
- unsupported modality;
- model disagreement;
- low-confidence classification.

Never manufacture certainty.

A qualified answer is better than a confidently wrong answer.

---

# 32. RESEARCH RULE

When choosing models, datasets or libraries:

- verify the source;
- prefer official repositories/model cards/papers;
- check licensing;
- check hardware requirements;
- check maintenance status;
- check whether the approach actually matches the task.

Do not adopt a tool solely because it is popular.

Before introducing a major dependency, explain:

1. why it is needed;
2. what problem it solves;
3. alternatives considered;
4. impact on installation/deployment;
5. hardware requirements.

---

# 33. DOCUMENTATION RULE

Keep the following documentation current:

docs/
├── ARCHITECTURE.md
├── REQUIREMENTS_TRACEABILITY.md
├── DATASET_PLAN.md
├── MODEL_PLAN.md
├── API_CONTRACT.md
├── EVALUATION.md
└── IMPLEMENTATION_PLAN.md

README.md must eventually include:

- project overview;
- architecture;
- setup;
- supported inputs;
- supported tasks;
- models;
- datasets;
- benchmark results;
- usage;
- demo instructions;
- known limitations.

---

# 34. REQUIREMENT TRACEABILITY

Every mandatory SIH requirement must map to:

REQUIREMENT
→ MODULE
→ MODEL/TOOL
→ TEST
→ DEMO
→ EVIDENCE

Maintain a requirements traceability document.

No mandatory requirement should exist only in prose.

---

# 35. PRIORITY RULE

When time is limited, prioritize:

P0:
Mandatory SIH functionality

P1:
Model quality and remote-sensing adaptation

P2:
Agentic orchestration

P3:
Evidence and robustness

P4:
Evaluation

P5:
UI polish

P6:
Nice-to-have features

Never sacrifice mandatory functionality for visual polish.

---

# 36. WHAT NOT TO BUILD EARLY

Do not spend early development time on:

- user authentication;
- complex account systems;
- social features;
- unnecessary databases;
- payment systems;
- elaborate landing pages;
- chat history;
- unnecessary cloud architecture;
- microservices.

The hackathon objective is the remote-sensing intelligence workflow.

---

# 37. DEFINITION OF "DONE"

A feature is NOT done because the code exists.

A feature is done only when:

- implementation exists;
- dependencies are reproducible;
- tests pass;
- real input works;
- output is verified;
- errors are handled;
- documentation is updated;
- integration works;
- the feature can be demonstrated.

---

# 38. CRITICAL ENGINEERING BEHAVIOR

Challenge weak decisions.

If an implementation:

- is generic rather than remote-sensing specific;
- uses a fake agent architecture;
- relies only on keyword routing;
- creates unsupported assumptions;
- cannot run on available hardware;
- has no measurable evaluation;
- produces unverifiable answers;
- duplicates existing functionality;
- introduces unnecessary complexity;

say so clearly and recommend a better approach.

Do not optimize for agreement.

Optimize for technical quality and SIH26167 compliance.

---

# 39. FIRST-RUN BEHAVIOR

When an agent starts in a new repository:

DO NOT immediately generate the entire application.

First:

1. inspect the repository;
2. read `AGENTS.md`;
3. inspect environment;
4. identify available Python/Node/GPU tooling;
5. create an architecture proposal;
6. create requirements traceability;
7. create dataset/model plan;
8. create API contract;
9. create implementation plan;
10. identify technical risks.

Only after this planning stage should implementation begin.

The first implementation should be the smallest end-to-end vertical slice that proves the architecture.

---

# 40. FINAL PRINCIPLE

Build a smaller system that actually works than a huge system that only looks impressive.

Every important claim must be demonstrable.

Every important output should be supported by evidence.

Every model should have a clearly defined purpose.

Every mandatory SIH requirement should be traceable.

Every major result should be reproducible.

SatQuery AI should look and behave like a serious remote-sensing intelligence platform.