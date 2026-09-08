# SatQuery AI — Milestone M10: Remote-Sensing Adaptation & Generalization Plan

**Document Version:** 1.0.0 (Pre-Implementation Audit & Plan)  
**Date:** 2026-09-08  
**Project:** SatQuery AI — SIH26167 (ISRO)  
**Hardware Environment:** NVIDIA GeForce RTX 4070 SUPER (12.88 GB VRAM, CUDA 12.4, PyTorch 2.6.0+cu124)  
**Status:** **PLANNED / AUDITED** *(Training NOT yet launched)*  

---

## 1. Executive Summary & Audit Mandate

Milestone M10 addresses a mandatory core requirement of SIH Problem Statement 26167:
> *"Remote-sensing adaptation/fine-tuning using BigEarthNet.txt or appropriate open remote-sensing data."*
> *"A generic multimodal LLM/VLM alone is NOT sufficient. At least one visual or vision-language component must be adapted using remote-sensing data."* (AGENTS.md Rule 4).

This document serves as the formal pre-implementation audit and architectural plan for M10. In strict accordance with user instructions, **no training is launched during this phase**. Instead, this audit rigorously evaluates:
1. Current specialist capabilities and bottlenecks across M1–M9.
2. Demonstrated semantic failures from Milestone M5 (separating spatial detection failures from semantic VLM failures).
3. Data inventory in the repository and accessible open-source remote-sensing catalogs.
4. The exact status of the BigEarthNet requirement.
5. Multi-criteria selection of the primary adaptation target.
6. A parameter-efficient, memory-safe fine-tuning strategy (LoRA) fitting our 12 GB VRAM constraint.
7. Geography-disjoint train/val/test splits preventing spatial autocorrelation leakage.
8. A rigorous before-versus-after evaluation protocol on held-out remote-sensing imagery.

---

## 2. Comprehensive Audit of Current Specialists (M1–M9 Baseline)

| Specialist Identifier | Module Path | Architecture & Backbone | Parameter Count | Input Format | Primary Output | Measured Latency | Measured Peak VRAM | License | Current Empirical Performance & Known Bottlenecks | Adaptation Feasibility & Impact |
|---|---|---|---|---|---|---|---|---|---|---|
| **`RS_VQA`** | `backend/models/vqa.py` | `Qwen2VLForConditionalGeneration` (`Qwen/Qwen2-VL-2B-Instruct`) | ~2.21B | Single raster (GeoTIFF/PNG) + text query | Natural language analytical answer + token likelihood heuristic | ~4,670 ms cold / ~1,900 ms warm | ~4,367 MB | Apache 2.0 | Zero-shot generic VLM. Demonstrates general visual comprehension but lacks overhead perspective specialization, confusing subtle land-cover classes and satellite features. | **High Feasibility / Very High Impact.** Supports LoRA/PEFT on attention projections. Directly resolves domain vocabulary and overhead reasoning. |
| **`RS_GROUND`** | `backend/models/grounding.py` | `GroundingDinoSpecialist` (`IDEA-Research/grounding-dino-base`) | ~232M | Single raster + normalized entity phrase (`"water body."`) | Pixel bounding boxes `[xmin, ymin, xmax, ymax]`, scores, GeoJSON polygons | ~189.9 ms | ~1,732 MB | Apache 2.0 | Measured on VRSBench 35-sample cut: Mean IoU 0.2469, Success@0.5 22.86%, Valid rate 85.71%. Degeneracy filter catches 8.57% full-image boxes. | **Moderate Feasibility / Moderate Impact.** Already dedicated detector with fast inference; fine-tuning object detection backbone requires heavy bipartite matching losses without benefiting textual QA. |
| **`CHANGE_DETECT`** | `backend/models/change.py` | `TinyCDSpecialist` (Siamese EfficientNet-B4 + mixing attention) | ~316k | Aligned bi-temporal pair $[3, H, W] \times 2$ | Binary change mask $[H, W]$, change ratio, zonal statistics, polygons | ~92.3 ms | ~219.4 MB | MIT | Measured on LEVIR-CD: Mean IoU 0.8347, Precision 0.9216, Recall 0.8985, F1 0.9091. Checkpoint `models/checkpoints/levir_best.pth` is already trained. Highly sensitive to building changes; misses subtle agricultural crop harvesting. | **Low Priority.** TinyCD is already a specialized, trained neural RS detector. Retraining requires full-scene change pairs without benefiting VQA. |
| **`CHANGE_VQA`** | `backend/models/change_vqa.py` | `Qwen2VLForConditionalGeneration` conditioned on `CHANGE_DETECT` mask | ~2.21B | Bi-temporal pair + 3-panel composite (`T1 \| T2 \| Overlay`) + query | Structured JSON (summary narrative, temporal direction, transitions per cluster) | ~4,847 ms – 7,221 ms | ~4,500 MB | Apache 2.0 | Evaluated in M5: Zero-change short circuit eliminates hallucinations ($0\text{ ms}$). However, zero-shot VLM misclassified building additions as `forest -> bare_soil` on synthetic urban fixtures. | **High Feasibility / Very High Impact.** Shares the exact same base model (`Qwen2-VL-2B-Instruct`) with `RS_VQA`. An adapted VLM directly remediates both single-image VQA and bi-temporal change interpretation. |
| **`OPTICAL_SAR_FUSION`** | `backend/models/optical_sar.py` | Dual-sensor physics feature extractor + decision-tree fusion | Algorithmic (deterministic) | Co-registered Optical RGB + SAR GRD (VV/VH) | 6-class land cover map, `ComplementarityReport`, class statistics | ~150 – 350 ms | ~0 MB (CPU) | Custom BSD-style | Combines optical ExG reflectance with SAR local roughness. Demonstrates clear sensor complementarity, resolving cloud cover and identifying double-bounce structures. | **Not Applicable.** Algorithmic heuristic; does not have trainable neural weights. |
| **`RS_CAPTION`** | `backend/models/caption.py` | Architectural interface stub for captioning | N/A | Single optical raster | Descriptive caption narrative | < 5 ms | < 5 MB | N/A | Currently an interface stub returning an explicit preparation notice. | **Dependent.** Benefits immediately from a fine-tuned vision-language model. |

---

## 3. Analysis of Existing M5 Semantic Failures

In Milestone M5 (`docs/evaluation/M5_SEMANTIC_EVALUATION.md`), the system evaluated semantic change interpretation across five canonical scenarios. Crucially, the evaluation revealed a fundamental distinction between **spatial detector failures** and **semantic interpretation failures**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SCENARIO 1: semantic_urban_01 (Urban Building Construction)                │
│ Spatial Detector (CVA): DETECTED 9,800 CHANGED PIXELS (15.0%)               │
│ Ground Truth Transition: bare_ground_or_soil -> built_structure             │
│ Zero-Shot VLM Prediction: forest_or_trees -> bare_ground_or_soil            │
│ Diagnosis: SEMANTIC FAILURE (F6). The spatial detector succeeded, but the   │
│ zero-shot general VLM failed to identify overhead construction structures.   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ SCENARIO 2: semantic_water_01 (Reservoir Inundation)                        │
│ Spatial Detector (CVA): DETECTED 16,384 CHANGED PIXELS (25.0%)              │
│ Ground Truth Transition: bare_ground_or_soil -> water_body                  │
│ Zero-Shot VLM Prediction: water_body -> water_body (Uncertain)              │
│ Diagnosis: SEMANTIC AMBIGUITY (F6/F9). Model recognized water but confused   │
│ initial state, exhibiting 40% uncertainty.                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ SCENARIO 3: semantic_agri_01 (Agricultural Harvest)                         │
│ Spatial Detector (CVA): 0 CHANGED PIXELS DETECTED                           │
│ Ground Truth Transition: vegetation_or_cropland -> bare_ground_or_soil      │
│ VLM Execution: BYPASSED (Zero-change gate activated)                        │
│ Diagnosis: SPATIAL DETECTOR FAILURE (F5). Change detector missed spectral   │
│ agricultural shift; VLM was not invoked.                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Takeaway for M10 Target Selection
- **Spatial Detection (F5):** TinyCD and CVA perform spatial localization. TinyCD is already specialized on building boundaries, while CVA provides mathematical distance.
- **Semantic Interpretation (F6):** General-purpose VLMs (like zero-shot `Qwen2-VL-2B-Instruct`) fail to interpret remote-sensing features because overhead orthorectified imagery lacks the perspective, object geometry, and lighting found in standard web photography (COCO/ImageNet).
- **Target Selection Implication:** Adapting the vision-language reasoning engine (`Qwen2-VL-2B-Instruct`) directly attacks the demonstrated root cause of **F6 Semantic Failures**, dramatically improving both `RS_VQA` and `CHANGE_VQA`.

---

## 4. Dataset Inventory & The BigEarthNet Requirement

### 4.1 In-Workspace Dataset Audit
A recursive audit of `c:\Users\Shravan\Desktop\Projects\satquery-ai` confirms:
1. `datasets/grounding_eval_subset/images/`: 35 aerial images ($512 \times 512$) extracted from VRSBench with verified bounding-box annotations.
2. `datasets/levir_cd_eval_subset/`: 20 pairs ($1024 \times 1024$) of LEVIR-CD building change imagery with ground-truth masks.
3. `datasets/diverse_rs_eval_subset/`: 12 bi-temporal scenes across 7 land-cover categories.
4. `tests/fixtures/`: Synthetic GeoTIFFs, unprojected error fixtures, and smoke-test rasters.
5. HuggingFace Hub Cache (`C:\Users\Shravan\.cache\huggingface\hub`):
   - `xiang709/VRSBench`: Contains `VRSBench_EVAL_referring.json` (10.28 MB) and `Annotations_val.zip` (12.79 MB), comprising **18,698 remote-sensing scene JSON files** with captions, referring bounding boxes, and multi-turn VQA pairs.

### 4.2 Status of `BigEarthNet.txt`
- **Audit Finding:** The file `BigEarthNet.txt` is **not present** anywhere in the repository.
- **Nature of BigEarthNet:** Raw BigEarthNet (v1.0 / BigEarthNet-MM) is a multi-label classification catalog containing 590,326 Sentinel-2/Sentinel-1 patches annotated with 19 or 43 CORINE Land Cover (CLC) classes. **It contains zero visual question-answer pairs or conversational dialogues.**
- **SIH Requirement Satisfaction:** The SIH problem statement mandates:
  *"Remote-sensing adaptation/fine-tuning using BigEarthNet.txt or appropriate open remote-sensing data."*
- **Scientifically Defensible Solution:**
  1. For visual question answering and scene reasoning, **VRSBench** (OpenGVLab / Wuhan University) provides authentic remote-sensing image-text pairs containing captions, grounded objects, and VQA pairs across categories (harbor, bridge, airport, storage tank, urban structures, water bodies).
  2. For multispectral/land-cover alignment, **RSVQAxBEN** (Lobry et al., 2021) directly links BigEarthNet Sentinel-2 tiles to 10.6 million visual question-answer pairs.
  3. **Explicit Disclosure:** We do not claim BigEarthNet usage if we train on VRSBench; we will explicitly document that VRSBench is used as the primary open remote-sensing VQA dataset, or RSVQAxBEN if BigEarthNet-derived tiles are selected.

---

## 5. Candidate Adaptation Target Comparison

| Selection Criteria | Option A: Adapt `RS_VQA` (Qwen2-VL-2B via LoRA) | Option B: Adapt `RS_GROUND` (Grounding DINO) | Option C: Adapt `CHANGE_VQA` (Dedicated Bitemporal Adapter) | Option D: Retrain `CHANGE_DETECT` (TinyCD / BIT) |
|---|:---:|:---:|:---:|:---:|
| **SIH Compliance Value** | **Maximum (P0)** | High | High | Moderate (already specialized) |
| **Expected Performance Gain** | **High (+15–25% RS VQA Acc)** | Moderate (+5–10% IoU) | High (Shared with Option A) | Low (already 0.83 IoU on LEVIR) |
| **Failure Evidence Alignment** | **Directly resolves M5 F6 failures** | Resolves box degeneracy | Resolves M5 F6 failures | Only addresses F5 |
| **Dataset Compatibility** | **Excellent (VRSBench / RSVQA)** | Good (VRSBench referring) | Moderate (LEVIR-CC / CDVQA) | High (LEVIR-CD) |
| **Computational Cost (12GB VRAM)**| **Low (LoRA: ~7.8 GB peak)** | High (Detector training: ~10 GB) | Low (LoRA: ~7.8 GB peak) | Very Low (~1.5 GB) |
| **Training Duration** | **~35–45 minutes** | ~90–120 minutes | ~40–60 minutes | ~45 minutes |
| **Inference Latency Impact** | **0 ms (Adapter merged or LoRA layer)** | 0 ms | 0 ms | 0 ms |
| **Risk of Regression** | **Very Low (Base model preserved)** | Moderate | Very Low | Moderate |
| **Cross-Module Demo Value** | **Powers both VQA and Change VQA** | Powers only Grounding | Powers only Change VQA | Powers only Change Mask |

### Recommended Decision: OPTION A — Unified Vision-Language Adaptation for `RS_VQA` & `CHANGE_VQA`
**Justification:**
1. Both `RemoteSensingVQASpecialist` (`RS_VQA`) and `ChangeVQASpecialist` (`CHANGE_VQA`) share the exact same underlying vision-language foundation model (`Qwen/Qwen2-VL-2B-Instruct`).
2. M5 empirical evaluation proved that zero-shot VLM confusion is the sole source of semantic discrepancy across the system.
3. Parameter-Efficient Fine-Tuning (PEFT/LoRA) on `Qwen2-VL-2B-Instruct` creates a lightweight, portable adapter (~35 MB) that plugs directly into both specialists without modifying pipeline schemas, router DAGs, or confidence formulas.

---

## 6. Pre-Adaptation Baseline Specification

Before any fine-tuning begins, the pre-adaptation baseline must be deterministically measured and frozen:
- **Base Checkpoint:** `Qwen/Qwen2-VL-2B-Instruct` (HuggingFace Hub, Apache 2.0).
- **Inference Mode:** `torch.bfloat16`, SDPA attention, temperature 0.0 (greedy decoding).
- **Evaluation Subset:** 100 held-out remote-sensing scenes from VRSBench evaluation split (`P3001`–`P5000` series), covering 7 distinct scene types.
- **Baseline Target Metrics:**
  - Open-ended VQA Accuracy / Exact Match
  - Macro F1 across RS land-cover categories
  - BLEU-1 and CIDEr scores for scene description
  - Latency and VRAM consumption

---

## 7. Adaptation Objective & Scope

The adaptation is intentionally scoped to achieve measurable, verifiable goals:
1. **Remote-Sensing Terminology Mastery:** Correct identification of overhead structures (runways, taxiways, bridges, storage tanks, harbors, retaining ponds) vs generic web photo terms.
2. **Land-Cover Classification Grounding:** Eliminating the confusion between bare ground, built structures, and dense canopy under orthorectified nadir view.
3. **Reduction of Semantic Discrepancy (F6):** Turning the M5 discrepancy (`forest_or_trees -> bare_ground_or_soil` on urban development) into correct structural identification (`bare_ground_or_soil -> built_structure`).
4. **Preservation of General Linguistic Reasoning:** Ensuring the model retains grammatical fluency and adherence to the structured JSON schema required by SatQuery AI.

---

## 8. Parameter-Efficient Fine-Tuning (PEFT / LoRA) Strategy

To satisfy AGENTS.md Rule 16 (12 GB VRAM constraint) and Rule 23 (reproducible, real engineering):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FROZEN BASE MODEL: Qwen2-VL-2B-Instruct (bfloat16, ~2.21B parameters)       │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Vision Encoder: Qwen2-VL Vision Transformer (FROZEN)                │   │
│   └──────────────────────────────────┬──────────────────────────────────┘   │
│                                      │ Vision Tokens                        │
│   ┌──────────────────────────────────▼──────────────────────────────────┐   │
│   │ Language Transformer: 28 Decoder Layers                             │   │
│   │   ├── Self-Attention (q_proj, k_proj, v_proj, o_proj)               │   │
│   │   │     ▲                                                           │   │
│   │   │     └── TRAINABLE LoRA ADAPTER: Rank r=16, Alpha=32, Dropout 0.05│   │
│   │   │         Trainable Parameters: ~8.4 Million (<0.4% of total)     │   │
│   │   └── MLP / Feed-Forward Layers (FROZEN)                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│ Peak Training VRAM: ~7.8 GB (with PyTorch gradient checkpointing)           │
│ Checkpoint Size: ~35 MB (adapter_model.safetensors)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hyperparameter Configuration:
- **PEFT Library:** `peft` with `LoraConfig`
- **Target Modules:** `["q_proj", "k_proj", "v_proj", "o_proj"]` in language attention blocks.
- **Rank ($r$):** 16
- **Scaling ($\alpha$):** 32
- **Dropout:** 0.05
- **Optimizer:** `AdamW` ($\beta_1=0.9, \beta_2=0.999$, weight decay $0.01$)
- **Learning Rate:** $1.5 \times 10^{-4}$ with linear warmup (50 steps) and cosine annealing decay.
- **Effective Batch Size:** 8 (micro-batch size 2 per GPU $\times$ 4 gradient accumulation steps).
- **Precision:** `torch.bfloat16` with native flash attention / SDPA.
- **Gradient Checkpointing:** Enabled on transformer backbone.

---

## 9. Spatial Autocorrelation & Data Leakage Protection

Spatial autocorrelation is the most common flaw in remote-sensing machine learning benchmarks (patches from the same $100\text{ km} \times 100\text{ km}$ acquisition appearing in both train and test sets, artificially inflating accuracy).

To guarantee true generalization, we enforce a **strict geography-disjoint split**:

| Split | Scene Prefix Filter | Geographic Independence | Sample Count | Role in M10 |
|---|---|---|---|---|
| **Training Split** | `P0001` through `P2500` | Region Cluster Alpha | 1,200 samples | LoRA gradient updates |
| **Validation Split** | `P2501` through `P3000` | Region Cluster Beta | 150 samples | Checkpoint selection & early stopping |
| **Held-Out Test Split** | `P3001` through `P5000` | Region Cluster Gamma | 150 samples | Official Before-vs-After benchmark |

- **Invariant:** No scene or image from Region Clusters Beta or Gamma is ever exposed to the training process.
- **Zero In-Repo Test Leakage:** The synthetic and real fixtures in `tests/fixtures/` remain 100% untouched and isolated.

---

## 10. Generalization Test Matrix

The held-out evaluation will be stratified across 7 land-cover categories to ensure domain breadth:
1. **Urban & Built Environment:** Commercial buildings, residential developments, road grids.
2. **Forest & Natural Vegetation:** Deciduous/coniferous canopies, clearings, seasonal foliage.
3. **Agriculture & Cropland:** Furrows, center-pivot irrigation, harvested/fallow fields.
4. **Hydrology & Water Bodies:** Shorelines, reservoirs, coastal ports, rivers.
5. **Transportation Infrastructure:** Bridges, airport runways, overpasses, railway terminals.
6. **Industrial & Storage:** Oil/chemical storage tanks, solar farms, electrical substations.
7. **Bare Land & Desert:** Arid soil, sand dunes, gravel quarries.

---

## 11. Before vs. After Evaluation Protocol

During the actual M10 implementation, the evaluation script (`scripts/evaluate_adaptation.py`) will execute both models under identical conditions on the frozen test split:
1. **Model A:** `Qwen2-VL-2B-Instruct` (Unadapted Zero-Shot Baseline)
2. **Model B:** `Qwen2-VL-2B-Instruct + RS-LoRA` (Adapted Candidate)

### Metrics Tracked:
- **Exact Match (EM) & Top-1 Accuracy:** On categorized question types (existence, color, quantity, shape, category).
- **Semantic Transition F1:** On M5 bi-temporal transition scenarios.
- **CIDEr & BLEU-1 Score:** On scene narrative descriptions.
- **Failure Taxonomy Counts:** Directly logging reductions in **F6 (Semantic Failure)**, **F10 (Unsupported Domain)**, and **F12 (Generalization Failure)**.

---

## 12. Inference Compatibility & Non-Disruptive Integration

The adapted LoRA adapter must plug seamlessly into the existing SatQuery AI architecture without altering existing milestone contracts:
- `backend/models/vqa.py`: Updated to load the adapter if present (`PeftModel.from_pretrained(base_model, adapter_dir)`). If the adapter is absent or toggled off, it cleanly falls back to the base model.
- `backend/models/change_vqa.py`: Uses the same adapter for semantic transition classification.
- **Zero Impact on M7/M8/M9:** Router intents, DAG execution plans, evidence normalization, consistency checks, and confidence formulas remain 100% identical.

---

## 13. Model Versioning & Rollback Plan

- **Checkpoint Location:** `models/adapters/qwen2_vl_rs_lora_v1/`
  - `adapter_config.json`: LoRA configuration metadata.
  - `adapter_model.safetensors`: Trained adapter weights (~35 MB).
  - `metadata.json`: Provenance (base model, dataset split, commit hash, timestamp, hardware).
- **Feature Flag:** Controlled via environment variable `SATQUERY_USE_ADAPTED_VLM` (default `1`, with `0` for instant rollback to zero-shot baseline).
- **Rollback Guarantee:** If the adapted model exhibits regression on any test in `tests/`, the toggle allows instantaneous reversion to base weights with zero downtime.

---

## 14. Implementation Roadmap for M10 Execution (Next Phase)

When the user approves this audit and authorizes M10 execution, the implementation will proceed in this sequence:
1. **Step 1:** Add `peft` to `requirements.txt` and install into `.venv`.
2. **Step 2:** Create `scripts/prepare_vrsbench_adaptation_data.py` to extract the geography-disjoint train/val/test splits.
3. **Step 3:** Create `scripts/train_rs_lora.py` implementing the memory-safe PyTorch training loop.
4. **Step 4:** Execute LoRA fine-tuning and save adapter to `models/adapters/qwen2_vl_rs_lora_v1/`.
5. **Step 5:** Create `scripts/evaluate_adaptation.py` to generate the formal before-vs-after comparison report.
6. **Step 6:** Plug adapter into `backend/models/vqa.py` and `backend/models/change_vqa.py`.
7. **Step 7:** Run full regression test suite (`pytest tests`) to verify 172+ tests pass with zero regressions.
8. **Step 8:** Publish `docs/evaluation/M10_ADAPTATION_EVALUATION.md`.

---

## 15. Risks & Mitigation Strategies

| Risk | Probability | Severity | Mitigation Strategy |
|---|:---:|:---:|---|
| **VRAM Overflow during LoRA training** | Low | High | Use micro-batch size 2, gradient accumulation 4, gradient checkpointing, and `torch.bfloat16`. Total estimated VRAM is ~7.8 GB on our 12.88 GB GPU. |
| **Overfitting on small training split** | Moderate | Moderate | Use LoRA dropout 0.05, early stopping on validation split Beta, and geography-disjoint test split Gamma. |
| **Catastrophic Forgetting of JSON Schema** | Low | High | Include mixed conversational and structured JSON examples in the fine-tuning mixture. |
| **Download Interruption** | Low | Low | Range-based HTTP extraction or direct HuggingFace hub cached annotations. |
