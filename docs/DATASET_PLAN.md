# SatQuery AI — Dataset Strategy & Benchmark Management Plan

**Document Version:** 2.0.0 (Post-Audit Revision)  
**Project:** SatQuery AI (SIH26167 / ISRO)  
**Standard:** Compliance with `AGENTS.md` Rule 15 (Purposeful use, no benchmark leakage, license tracking, storage budgeting).

---

## 1. Critical Audit Findings & Strategy Adjustments

### 1.1 Critical Audit Correction: BigEarthNet vs. VQA
- **Audit Finding:** BigEarthNet (v1.0 and BigEarthNet-MM) is a **multi-label land cover classification dataset** of Sentinel-2 (12-band) and Sentinel-1 (VV, VH) patches with CORINE Land Cover (CLC) classes (43 original or 19 merged classes). **It contains NO question-answer pairs.**
- **Impact on Previous Plan:** Claiming to fine-tune a conversational VQA model directly on raw BigEarthNet labels was technically invalid.
- **Correction:** 
  1. We incorporate **RSVQAxBEN** (Lobry et al., 2021), which directly takes BigEarthNet Sentinel-2 image tiles and generates ~10.6 million visual question-answer pairs (covering object presence, comparison, rural/urban classification, and land-use area). This explicitly satisfies both the BigEarthNet data mandate and the VQA fine-tuning mandate.
  2. For general remote-sensing VQA, captioning, and text-guided region grounding, we utilize **VRSBench** (OpenGVLab).
  3. Raw BigEarthNet-S2 (10,000 patch curated subset) is utilized for adapting the remote-sensing **vision feature backbone** to multispectral reflectance bands.

### 1.2 Spatial Autocorrelation Leakage Prevention
- **Risk:** Sentinel-2 and aerial images exhibit high spatial autocorrelation. Randomly shuffling $120 \times 120$ patches into train and test sets causes patches from the exact same $100 \text{ km} \times 100 \text{ km}$ tile to appear in both sets, inflating evaluation accuracy artificially.
- **Correction:** We strictly enforce the **official tile-based split** (Sumbul et al., 2021) for BigEarthNet and official disjoint geographic splits for LEVIR-CD and VRSBench. No patch from a training tile is ever placed in validation or test sets.

---

## 2. Dataset Strategy Status & Decision Classifications

### 2.1 CONFIRMED Decisions
- **Curated Multi-Tier Strategy:** We do not download multi-terabyte raw catalogs. We maintain:
  - **Tier 0:** Lightweight synthetic/micro-cropped fixtures in `tests/fixtures/` (< 5MB total) for instantaneous CI/CD testing.
  - **Tier 1:** Standard held-out evaluation benchmarks with frozen ground-truth annotations.
  - **Tier 2:** Curated, balanced subsets for local fine-tuning on our RTX 4070 (12 GB VRAM).
- **Dual-Task Change Benchmarks:** We separate **pixel change detection** (evaluated on LEVIR-CD with IoU/F1 metrics) from **change captioning / change VQA** (evaluated on LEVIR-CC / CDVQA with BLEU-4, METEOR, and CIDEr metrics).
- **Zero In-Repo Test Leakage:** The in-repo test fixtures (`tests/fixtures/`) are solely dedicated to functional and regression unit tests and are never used in training.

### 2.2 ASSUMPTIONS
- Curated subsets of BigEarthNet-S2/MM, RSVQAxBEN, LEVIR-CD, and VRSBench will be stored in a local `datasets/` directory ignored by Git.
- Image patches can be efficiently loaded in-memory or streamed as GeoTIFF / PNG chunks.

### 2.3 TO VERIFY Items
- **Download Bandwidth & Checksums:** Verify SHA256 checksums and reliable download endpoints for RSVQAxBEN, VRSBench, and LEVIR-CD subsets.
- **Pretrained Weights Availability:** Verify whether verified public checkpoints trained on LEVIR-CD (TinyCD / BIT) and VRSBench (Florence-2-RS) can be downloaded directly before executing local adaptation.

### 2.4 RISKS
- **License Incompatibilities:** VRSBench is CC BY-NC-SA 4.0 (academic non-commercial). Ensure hackathon / research demonstration compliance.
- **Band Resampling Discrepancies:** Sentinel-2 bands have differing spatial resolutions (10m, 20m, 60m). Improper interpolation can introduce edge artifacts.  
  *Mitigation:* Use standard bilinear interpolation for continuous reflectance and nearest-neighbor for categorical land cover.

### 2.5 RECOMMENDED Datasets
- **RSVQAxBEN / VRSBench:** For RS VQA and Region Grounding adaptation.
- **BigEarthNet-S2 / BigEarthNet-MM:** For multispectral representation and optical-SAR joint classification.
- **LEVIR-CD & LEVIR-CC:** For bi-temporal change detection and semantic change captioning.
- **Sen1-2:** For optical-SAR cross-modal alignment and cloud penetration validation.

---

## 3. Dataset Master Catalog

| Dataset Name | Source / Provider | License | Modality | Primary Task | Split Strategy | Curated Subset Size | Storage Footprint | Role in SatQuery AI |
|---|---|---|---|---|---|---|---|---|
| **RSVQAxBEN** | Sylvain Lobry et al. (built on BigEarthNet) | Open Academic / MIT | Sentinel-2 (RGB / Multi-spectral) | Remote-Sensing VQA | Official Tile-based Split (Train / Val / Test) | 10,000 QA pairs across 2,000 tiles | ~1.8 GB | Primary dataset for fine-tuning RS VQA on BigEarthNet imagery (REQ-03, REQ-08). |
| **BigEarthNet-S2 (v1.0)** | TU Berlin / ESA | CDLA-Permissive-1.0 | 12-Band Sentinel-2 GeoTIFF | Multi-label Land Cover (19 classes) | Official Tile Split (Sumbul et al. 2021) | 5,000 multi-band patches | ~3.2 GB | Feature representation adaptation for multi-spectral band understanding (REQ-01, REQ-08). |
| **BigEarthNet-S1 (v1.0)** | TU Berlin / ESA | CDLA-Permissive-1.0 | Dual-pol Sentinel-1 SAR (VV, VH) GeoTIFF | SAR backscatter representation | Official Tile Split | 2,500 SAR patches | ~1.6 GB | SAR backscatter calibration and cross-modal feature alignment (REQ-02, REQ-07). |
| **VRSBench** | OpenGVLab / Wuhan University | CC BY-NC-SA 4.0 | High-Res Aerial / Satellite (RGB) | RS VQA, Captioning, Region Grounding | Official Test Split | 1,500 images + QA + bounding boxes | ~1.8 GB | Evaluation benchmark for single-image VQA (REQ-03) and region grounding (REQ-04). |
| **LEVIR-CD** | Beihang University | Open Academic | Bi-temporal Optical Pairs (RGB, 0.5m) | Building & Surface Change Detection | Official Split (Train: 70%, Test: 20%, Val: 10%) | 500 pairs ($512 \times 512$ crops) | ~2.2 GB | Benchmark for pixel change detection mask generation (REQ-05, DEMO 3). |
| **LEVIR-CC / CDVQA** | Beihang University | Open Academic | Bi-temporal Optical Pairs + Captions/QA | Change Captioning & Change VQA | Official Test Split | 800 change pairs with descriptions | ~1.2 GB | Benchmark for semantic change narrative and change-based VQA (REQ-06, DEMO 3). |
| **Sen1-2** | TU Munich (TUM) | Open Academic | Co-registered Sentinel-1 SAR & Sentinel-2 Optical | Optical + SAR Cross-Modal Fusion | Validation Split | 500 co-registered pairs | ~1.2 GB | Benchmark for optical-SAR joint analysis and cloud penetration (REQ-07, DEMO 4). |

**Total Estimated Storage Footprint:** **~13.0 GB** (Well within our 172 GB free disk space on Drive C).

---

## 4. In-Repo Test Fixtures (`tests/fixtures/`)

To enable lightning-fast, fully offline unit and integration testing (< 3 seconds total runtime), the repository includes self-contained synthetic and micro-cropped test fixtures checked into Git:

```
tests/fixtures/
├── optical_s2_sample.tif         # 3-band RGB GeoTIFF, EPSG:32643, 256x256, 10m resolution (~250 KB)
├── sar_s1_sample.tif             # 2-band VV/VH GeoTIFF, EPSG:32643, 256x256, 10m resolution (~200 KB)
├── bitemporal/
│   ├── time1_pre_change.tif      # Timestamp 1 GeoTIFF, 256x256 (~250 KB)
│   ├── time2_post_change.tif     # Timestamp 2 GeoTIFF, 256x256 with synthetic new structure (~250 KB)
│   └── change_ground_truth.png   # Binary change mask (256x256) (~15 KB)
├── invalid/
│   ├── corrupted_header.tif      # Deliberately broken GeoTIFF header for error tests (~2 KB)
│   ├── unprojected_image.png     # Plain PNG with no geospatial CRS or transform (~50 KB)
│   └── mismatched_crs.tif        # GeoTIFF with conflicting EPSG:4326 vs UTM bounds (~50 KB)
└── mock_qa_pairs.json            # Structured sample VQA, Grounding, and Change queries for offline tests
```
Total footprint: **< 1.5 MB**, requiring zero network connectivity for standard CI/CD test execution.
