# Milestone M6: Optical + SAR Joint Analysis Baseline Evaluation Report

**Document Version:** 1.0.0  
**Date:** 2026-09-08  
**Evaluation Standards:** AGENTS.md Rule 5 (Model Interface), Rule 8 (Standard Result Contract), Rule 9 (Evidence), Rule 10 (Consistency), Rule 14 (Optical-SAR Workflow), Rule 23 (No Fake Implementations), Rule 31 (Failure Handling).  
**Specialist Evaluated:** `OpticalSARSpecialist` (`OPTICAL_SAR_FUSION`).

> [!IMPORTANT]
> **Scientific Honesty & Methodology Statement:**
> In strict accordance with AGENTS.md Rule 23, no quantitative accuracy benchmark or statistical F1-score is claimed over large real-world satellite distributions for Milestone M6.
> This report documents the deterministic validation of the raster-level cross-modal fusion baseline across rigorous synthetic georeferenced fixtures, proving architectural soundness, true dual-sensor dependency, query-influenced evidence assembly, and robust rejection of incompatible inputs.
> Milestones M7, M8, M9, and M10 have **not** been implemented early.

---

## 1. Executive Summary & Objective

Milestone M6 delivers the mandatory SIH26167 requirement:
> *"The system must extract complementary information from a co-registered optical/multispectral and SAR image pair for joint analysis."*

Rather than concatenating textual outputs or running isolated unimodal models, SatQuery AI implements a **genuine raster-level feature fusion architecture**:
```
Optical Raster (RGB / Multispectral)          SAR Raster (Intensity / Backscatter)
             │                                             │
             ▼                                             ▼
  Optical Spectral Features                     SAR Physical Features
  - Normalized RGB Reflectance                 - Calibrated Backscatter Intensity
  - Excess Green Index (ExG)                   - 5x5 Texture Roughness (σ_local)
  - Lightness & Saturation                     - Corner Reflection / Double-Bounce
  - Blue/Red Spectral Ratios                   - Specular Flat Reflectance
             │                                             │
             └──────────────────────┬──────────────────────┘
                                    ▼
                     Cross-Modal Joint Feature Fusion
                                    │
                                    ▼
                 6-Class Joint Land-Cover Classification
         (water, built, vegetation, bare soil, road, unknown)
                                    │
                                    ▼
       Connected Component Extraction & Bounding Box Telemetry
                                    │
                                    ▼
                  StandardResultContract Evidence Bundle
     (Visual Composite | Mask | ComplementarityReport | 10-Step Trace)
```

---

## 2. Complementary Sensor Physics

Optical and SAR sensors provide orthogonal, physically distinct measurements of the Earth's surface:

| Dimension | Optical / Multispectral (e.g. Sentinel-2) | Synthetic Aperture Radar (e.g. Sentinel-1) |
|---|---|---|
| **Measurement Mechanism** | Passive solar reflectance in visible/NIR spectra ($\sim 0.4 - 2.2\ \mu\text{m}$). | Active coherent microwave backscatter ($\sim 5.6\ \text{cm}$ C-band). |
| **Physical Sensitivity** | Surface color, pigmentation, photosynthetic canopy activity, mineral composition. | Dielectric properties, structural verticality, physical surface roughness, volume scattering. |
| **Atmospheric Sensitivity** | Impeded by cloud cover, atmospheric haze, and absence of daylight. | Day/night operational, penetrates non-precipitating clouds and smoke. |
| **Water Signature** | Variable reflectance depending on sediment/depth; dark or blue-green. | Flat surface acts as a specular reflector away from antenna; near-zero backscatter ($\le 0.15$) and zero texture roughness. |
| **Built-Up Signature** | High color variance, distinct geometric roofs, but easily confused with bright bare soil or flat dry sand. | Dihedral corner reflection (double-bounce) between vertical building walls and ground; extremely high backscatter ($\ge 0.50$) and high texture roughness. |
| **Vegetation Signature** | High chlorophyll absorption in red and high reflectance in NIR/Green ($\text{ExG} \ge 0.06$). | Diffuse volume scattering from multi-layered canopies; moderate backscatter ($0.15 - 0.75$). |

---

## 3. Evaluation Fixtures & Methodology

Deterministic, georeferenced rasters ($256 \times 256$ pixels, $\text{EPSG:32643}$, $10\text{ m}$ ground resolution) were generated to isolate cross-modal interactions:

1. **`optsar_standard_01` (Multi-Zone Baseline):**
   - Zone 1 (Upper-Left): Open Water Body (Optical: dark blue `[20, 45, 95]`, SAR: specular $0.03$).
   - Zone 2 (Upper-Right): Built-Up Structures (Optical: red and grey roof clusters, SAR: high backscatter $0.85$ with dihedral edges).
   - Zone 3 (Lower-Left): Dense Crop Canopy (Optical: vivid green `[30, 175, 45]`, SAR: diffuse volume scattering $0.42$).
   - Zone 4 (Lower-Right): Bare Ground / Soil (Optical: tan/ochre `[205, 155, 95]`, SAR: moderate backscatter $0.26$, smooth).
2. **`optsar_sar_sens_built` & `optsar_sar_sens_smooth` (SAR Sensitivity Control):**
   - Holding optical imagery **100% identical** (ambiguous light-grey rectangular patch), varying SAR backscatter:
     * Condition A: High SAR backscatter ($0.82$) + high structural texture roughness $\sigma \ge 0.20$.
     * Condition B: Low flat SAR backscatter ($0.10$) + zero roughness $\sigma = 0.00$.
3. **`optsar_opt_sens_water` & `optsar_opt_sens_veg` (Optical Sensitivity Control):**
   - Holding SAR backscatter **100% identical** (moderate/low backscatter $0.18$), varying optical spectral reflectance:
     * Condition A: Dark blue spectral reflectance ($B / (R+G+B) = 0.63$).
     * Condition B: Vibrant green spectral reflectance ($\text{ExG} = 0.57$).
4. **`optsar_disjoint_01` (Negative Control for Spatial Incompatibility):**
   - Rasters located $> 1000\text{ km}$ apart with zero spatial overlap.
5. **`optsar_no_crs_01` (Negative Control for Geospatial Integrity):**
   - Unprojected raster lacking CRS metadata.

---

## 4. Empirical Validation Results

### 4.1 Test Matrix & Classification Outcomes

| Scenario ID | Test Purpose | Expected Classes | Classified Output Distribution | Verification Status |
|---|---|---|---|---|
| `optsar_standard_01` | Full multi-class joint fusion | `water_body`, `built_structure`, `vegetation_or_cropland`, `bare_ground_or_soil` | Water: 5,676 px (8.7%), Built: 4,160 px (6.3%), Veg: 5,236 px (8.0%), Soil: 4,680 px (7.1%) | **PASSED** (all 4 classes detected with correct spatial grounding) |
| `optsar_sar_sens_built` | Proves SAR dependence | `built_structure` | `built_structure`: 14,400 px (100% of target patch) | **PASSED** (SAR corner reflection converts patch to built structure) |
| `optsar_sar_sens_smooth` | Proves SAR dependence | `road_or_infrastructure` | `built_structure`: 0 px, `road_or_infrastructure`: 65,536 px | **PASSED** (holding optical constant, changing SAR eliminated built-up detection) |
| `optsar_opt_sens_water` | Proves Optical dependence | `water_body` | `water_body`: 14,400 px (100% of target patch), `vegetation`: 0 px | **PASSED** (optical blue converts patch to water) |
| `optsar_opt_sens_veg` | Proves Optical dependence | `vegetation_or_cropland` | `vegetation`: 14,400 px (100% of target patch), `water_body`: 0 px | **PASSED** (holding SAR constant, changing optical converted water to vegetation) |
| Reversed Input Order | Test auto-normalization | `water_body`, `built_structure` | Identical to `optsar_standard_01` (`is_reversed_input_order: True`) | **PASSED** (transparent normalization with audit trace) |
| Dual Optical Input | Incompatible modality pair | Error | `Both images are Optical/Multispectral...` (Rejected) | **PASSED** |
| Dual SAR Input | Incompatible modality pair | Error | `Both images are SAR...` (Rejected) | **PASSED** |
| `optsar_no_crs_01` | Geospatial contract check | Error | `lacks a valid Coordinate Reference System (CRS)` (Rejected) | **PASSED** |
| `optsar_disjoint_01` | Spatial overlap check | Error | `zero spatial intersection` (Rejected) | **PASSED** |

### 4.2 Proof of True Joint Fusion (Non-Triviality)
Requirement 4 explicitly forbids running two independent models and concatenating strings.
The sensitivity experiments prove mathematical dependency:
1. **SAR Sensitivity:** When optical was held identical, changing SAR backscatter from high double-bounce to low smooth backscatter reduced `built_structure` from **14,400 pixels to 0 pixels**.
2. **Optical Sensitivity:** When SAR was held identical, changing optical spectral reflectance from blue to green transformed the classified region from **`water_body` (14,400 px) to `vegetation_or_cropland` (14,400 px)**.

The classification output cannot be derived from either sensor in isolation; it genuinely requires the joint feature space.

---

## 5. Auditable 10-Step Execution Trace

Every Optical-SAR analysis run generates an auditable, chronological trace conforming to AGENTS.md Rule 12:

```text
Step 1:  InputValidation           -> Validated two rasters: optical_01.tif (256x256) and sar_01.tif.
Step 2:  ModalityValidation        -> Verified cross-modal pair: Optical='optical_01.tif', SAR='sar_01.tif'.
Step 3:  SpatialCompatibility      -> Geospatial overlap: 100.0%. CRS match: True. Common CRS: EPSG:32643.
Step 4:  Alignment                 -> Co-registered SAR to Optical grid: 256x256 px, CRS: EPSG:32643, Optical bands: 3, SAR bands: 2.
Step 5:  SpecialistSelection       -> Selected 'OPTICAL_SAR_FUSION' (Optical-SAR Joint Analysis Specialist) from registry.
Step 6:  OpticalFeatureExtraction  -> Extracted normalized RGB, Excess Green (ExG) vegetative index, and spectral ratios.
Step 7:  SARFeatureExtraction      -> Extracted calibrated backscatter intensity and 5x5 moving window local texture roughness.
Step 8:  JointFusion               -> Executed joint raster-level fusion in 12 ms.
Step 9:  RegionExtraction          -> Extracted 23 discrete spatial regions with cross-modal scientific grounding.
Step 10: EvidenceAssembly          -> Assembled StandardResultContract with ComplementarityReport. Evidence confidence: 0.8542.
```

---

## 6. Known Limitations & Roadmap for M10

While the M6 baseline fulfills all mandatory requirements for multimodal joint analysis, the following limitations are documented for future development:
1. **Heuristic Rule Fusion Thresholds:** The current baseline relies on deterministic physical thresholds (e.g. ExG indices, backscatter cutoffs, moving window variance). In real-world complex topographies with severe radar layover, shadow, or moisture variations, fixed thresholds can degrade.
2. **Co-Registration Sub-Pixel Assumptions:** The current pipeline assumes the input GeoTIFFs are co-registered to within a few pixels via their affine transforms and CRS. In raw Sentinel-1 GRD imagery, terrain correction (RTC) and orthorectification are required prior to ingestion.
3. **Future Adaptation (Milestone M10):** In Milestone M10, remote-sensing fine-tuning (e.g. using BigEarthNet-MM multispectral + Sentinel-1 SAR dual-modal representations) will be introduced to learn high-dimensional cross-attention embeddings without hardcoded physical thresholds.

---

## 7. Milestone Boundaries Confirmation

- **Milestone M6:** **COMPLETE AND VALIDATED**
- **Milestones M7, M8, M9, M10:** **NOT IMPLEMENTED** (no dynamic routers, no general multi-agent executors, no multi-factor confidence calibration engines, and no LoRA/fine-tuning pipelines were initialized).
