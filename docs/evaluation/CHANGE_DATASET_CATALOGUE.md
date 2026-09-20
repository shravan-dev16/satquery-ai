# SatQuery AI — Bi-Temporal Change Dataset Catalogue & Format Inventory

**Document Version:** 2.0.0 (Format-Robust Bi-Temporal Extension)  
**Date:** 2026-09-20  
**Project:** SatQuery AI — SIH26167 (ISRO)  
**Hardware Target:** NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM)

---

## 1. Prioritized Bi-Temporal Datasets Ranking

| Priority Rank | Dataset Name | Primary Modality & Sensor | Native Image Format | Annotation Level | Primary Research / Operational Role | License / Access |
|---|---|---|---|---|---|---|
| **1** | **LEVIR-CD** | Optical / Google Earth ($0.5\text{ m}$) | **PNG** ($1024\times 1024$) | Binary Building Masks | Building construction / demolition baseline | CC-BY-4.0 |
| **2** | **SYSU-CD** | Optical / Aerial & Satellite ($0.5\text{ m}$) | **PNG** ($256\times 256$) | Binary Change (Urban, Veg, Road) | Wide-area multi-class change generalization | Open Academic |
| **3** | **OSCD** | Multi-spectral Sentinel-2 ($10\text{ m}$) | **GeoTIFF** ($600\times 600$) | Pixel-level Binary Urban Change | Multi-spectral band change detection | Creative Commons |
| **4** | **LEVIR-CC / RSICC** | Optical / Google Earth ($0.5\text{ m}$) | **PNG** ($1024\times 1024$) | 5 Natural Language Captions / Pair | Change VQA & natural change description | Academic Open |
| **5** | **SECOND** | Optical / Aerial Multi-platform ($0.5 - 2\text{ m}$) | **PNG** ($512\times 512$) | 6-Class Semantic Transition Masks | Semantic change understanding (`from -> to`) | Academic Open |
| **6** | **TERRA-CD / DynamicEarthNet** | Multi-spectral PlanetScope ($3\text{ m}$) | **GeoTIFF** ($1024\times 1024$) | Daily Multi-class Semantic Changes | Phenology vs physical change discernment | Open Research |
| **7** | **Landsat-SCD** | Multi-spectral Landsat ($30\text{ m}$) | **GeoTIFF** ($416\times 416$) | 10-Class Land Cover Changes | Regional macroscopic environmental change | Academic Open |
| **8** | **HRSCD** | Optical Orthoimagery ($0.5\text{ m}$) | **GeoTIFF** / **TIFF** ($10,000\times 10,000$) | 5 Land-Cover Transition Masks | High-resolution regional change verification | Open Access (IGN) |
| **9** | **MSOSCD / MultiModalOSCD** | Optical + SAR (Sentinel-1/2, $10\text{ m}$) | **GeoTIFF** ($600\times 600$) | Multi-sensor Binary Change | Cross-sensor optical/SAR temporal change | Open Academic |
| **10** | **Hi-UCD** | Ultra-High-Res Orthoimagery ($0.1\text{ m}$) | **TIFF** / **PNG** ($1024\times 1024$) | 9-Class Urban Semantic Change | Fine-grained structural change attribution | Academic Open |

---

## 2. Format Inventory Breakdown

To ensure SatQuery's pipeline learns **visual and physical content rather than format idiosyncrasies**, the training and benchmark corpora are systematically partitioned across formats:

### 2.1 Format Group A: PNG Datasets
- **LEVIR-CD:** 637 pairs ($1024\times 1024$), 24-bit RGB PNG. Clean lossless overhead imagery.
- **SYSU-CD:** 20,000 pairs ($256\times 256$), 24-bit RGB PNG. High-density varied land-cover change.
- **SECOND:** 4,662 pairs ($512\times 512$), 24-bit RGB PNG with 8-bit PNG semantic masks.
- **LEVIR-CC:** 10,077 pairs ($1024\times 1024$), 24-bit RGB PNG with descriptive JSON captions.

### 2.2 Format Group B: JPEG Datasets
- **Diverse RS JPEG Robustness Subset:** Standard baseline crops encoded at controlled JPEG compression factors ($Q=70, 75, 85, 95$).
- **PatternNet & DIOR Temporal Crops:** 8-bit RGB JPEG imagery evaluating compression-artifact resilience.
- **Controlled Distortion Benchmark:** High-frequency DCT blockiness injection to teach models that compression discrepancies $\neq$ surface change.

### 2.3 Format Group C: Unprojected TIFF Datasets
- **WHU Building Dataset (TIFF subset):** Uncompressed 24-bit TIFF tiles lacking embedded projection headers.
- **Hi-UCD Ortho TIFFs:** High-resolution unprojected TIFF pairs with local pixel coordinate grids.

### 2.4 Format Group D: Georeferenced GeoTIFF Datasets
- **OSCD (Onera Satellite CD):** 24 multi-band GeoTIFF scenes (Sentinel-2 Level-1C/2A) with EPSG projected coordinates.
- **Diverse RS Benchmark (M4/M5):** 12 multi-spectral GeoTIFF pairs with UTM coordinates and calibrated pixel-ground dimensions.
- **ISRO Bhoonidhi Time-Series (M11):** Cartosat-2, ResourceSat-2A, and EOS-04 georeferenced rasters.

### 2.5 Format Group E: Multi-Spectral & SAR Datasets
- **Sentinel-2 L2A (12 Bands):** Blue, Green, Red, RedEdge, NIR, SWIR1, SWIR2 in 16-bit GeoTIFF.
- **Sentinel-1 SAR GRD:** Dual-pol (VV + VH) backscatter amplitude in 32-bit floating point GeoTIFF.
- **SEN12MS & MultiModalOSCD:** Spatially aligned optical-SAR bi-temporal cubes.

---

## 3. Format Normalization Policy

The system introduces `backend/preprocessing/bi_temporal_normalizer.py` implementing two distinct processing regimes:

```
                  ┌───────────────────────────────┐
                  │ Input Pair (T1, T2)           │
                  │ PNG, JPEG, TIFF, or GeoTIFF   │
                  └──────────────┬────────────────┘
                                 │
                     Inspect Spatial Headers
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [Mode A: Georeferenced]        [Mode B: Non-Georeferenced]
       - Valid CRS & Transform        - PNG / JPEG / Local TIFF
       - Reprojection & Warping       - Phase Correlation Alignment
       - Common Intersection Crop     - Shift Offset Estimation
       - Ground Area ($m^2$, ha)      - Alignment Score [0.0, 1.0]
       - Geospatial Polygons          - Pixel Area & Percentages
                                      - Explicit Limitation Notice
```

Both modes output a unified internal structure (`NormalizedPair`) consumed transparently by TinyCD, CVA, and Qwen2-VL.
