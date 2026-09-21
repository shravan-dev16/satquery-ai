# SatQuery AI — 48-Hour Model Expansion Dataset Inventory & Integrity Audit

**Audit Timestamp:** 2026-09-20T19:30:00+05:30  
**Integrity Status:** PASSED (0 Violations)

## 1. Summary of Acquired Datasets

| Dataset | Tasks Supported | Usable Samples / Pairs | Format | Channels | CRS Availability | Labels | License |
|---|---|---|---|---|---|---|---|
| **RSVQA-LR** | Single-Image Remote-Sensing VQA | 2000 | PNG | 3 | No | Natural Language Answers | CC BY 4.0 |
| **CDVQA** | Bi-Temporal Change VQA | 200 | PNG | 3 | No | Change QA Strings | Academic / Non-Commercial Research |
| **OSCD** | Bi-Temporal Change Detection & Hard Negatives | 103 | PNG | 3 | No | Binary Change Mask (0/255 uint8) | Open Access (IEEE DataPort open tier) |
| **VRSBench** | Remote-Sensing VQA & Overhead Object Understanding | 92 | PNG | 3 | No | Object Categories, Counts, and Presence | CC BY 4.0 |
| **LEVIR-CD** | Bi-Temporal Building Change Detection | 20 | PNG | 3 | No | Pixel-Level Building Change Mask | CC BY 4.0 |
| **Geospatial_GeoTIFF** | Geospatial Grounding, CRS Coordinate Projection & Physical Area Calculation | 47 | GeoTIFF (16-bit / 8-bit multi-band) | 3 | Yes | Deterministic Affine Geotransform & Extent Bounds | Open Remote Sensing / Bhoonidhi Open Tier |
| **Change_Robustness_Hard_Negatives** | Nuisance Invariance (Illumination, Seasonal, Shadow, Registration) & Hard Negatives | 10 | PNG / GeoTIFF | 3 | Yes | Zero-Change & Real-Change Binary Labels | SatQuery Project Curated |
| **BigEarthNet_MM** | Multimodal Cross-Modal Reasoning & Corine Land Cover Semantics | 500 | PNG | 2-band SAR (VV/VH) + 12-band Optical | Yes | Multi-Label Corine Land Cover (43 / 19 classes) | CDLA-Permissive-1.0 |

## 2. Integrity & Leakage Controls

- **Frozen Test Parent Scenes:** `['P1225', 'P2912', 'P2982', 'P4055', 'P4265', 'P4627', 'diagnostic_agriculture', 'diverse_agriculture_01', 'diverse_nuisance_registration_01', 'diverse_water_01', 'levir_val_18', 'levir_val_19', 'levir_val_20']`
- **Benchmark Leakage Violations:** 0
- **Corrupted Files Detected:** 0
- **Verified GeoTIFFs with UTM CRS:** 47

## 3. Manual Action Requirements

- **OSCD Full 13-Band Archives:** IEEE DataPort account required. Handled via open RGB mirror + local 13-band GeoTIFF fixtures.
- **SYSU-CD Full Archive:** Baidu Netdisk app / Chinese phone number required. Core suburban & vegetation change covered via LEVIR-CD + DiverseRS.
- **SpaceNet AWS Requester Pays:** Requires billing AWS credentials. Handled via local verified UTM GeoTIFFs (EPSG:32643 / EPSG:32618).
