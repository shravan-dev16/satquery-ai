# Change Detection Model Comparison & Benchmark Results

## 1. Executive Summary & Evaluation Protocol

As part of **Milestone M3 (Bi-Temporal Change Analysis Foundation)** for SatQuery AI (SIH26167 / ISRO), this document presents an empirical evaluation comparing:
1. **Candidate B — Deterministic Baseline:** Change Vector Analysis (**CVA**) with multi-channel Euclidean difference, adaptive thresholding ($T = \mu_D + 1.5\sigma_D$), and morphological speckle filtering.
2. **Candidate A — Learned Specialist:** **TinyCD** (~316k parameters), a lightweight Siamese convolutional neural network using an EfficientNet-B4 feature extractor with spatial-spectral mixing attention, pretrained on LEVIR-CD.

### Evaluation Dataset Integrity
- **Dataset:** LEVIR-CD (Large-scale Visible Image Remote Sensing Change Detection Dataset)
- **License:** CC-BY-4.0
- **Evaluation Split:** Held-out validation subset of 20 representative image pairs ($1024 \times 1024$ pixels, 3-band RGB at 0.5 m spatial resolution).
- **Manifest:** Documented deterministically in `docs/evaluation/levir_cd_subset.json`.
- **Integrity Assertions:** Every sample was verified to have matching $T_1$, $T_2$, and ground-truth binary masks with zero dimension or coordinate mismatch.
- **Rule 23 Compliance:** All metrics were computed using reproducible scientific code in `scripts/evaluate_change_detection.py` and saved to `docs/evaluation/change_detection_evaluation_results.json`. No metrics are fabricated or selectively filtered.

---

## 2. Quantitative Comparison Table

The benchmark was executed locally on the primary development workstation.

| Model / Candidate | Architecture / Method | Parameters | Mean IoU | Precision | Recall | F1 Score | False Pos Ratio | Empty Rate | Full-Img Rate | Mean Latency (ms) | Peak VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deterministic CVA Baseline** | Multi-spectral difference + adaptive std-dev threshold | N/A (Rule-based) | **0.0327** | 0.0913 | 0.0547 | **0.0619** | 0.0346 | 0.0% | 0.0% | **44.6 ms** | **0.0 MB** (CPU) |
| **TinyCD Specialist (Pretrained)** | Siamese EfficientNet-B4 + Mixing Attention | **~316k** | **0.8347** | **0.9216** | **0.8985** | **0.9091** | **0.0052** | 0.0% | 0.0% | **92.3 ms** | **219.4 MB** (CUDA) |
| **Empirical Delta (TinyCD vs CVA)** | — | — | **+0.8020** | **+0.8303** | **+0.8438** | **+0.8472** | **-0.0294** | 0.0% | 0.0% | +47.7 ms | +219.4 MB |

*Hardware: NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM), Intel Core i7, Windows 11, PyTorch 2.6.0+cu124, FP32 inference.*

---

## 3. Deep-Dive Analytical Findings

### 3.1 Why the CVA Baseline Fails on Optical Remote Sensing
The deterministic CVA method computes Euclidean distance across spectral channels. In real-world remote sensing imagery:
1. **Phenological & Illumination Variations:** Sun angle changes, seasonal foliage transitions, and atmospheric scattering alter pixel-level reflectances without indicating genuine structural or land-use changes.
2. **Diffuse False Positives:** CVA produces high background noise (average false-positive ratio of 3.46%, representing >36,000 erroneously flagged pixels per $1024 \times 1024$ tile).
3. **Threshold Sensitivity:** Even with adaptive thresholding ($T = \mu_D + 1.5\sigma_D$) and morphological opening/closing, CVA achieves a mean IoU of only **0.0327** and an F1 score of **0.0619**.

### 3.2 Why TinyCD Excels
1. **Deep Feature Abstraction:** Siamese EfficientNet-B4 extracts scale-invariant morphological and texture features rather than relying on raw spectral values.
2. **Mixing Attention:** Cross-temporal attention blocks compare feature representations across time steps, isolating genuine structural surface additions or demolitions from environmental variations.
3. **High Specificity & Localization:** TinyCD achieved **92.16% precision** and **89.85% recall** across the 20 held-out LEVIR-CD pairs, yielding a mean F1 of **0.9091** and a mean IoU of **0.8347**.
4. **Negligible False Positives:** False positive ratio dropped to **0.0052** (only 0.52% of non-changed pixels), ensuring clean, discrete bounding boxes for spatial evidence.

---

## 4. Hardware Budget & Operational Efficiency

| Metric | System Budget (RTX 4070 12 GB) | TinyCD Measured | Margin / Headroom |
| :--- | :--- | :--- | :--- |
| **Peak VRAM** | 7,500 MB (Working ceiling) | **219.4 MB** | **97.1% headroom available** |
| **Inference Latency** | $\le 1,500$ ms (Interactive target) | **92.3 ms** | **16x faster than target** |
| **Checkpoint Size** | $< 100$ MB | **1.28 MB** (`levir_best.pth`) | Negligible disk footprint |
| **Tile Resolution** | $1024 \times 1024$ native | Tiled $512 \times 512$ with zero downsampling | Full spatial resolution preserved |

---

## 5. Architectural Decision & Milestone Conclusion

Based on empirical evidence:
1. **Model Selection:** **TinyCD** is selected as the primary learned change detection engine registered under `CHANGE_DETECT` in `ModelRegistry`.
2. **Baseline Retention:** `DeterministicCVASpecialist` is retained in `backend/models/change.py` as an auditable fallback and baseline for testing and zero-parameter environments.
3. **Confidence Grounding:** Because TinyCD outputs uncalibrated sigmoid probabilities, the system computes the mean probability across detected change pixels and combines it with input quality and spatial registration scores in `ConfidenceBreakdown`, adhering to AGENTS.md Rule 11.
