# Project Architectural Notes: Image Synthesis, Data Cleansing, & Model Distillation

> [!NOTE]
> **Document Purpose:** This document contains high-level technical & architectural design notes (MedNeXt-B, distillation, multi-phase synthesis).  
> For actual scan-by-scan radiologist annotations, clinical error analysis (vessels, rectum/esophagus, tumor lesions), and mask propagation rules for `BDMAP_*` scans, refer to:
> - 📄 [radiologist_clinical_feedback.md](file:///Users/chrissong/research_su26/03_Datasets/JHU_data_radiologist_corrected/radiologist_clinical_feedback.md) (Master Clinical & Reusability Guide)
> - 📄 [notes.md](file:///Users/chrissong/research_su26/03_Datasets/JHU_data_radiologist_corrected/notes.md) (Mask Reuse Rules)

**Context:** These notes outline the technical bridge between **Project 9 (GI Tract Segmentation)** and **Project 10 (Multi-Phase Image Synthesis)**, specifically detailing the MedNeXt-B training pipeline, TotalSegmentator distillation, and topological data cleansing over the CancerVerse dataset.


---

## Note 1: Multi-Phase Synthesis & Alignment Strategy

### Translated Text
**Core Objective:** Establish structural invariance across sequential clinical imaging phases.
* **Phase-Invariant Segmentation Requirement:**
  * The downstream generative model (e.g., Project 10 Anatomy-Conditional VAE) requires robust organ masks that remain invariant across variable contrast phases (e.g., non-contrast, arterial hyperenhancement, portal-venous washout).
  * If the underlying segmentation model lacks multi-phase robustness, phase-consistency validation metrics will inherit extreme noise.
* **Action Item Matrix:**
  1. Finalize the 5-organ GI segmenter (Stomach, Duodenum, Jejunum, Ileum, Colon) as the core anatomical constraint.
  2. Enforce strict intensity clipping ($1^{\text{st}}$ to $99^{\text{th}}$ percentile) to cap scanner gain variations without destroying high-frequency diagnostic boundaries.
  3. Integrate physics-constrained normalization layers to handle multi-vendor and multi-scanner domain shifts.

### 🔗 Project Context & Dependencies
* **Dependency on Project 9 (Arm A):** As noted in the *Segmentation Model's Role in Project 10* memo, Project 10 is a "plug-in synthetic-data factory." We cannot condition a generative model for GI organs without the rock-solid 5-organ GI boundaries established in Project 9. 
* **The "Turing Test" for Consistency:** If our 5-organ segmenter fails on an arterial phase but succeeds on a non-contrast phase, our downstream evaluation loop will falsely penalize the Phase 1 VAE Generator. Fixing phase-invariance (Action Item 1) is a hard blocker for Project 10.
* **Harmonization Alignment:** Action Item 3 aligns directly with the "PhyCHarm-style" methods referenced in Project 7, ensuring the latent space remains vendor-agnostic.

---

## Note 2: Automated Data Cleansing & Weakly-Supervised Pipeline

### Translated Text
**Core Objective:** Pivot from human-in-the-loop Active Learning queues to a fully automated data-cleansing pipeline under massive data scales ($22,000+$ raw scans).
* **Automated Triage & Rejection Logic:**
  * Eliminate manual radiologist correction loops for edge cases.
  * **Hard Exclusion Threshold:** Cases exceeding topological discrepancy thresholds ($\Delta\beta_0 > 5$) or high predictive uncertainty are automatically dropped (`NOISE_REJECT`) to protect the training pool from label poisoning.
* **Weakly-Supervised Label Mechanics:**
  * **Coarse Human Annotations:** Confusing pixels where external human labels conflict with TotalSegmentator consensus are assigned an **`ignore` class** in the loss function mask.
  * **Asymmetric Loss Integration:** Train the MedNeXt-B student model exclusively using `AsymmetricPDCELoss` (Asymmetric Partial Cross-Entropy + Dice Loss) to maximize recall on sparse tissue regions while clamping probabilities at $1\times 10^{-7}$ to avoid `float32` NaN overflows.

### 🔗 Project Context & Dependencies
* **CancerVerse Scale:** The $22,000+$ raw scans refer to the massive unannotated CancerVerse dataset.
* **Betti-0 Rejection Gate:** The $\Delta\beta_0 > 5$ rule is the topological fragmentation check explicitly documented in the *Data Validation and Model Training Strategy*. Scans failing this check must be routed via `evaluate_multi_model_ensemble.py` straight to the Reject/Noise bucket.
* **Ignore Class Implementation:** When auditing the 1,700 `CancerVerse_data` pre-annotated scans against TotalSegmentator, we cannot trust either blindly. The `ignore` class strategy safely bridges this gap for the MedNeXt-B training loop.

---

## Note 3: Model Distillation & Architecture Specifications

### Translated Text
**Core Objective:** Operationalize Teacher-Student Knowledge Distillation and architectural setups for high-performance volumetric segmentation.
* **Teacher-Student Setup:**
  * **Teacher Model:** Pre-trained TotalSegmentator weights serving as the stable structural reference.
  * **Student Model:** MedNeXt-B backbone optimized via Generalizable Knowledge Distillation (GKD).
* **Execution Parameters:**
  * Request high-performance cluster allocation (e.g., `nvidia_l40s` GPU nodes) to handle full 3D volumetric tensor operations.
  * Run parallel scaling sweeps ($N = 4$ to $50$ subsets) using `run_scaling_sweep.py` to plot performance-degradation curves under varying noise and data volumes.
  * Integrate advanced evaluation metrics: **HD95** (95% Hausdorff Distance) for organ boundaries and **FROC** (Free-response ROC) for tumor/lesion detection validation.

### 🔗 Project Context & Dependencies
* **GKD (Generalizable Knowledge Distillation):** This fulfills the Phase 2 Master Research Plan goal. By using TotalSegmentator as the teacher on the 22K unlabeled scans, MedNeXt-B learns domain-invariant boundaries from soft feature maps rather than noisy, hard external labels.
* **GPU Compute:** The explicit note about `nvidia_l40s` GPUs reflects recent repo commits in `mp-factory/code` requesting L40s nodes for full CUDA support on 3D MedNeXt-B tensors.
* **HD95 Anisotropy Fix Required:** As noted in the Data Validation audit, standard `distance_transform_edt` fails on anisotropic medical volumes. The implementation of HD95 here *must* use the 3D morphological erosion (`scipy.ndimage.binary_erosion`) fix to prevent evaluation crashes.