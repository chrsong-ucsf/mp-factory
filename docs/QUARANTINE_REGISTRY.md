# QUARANTINE REGISTRY & EVIDENCE WITHDRAWAL DOSSIER

**Registry Version:** `QUARANTINE-v1.0`  
**Last Updated:** `2026-11-08`  
**Committed Location:** `docs/quarantine_registry.json`

## Purpose
This registry documents and tracks historical research outputs, summary metrics, and data archives that have been formally invalidated by forensic code/data audits. Quarantined artifacts are strictly excluded from all manuscript tables, production claims, and downstream pipeline consumption.

---

## Quarantined Artifact Inventory

### 1. `QR-2026-001`: Historical Ensemble Phase 3 Audit Summary
- **Artifact Path:** `results/ensemble_phase3_audit_summary.csv`
- **Generating Commit:** `b3ff512`
- **Quarantine Status:** `KNOWN_INVALID`
- **Defect Classification:** `FALSE_BACKGROUND_SUPERVISION_AND_AFFINE_MISMATCH`
- **Invalidation Evidence:**
  The ensemble consensus engine previously failed to read `support_mask`, injecting 100% background votes in areas outside individual model coverage. Additionally, legacy probability grids inherited segmentation affines based solely on equal voxel dimensions. This distorted predictive uncertainty and caused 1,598 false `NOISE_REJECT` classifications out of 1,632 evaluated scans.
- **Affected Reports:** `docs/SHOWCASE.md` Table 2, `results/ensemble_audit_summary.csv`.
- **Replacement:** Prospective re-ensemble rerun incorporating per-voxel active-support weight normalization (`ENS-SUPPORT-v1.0`).

### 2. `QR-2026-002`: Unverified Multiphase Scanning Pairs
- **Artifact Path:** `CancerVerse/multiphase_pairs.csv`
- **Generating Commit:** `historical_pre_b3ff512`
- **Quarantine Status:** `QUARANTINED_UNVERIFIED`
- **Defect Classification:** `UNVERIFIED_TEMPORAL_SESSION_PAIRING`
- **Invalidation Evidence:**
  1,545 candidate scanning pairs have both source (NCCT) and target (contrast) exam dates blank. Without matching accession numbers or valid examination dates, same-session anatomical identity cannot be established.
- **Affected Reports:** `multiphase_paired_dataset.py` Stage 2 diffusion training loader.
- **Replacement:** `discover_multiphase_cohort.py` running strict session matching (`REG-LANDMARK-v1.0`).

### 3. `QR-2026-003`: Historical Lesion Utility Benchmark Output
- **Artifact Path:** `results/lesion_augmentation_benchmark_summary.json`
- **Generating Commit:** `historical_pre_b3ff512`
- **Quarantine Status:** `KNOWN_INVALID`
- **Defect Classification:** `GAUSSIAN_JITTER_FABRICATION_AND_UNMATCHED_BUDGET`
- **Invalidation Evidence:**
  When synthetic feature columns were absent, the benchmark silently substituted `X_train_real + Normal(0, 5)` or `Normal(0, 8)` and copied real training labels (`y_syn = y_train_real`). Sample sizes were unmatched ($N$ vs $2N$).
- **Affected Reports:** `docs/SHOWCASE.md` Section 5.
- **Replacement:** Protocol `P10-EXP-v1.0` (12 physical measurements, 1:1 matched instances, multi-seed runs).
