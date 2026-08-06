# AGENTS.md — Workspace Architecture & Project Guide

## 📌 Repository Overview

This repository is **mp-factory** (Massive Processing Factory), an HPC-scale medical
image processing and deep learning factory built around the **CancerVerse** and
**BodyMaps/AbdomenAtlas** CT datasets. On the cluster it lives at
`/mnt/scratch/user/chrsong/mp-factory`.

The project delivers **Project 9 (GI Organ Segmentation)** of the wider research plan
via a *fully-automated data-cleansing and weakly-supervised distillation* pipeline
(see `ENSEMBLE_FIX_PLAN.md`). It is designed to be "under-annotation-tolerant": rather
than manually correcting noisy masks, it mathematically filters ~22,000 scans and
distills clean pseudo-labels into a student model.

The codebase supports:
1. **Data Ingestion & HPC Downloading** — pull and manage large-scale 3D CT datasets from Hugging Face.
2. **Automated 3D Organ Segmentation** — scalable TotalSegmentator inference plus trained MedNeXt-B and Swin-UNETR GI segmenters over `BDMAP_*` and `CV_*` cohorts.
3. **Multi-Model Ensemble & Automated Cleansing** — consensus assembly, spatial uncertainty, topological/metric auditing (Dice, HD95, Betti-0, ARI, VOI, ECE), and triage into CLEAN / WEAK / REJECT buckets.
4. **Teacher–Student Distillation (Phase 2)** — train MedNeXt-B on consensus pseudo-ground-truth with an `ignore_index` for conflicting boundary pixels.
5. **Expert Validation** — GPU-accelerated evaluation against JHU radiologist-corrected ground-truth masks.
6. **Synthetic Tumor Synthesis** — `SyntheticTumors` submodule for procedural liver/pancreatic tumor generation and 3D model training.

The 5 GI organ classes throughout are: `0` background, `1` stomach, `2` duodenum,
`3` small bowel / intestine, `4` colon.

---

## 📂 Codebase Directory Layout

```
mp-factory/
├── AGENTS.md                       # This guide
├── README.md                       # User-facing overview
├── ENSEMBLE_FIX_PLAN.md            # Automated-cleansing & distillation design doc
├── requirements.txt / setup.py     # Python packaging (environment.yml for conda)
├── CancerVerse/                    # Dataset repo, metadata CSV, HF download specs (data gitignored)
├── JHU_data_radiologist_corrected/ # Expert radiologist GT (*.seg.nrrd) + clinical feedback notes
├── code/
│   ├── active_learning/            # loop.py: UncertaintyEstimator, ActiveLearningPool, Orchestrator
│   ├── data_download/              # download.sbatch (Hugging Face pull)
│   ├── evaluation/                 # Auditing, ensemble cleansing, and verification
│   │   ├── evaluate_gi_masks.py            # GT-vs-TotalSeg GI audit (Dice/HD95/Betti-0)
│   │   ├── evaluate_multi_model_ensemble.py# Consensus + uncertainty + triage cleansing (669 LOC)
│   │   ├── evaluate_jhu_radiologist_gpu.py # GPU eval vs JHU radiologist GT (.nrrd)
│   │   ├── evaluate_jhu_radiologist_set.py # CPU eval + phase-propagation rules
│   │   ├── verify_predictions.py           # Sanity-check prediction label content
│   │   ├── verify_pipeline_run.py          # Confirm pipeline output deliverables
│   │   └── watch_pipeline.py               # Live SLURM progress monitor
│   ├── prediction/                 # Batch inference
│   │   ├── predict_mednext.py              # MedNeXt checkpoint → 5-class GI masks
│   │   ├── predict_swin_unetr.py           # Swin-UNETR checkpoint → 5-class GI masks
│   │   └── run_totalsegmentator_batch.py   # Standalone TotalSegmentator batch runner
│   ├── slurm_scripts/              # All *.sbatch job configs + TotalSeg array runners
│   │   ├── run_totalseg*.py/.sbatch        # TotalSegmentator CV/BDMAP/GI array jobs
│   │   ├── run_mednext{,_phase2}.sbatch    # MedNeXt training (4-fold arrays)
│   │   ├── run_swin_unetr.sbatch           # Swin-UNETR training
│   │   ├── run_predict_{mednext,swin_unetr}.sbatch
│   │   ├── run_evaluation.sbatch / run_ensemble_eval.sbatch
│   │   ├── run_compare_models.sbatch / run_jhu_radiologist_eval_gpu.sbatch
│   │   ├── submit_audit_phase1.sbatch / smoke_test.sbatch
│   ├── training/                   # Model training
│   │   ├── train_mednext.py                # Phase 1 MedNeXt-B (real/coarse labels)
│   │   ├── train_mednext_phase2.py         # Phase 2 student on consensus pseudo-GT (ignore_index=255)
│   │   └── train_swin_unetr.py             # Swin-UNETR GI segmenter
│   ├── utilities/                  # Analysis, comparison, and tests
│   │   ├── audit_phase1_gi.py / analyze_audit_results.py
│   │   ├── compare_models.py               # MedNeXt vs Swin-UNETR per-organ Dice
│   │   ├── merge_ensemble_chunks.py        # Merge sharded ensemble CSVs
│   │   ├── run_active_learning.py          # Radiologist query-queue builder
│   │   ├── submit_audit.sh                 # Legacy audit launcher
│   │   └── test_active_learning.py / test_ensemble_evaluation.py
│   └── SyntheticTumors/            # Procedural tumor generator & trainer (CVPR 2023, gitignored submodule)
├── graphify-out/                   # Knowledge-graph artifacts (GRAPH_REPORT.md, graph.json/html)
├── logs/                           # SLURM *.out / *.err (gitignored)
└── results/                        # Masks, consensus, and CSV summaries (gitignored)
```

> **Note on paths:** `logs/`, `results/`, `CancerVerse/` data, `*.nii.gz`, `envs/`,
> and `SyntheticTumors/` are gitignored (see `.gitignore`). Only code and docs are tracked.

---

## 🛠️ Key Scripts & Execution Pipelines

The end-to-end flow is: **download → segment → ensemble-cleanse → distill → validate.**

### 1. Data Ingestion & Management
- **`code/data_download/download.sbatch`** — SLURM job that downloads `BodyMaps/CancerVerse` via `hf download`.
- **`CancerVerse/download_cancerverse.sh`** — shell utility for dataset retrieval.

### 2. Automated 3D Segmentation
- **`code/prediction/run_totalsegmentator_batch.py`** — standalone runner scanning `BDMAP_*` dirs; supports `--fast` and `--task`.
- **`code/slurm_scripts/run_totalseg.py`** — multi-label (`--ml`) TotalSegmentator on `CV_*` scans.
- **`code/slurm_scripts/run_totalseg_bdmap.py`** — chunked parallel segmentation of `BDMAP_*` cases.
- **`code/slurm_scripts/run_totalseg_gi_array.py`** — SLURM array worker for GI organs with per-scan `torch.cuda.empty_cache()` + `gc.collect()`.
- **`code/prediction/predict_mednext.py` / `predict_swin_unetr.py`** — sliding-window inference (96³ ROI) from trained checkpoints; shard across a 4-worker SLURM array via `--array_id/--total_workers`.

### 3. Multi-Model Ensemble & Automated Data Cleansing
- **`code/evaluation/evaluate_multi_model_ensemble.py`** (core engine, 669 LOC):
  - Loads masks/softmax maps from N models (TotalSeg, MedNeXt, Swin-UNETR, optional DeepGI).
  - Diversity-promoting weighting via a pairwise inter-model Dice matrix.
  - Voxel-wise spatial predictive uncertainty (entropy/variance) heatmaps.
  - Assembles a weighted consensus pseudo-GT mask `<subject_id>_consensus.nii.gz`.
  - Metrics: Dice, IoU, HD95, Betti-0 diff (|Δβ0|), ARI, VOI, ECE.
  - **Triage (strict automated data-cleansing; see `triage_case` + `NOISE_*`/`CLEAN_*` constants):**
    - `NOISE_REJECT` — `|Δβ0| > 5` **or** consensus Dice < 0.50 **or** uncertainty > 0.15 → discard from training pool (any hard gate wins).
    - `CLEAN_HIGH_CONFIDENCE` — consensus Dice ≥ 0.82 **and** inter-model Dice ≥ 0.85 **and** `|Δβ0| ≤ 2` → auto-approve for GKD/VAE.
    - `WEAK_COARSE` — otherwise → hard-threshold conflicting pixels to the ignore class.
  - **Dataset splits:** `export_dataset_splits()` writes `dataset_splits/` next to the CSV — per-bucket `ensemble_split_<cat>.txt/.csv`, a combined `ensemble_split_train_pool.txt` (CLEAN+WEAK), and `dataset_splits.json` (thresholds + counts). Override the location with `--splits_dir`.
  - Supports `--num_chunks/--chunk_idx` sharding; merge with `merge_ensemble_chunks.py` (which re-exports splits from the merged CSV).
- **`code/evaluation/hard_threshold_autolabel.py`** — Task A.2: hard-threshold auto-labeling for `WEAK_COARSE` subjects. Compares human coarse labels (`gi_mask.nii.gz` or `segmentations/`) with consensus pseudo-GT; resamples consensus to the human grid; sets conflicting voxels (human ≠ consensus) to `IGNORE_INDEX=255` so `DiceCEWithIgnoreLoss` in Phase-2 training skips them. Writes `<subject>_autolabel.nii.gz` + a stats CSV. `IGNORE_INDEX=255` (not -1) matches `train_mednext_phase2.py` and uint8 NIfTI storage.
- **`code/evaluation/unified_gastro_dataset.py`** — Task A.3: unified PyTorch `Dataset` + manifest builder. `build_manifest()` discovers CLEAN (consensus labels), WEAK (autolabel with 255 ignore), and TEACHER (TotalSegmentator pseudo-labels) subjects from the audit CSV + data dirs. `verify_dataset_integrity()` checks every CT/label NIfTI pair for missing paths, corrupt headers, and shape mismatches so zero broken volumes enter the training pipeline.
- **`code/utilities/audit_phase1_gi.py` + `analyze_audit_results.py`** — Phase-1 GT-vs-consensus audit and report.
- **`code/evaluation/evaluate_gi_masks.py`** — multi-processed GT-vs-TotalSeg GI audit → `results/audit_summary.csv`.

### 4. Model Training (Phase 1 & Phase 2 Distillation)
- **`code/training/train_mednext.py`** — Phase-1 MedNeXt-B (variants S/B/M/L, kernel 3 or 5), MONAI `DiceCELoss`, 4-fold arrays. Default data: `CancerVerse_dbox`.
- **`code/training/train_swin_unetr.py`** — Swin-UNETR GI segmenter (same data pipeline / folds).
- **`code/training/train_mednext_phase2.py`** — Phase-2 student trained on ensemble consensus pseudo-GT selected from `ensemble_audit_summary.csv` (CLEAN_HIGH_CONFIDENCE, optionally WEAK_COARSE); `DiceCE` with `ignore_index=255`, cosine-annealing LR, optional Phase-1 checkpoint finetuning.

### 5. Comparison & Expert Validation
- **`code/utilities/compare_models.py`** — per-organ + mean Dice, MedNeXt vs Swin-UNETR, on held-out GT subjects → `results/model_comparison.csv`.
- **`code/evaluation/evaluate_jhu_radiologist_gpu.py`** — full CUDA metrics (Dice/IoU/precision/sensitivity/specificity, HD95, |Δβ0|) for MedNeXt, Swin-UNETR, TotalSeg, and EnsembleConsensus vs JHU radiologist GT (`.seg.nrrd`), applying multi-phase mask-propagation rules from `radiologist_clinical_feedback.md`.
- **`code/evaluation/evaluate_jhu_radiologist_set.py`** — CPU counterpart with the same propagation matrix (e.g. `113→113/115/116`, `131→131/132/133`, `136→135/136`).

### 6. Monitoring & Active Learning
- **`code/evaluation/watch_pipeline.py` / `verify_pipeline_run.py`** — live progress and deliverable verification (target 1,735 scans).
- **`code/active_learning/loop.py`** — `UncertaintyEstimator`, `ActiveLearningPool`, `ActiveLearningOrchestrator` (retained as an optional human-in-the-loop path; the primary pipeline is now fully automated).
- **`code/utilities/run_active_learning.py`** — builds a radiologist query queue JSON from the audit CSV.

### 7. Synthetic Tumor Submodule (`code/SyntheticTumors/`)
- **`main.py`** — trains 3D backbones (UNet, Swin-UNETR v1/v2, ViT) on synthetic (`--syn`) vs real tumors.
- **`TumorGenerated/`** — procedural 3D tumor generation and texture synthesis.
- **`datafolds/`** — dataset-split JSON configs.

---

## ⚙️ Configuration & Environment Setup

### Environment Activation (HPC)
```bash
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory
```
Python 3.11 with PyTorch, MONAI, TotalSegmentator, nnU-Net MedNeXt (`nnunet_mednext`),
`nibabel`, `pynrrd`, `scikit-image`, `scikit-learn`. See `requirements.txt`.

### Data & Output Locations (cluster)
- **Input CT scans:**
  - Training/inference default: `/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox/<subject>/ct.nii.gz`
  - BDMAP cohort: `/mnt/scratch/user/chrsong/CancerVerse_data/BDMAP_XXXX/ct.nii.gz`
  - CV cohort: `/mnt/scratch/user/chrsong/mp-factory/CancerVerse/CancerVerse/CV_XXXX/ct.nii.gz`
- **Outputs:** `results/` (mednext_predictions, swin_unetr_predictions, totalseg_gi_masks*, ensemble_out, `*_consensus.nii.gz`, audit CSVs); logs in `logs/`.

---

## 🔁 Typical End-to-End Run

```bash
# 1. Download data
sbatch code/data_download/download.sbatch

# 2. Baseline segmentation (TotalSegmentator, BDMAP array)
sbatch code/slurm_scripts/run_totalseg_bdmap.sbatch

# 3. Train deep models (4-fold arrays)
sbatch code/slurm_scripts/run_mednext.sbatch
sbatch code/slurm_scripts/run_swin_unetr.sbatch

# 4. Batch inference
sbatch code/slurm_scripts/run_predict_mednext.sbatch
sbatch code/slurm_scripts/run_predict_swin_unetr.sbatch

# 5. Ensemble consensus + automated cleansing
sbatch code/slurm_scripts/run_evaluation.sbatch      # or run_ensemble_eval.sbatch (GPU)
python code/utilities/merge_ensemble_chunks.py \
  --pattern "results/ensemble_audit_summary_*.csv" \
  --out_csv "results/ensemble_audit_summary.csv"

# 6. Phase-2 distillation on consensus pseudo-GT
sbatch code/slurm_scripts/run_mednext_phase2.sbatch

# 7. Validate vs expert radiologist GT
sbatch code/slurm_scripts/run_jhu_radiologist_eval_gpu.sbatch
```

---

## 📝 Conventions for Agents

- **Git:** this directory is the git root (`origin: chrsong-ucsf/mp-factory`). Commit code/doc changes; never commit data, `results/`, `logs/`, or `*.nii.gz` (already gitignored).
- **Paths:** scripts hardcode cluster paths under `/mnt/scratch/user/chrsong/mp-factory`; when editing locally, keep those defaults unless the task says otherwise.
- **SLURM ↔ script coupling:** `slurm_scripts/*.sbatch` reference scripts by absolute path. After the subdirectory reorg, **12 of the 13** `.sbatch` jobs still point at the old flat `code/<script>.py` location (only `run_jhu_radiologist_eval_gpu.sbatch` was updated to the new `code/evaluation/...` path). Fix the `python -u ...` target to the correct subdirectory (e.g. `code/training/train_mednext.py`, `code/evaluation/evaluate_multi_model_ensemble.py`) before submitting any job.
- **Organ label map is fixed** (`0..4` above); keep it consistent across training, prediction, and evaluation.
- **Knowledge graph:** `graphify-out/GRAPH_REPORT.md` is a generated map of modules/relationships; regenerate rather than hand-edit.
