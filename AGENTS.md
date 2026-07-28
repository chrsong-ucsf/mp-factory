# AGENTS.md — Workspace Architecture & Project Guide

## 📌 Repository Overview

This workspace (`/mnt/scratch/user/chrsong`) hosts **mp-factory** (Massive Processing Factory), an HPC-scale medical image processing and deep learning factory built around the **CancerVerse** and **BodyMaps** datasets. 

The codebase supports:
1. **Data Ingestion & HPC Downloading**: Pipeline scripts to pull and manage large-scale 3D CT scan datasets.
2. **Automated 3D Organ Segmentation**: Scalable TotalSegmentator inference on full-body CT volumes (`BDMAP_*` and `CV_*` cohorts).
3. **Topological & Metric Auditing**: Parallel evaluation scripts computing Dice coefficient, 95th-percentile Hausdorff Distance (HD95), and Betti-0 connected component counts across organ masks.
4. **Synthetic Tumor Synthesis & Model Training**: Submodule (`SyntheticTumors`) for generating synthetic liver/pancreatic tumors and training 3D vision models (UNet, Swin-UNETR, ViT).

---

## 📂 Codebase Directory Layout

```
/mnt/scratch/user/chrsong/
├── AGENTS.md                          # Repository guide for AI agents & developers
├── CancerVerse/                       # Local Hugging Face download target directory
├── CancerVerse_data/                  # Primary CT scan dataset (BDMAP_XXXX/ct.nii.gz)
├── envs/                              # Virtual environment store
│   └── mp-factory/                    # Dedicated Python 3.11 / PyTorch / MONAI environment
├── mp-factory/                        # Core factory project directory
│   ├── AGENTS.md                      # Copy of project reference guide
│   ├── CancerVerse/                   # Dataset repository, metadata CSV, and Hugging Face specs
│   │   ├── CancerVerse_dataset_metadata.csv
│   │   ├── README.md
│   │   └── download_cancerverse.sh
│   ├── code/                          # Core scripts, SLURM batch configs, and SynTumor module
│   │   ├── SyntheticTumors/           # Synthetic tumor generator & model trainer (CVPR 2023)
│   │   ├── download.sbatch            # Hugging Face download job script
│   │   ├── evaluate_gi_masks.py       # Quality & topological metric evaluation
│   │   ├── run_evaluation.sbatch      # SLURM trigger for evaluation
│   │   ├── run_totalseg.py            # TotalSegmentator runner for CV_* scans
│   │   ├── run_totalseg.sbatch        # SLURM script for run_totalseg.py
│   │   ├── run_totalseg_bdmap.py      # Chunked TotalSegmentator runner for BDMAP_* scans
│   │   ├── run_totalseg_bdmap.sbatch  # SLURM job array script for BDMAP segmentation
│   │   ├── run_totalseg_gi_array.py   # Array job worker for GI organ segmentation
│   │   ├── run_totalseg_gi_array.sbatch # SLURM job array for GI mask generation
│   │   ├── smoke_test.sbatch          # Environment & GPU verification script
│   │   └── submit_audit.sh            # Launcher script for audit pipeline
│   ├── logs/                          # SLURM standard output & error logs (*.out, *.err)
│   └── results/                       # Output masks and CSV summary reports
│       ├── audit_summary.csv          # Evaluation output from evaluate_gi_masks.py
│       ├── totalseg_gi_masks/         # Output GI masks (CV_* cohort)
│       ├── totalseg_gi_masks_bdmap/   # Output GI masks (BDMAP_* cohort)
│       └── totalseg_masks/            # Output multi-label total organ masks
└── run_totalsegmentator_batch.py      # Root-level standalone batch runner for BDMAP scans
```

---

## 🛠️ Key Scripts & Execution Pipelines

### 1. Data Ingestion & Management
- **`mp-factory/code/download.sbatch`**: SLURM batch job for downloading the `BodyMaps/CancerVerse` dataset via Hugging Face Hub (`hf download`).
- **`mp-factory/CancerVerse/download_cancerverse.sh`**: Shell utility for dataset retrieval.

### 2. Automated 3D Segmentation Workflows
- **`run_totalsegmentator_batch.py`**:
  - *Location*: Root directory (`/mnt/scratch/user/chrsong/run_totalsegmentator_batch.py`)
  - *Purpose*: Standalone CLI script that scans all `BDMAP_*` directories under `CancerVerse_data/`, checks for existing segmentations, and runs `TotalSegmentator`. Supports `--fast` and `--task` flags.
- **`mp-factory/code/run_totalseg.py`**:
  - *Purpose*: Batch runner executing multi-label TotalSegmentator predictions (`--ml`) on `CV_*` scans under `mp-factory/CancerVerse/CancerVerse`.
- **`mp-factory/code/run_totalseg_bdmap.py`**:
  - *Purpose*: Chunk-based processor for parallel segmentation of `BDMAP_*` cases in `CancerVerse_data/`.
- **`mp-factory/code/run_totalseg_gi_array.py`**:
  - *Purpose*: Distributed SLURM array job worker targeting GI organs. Features automated GPU memory management (`torch.cuda.empty_cache()` and `gc.collect()` after each scan).

### 3. Quantitative Evaluation & Quality Audit
- **`mp-factory/code/evaluate_gi_masks.py`**:
  - *Purpose*: Multi-processed quantitative audit comparing ground-truth GI masks (`CancerVerse_data/new_database_masks`) against generated TotalSegmentator masks.
  - *Organs Audited*: Stomach (1), Duodenum (2), Small Bowel (3), Colon (4).
  - *Metrics Computed*:
    - **Dice Similarity Coefficient (DSC)**
    - **95th Percentile Hausdorff Distance (HD95)**
    - **Betti-0 Connected Component Count** (`skimage.measure.label`)
  - *Output*: `mp-factory/results/audit_summary.csv`
- **`mp-factory/code/submit_audit.sh`**:
  - *Purpose*: SLURM CPU job script allocating 32 CPU cores and 64GB RAM to execute `evaluate_gi_masks.py`.

### 4. Synthetic Tumor Submodule (`mp-factory/code/SyntheticTumors/`)
- **`main.py`**: CLI entry point for training 3D segmentation backbones (UNet, Swin-UNETR v1/v2, ViT) on synthetic tumor datasets (`--syn`) vs real tumor datasets.
- **`validation.py`**: Script for model evaluation across validation folds.
- **`TumorGenerated/`**: Core algorithms for procedural 3D tumor generation and texture synthesis (`TumorGenerated.py`, `utils.py`).
- **`networks/` & `networks2/`**: Architectures including `swin3d_unetr.py`, `swin3d_unetrv2.py`, `unetr.py`, `basicunetplusplus.py`, `vit.py`.
- **`datafolds/`**: JSON configuration files defining dataset splits (`healthy.json`, `lits.json`, `mix_*.json`, `real_*.json`).

---

## ⚙️ Configuration & Environment Setup

### Environment Modules & Path
The project runs within a managed Conda environment on the HPC cluster:
- **Environment Location**: `/mnt/scratch/user/chrsong/envs/mp-factory`
- **Activation Command**:
  ```bash
  module load CBI
  module load miniforge3/26.3.2-3
  eval "$(mamba shell hook --shell bash)"
  mamba activate /mnt/scratch/user/chrsong/envs/mp-factory
  ```

### Data & Output Locations
- **Input CT Scans**:
  - BDMAP Cohort: `/mnt/scratch/user/chrsong/CancerVerse_data/BDMAP_XXXX/ct.nii.gz`
  - CV Cohort: `/mnt/scratch/user/chrsong/mp-factory/CancerVerse/CancerVerse/CV_XXXX/ct.nii.gz`
- **Output Directories**:
  - Masks: `/mnt/scratch/user/chrsong/mp-factory/results/`
  - Logs: `/mnt/scratch/user/chrsong/mp-factory/logs/`
