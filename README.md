# mp-factory (Massive Processing Factory)

**mp-factory** is an HPC-scale medical image processing and deep learning factory built around the **CancerVerse** and **BodyMaps** datasets. It provides a highly scalable and automated pipeline for data ingestion, 3D organ segmentation, quality auditing, and synthetic tumor generation for 3D vision models.

## 🚀 Key Features

1. **Data Ingestion & HPC Downloading**
   - Automated pulling and management of large-scale 3D CT scan datasets (`BodyMaps`/`CancerVerse`) via Hugging Face Hub.
2. **Automated 3D Organ Segmentation**
   - Scalable, highly parallelized TotalSegmentator inference on full-body CT volumes across `BDMAP_*` and `CV_*` cohorts.
3. **Topological & Metric Auditing**
   - Parallel evaluation pipeline computing Dice Coefficient, 95th-percentile Hausdorff Distance (HD95), and Betti-0 connected component counts across generated organ masks.
4. **Synthetic Tumor Synthesis & Model Training**
   - Built-in submodule (`SyntheticTumors`) for procedural 3D tumor generation (liver/pancreatic) and model training for 3D vision backbones (UNet, Swin-UNETR, ViT).

---

## 📂 Repository Layout

```
mp-factory/
├── CancerVerse/                   # Dataset repository, metadata, and HF specs
├── code/                          # Core scripts, SLURM configs, and SynTumor module
│   ├── SyntheticTumors/           # Synthetic tumor generator & model trainer (CVPR 2023)
│   ├── download.sbatch            # HF download job script
│   ├── evaluate_gi_masks.py       # Quality & topological metric evaluation
│   ├── run_evaluation.sbatch      # SLURM trigger for evaluation
│   ├── run_totalseg*.py           # TotalSegmentator runners for CV and BDMAP scans
│   ├── run_totalseg*.sbatch       # SLURM scripts for batch processing
│   ├── smoke_test.sbatch          # Env & GPU verification
│   └── submit_audit.sh            # Launcher for audit pipeline
├── logs/                          # SLURM standard output & error logs
└── results/                       # Output masks and CSV summary reports
```
*(Note: A standalone CLI script, `run_totalsegmentator_batch.py`, resides at the workspace root to scan and process `BDMAP_*` cases.)*

---

## ⚙️ Environment & Setup

The project is built to run within a managed Conda environment on an HPC cluster. 

**Requirements:**
- Python 3.11
- PyTorch
- MONAI
- TotalSegmentator

**Environment Activation:**
```bash
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory
```

---

## 🛠️ Workflows & Execution

### 1. Data Ingestion
Use the provided SLURM batch job or shell utility to download the datasets:
```bash
sbatch code/download.sbatch
# OR
bash CancerVerse/download_cancerverse.sh
```

### 2. 3D Segmentation
To run multi-label total organ segmentation in parallel batches:
- For CV Cohort: `sbatch code/run_totalseg.sbatch`
- For BDMAP Cohort (Job Array): `sbatch code/run_totalseg_bdmap.sbatch`
- For GI Organ extraction (Array Worker): `sbatch code/run_totalseg_gi_array.sbatch`

You can also run the standalone root-level batch processor directly:
```bash
python run_totalsegmentator_batch.py --fast
```

### 3. Quality Audit
To audit the generated masks (Stomach, Duodenum, Small Bowel, Colon) against ground-truth datasets:
```bash
bash code/submit_audit.sh
```
This multi-processed script computes DSC, HD95, and Betti-0 metrics and saves the output to `results/audit_summary.csv`.

### 4. Synthetic Tumors & Model Training
Navigate to `code/SyntheticTumors/` to procedural generate 3D tumors or train models.
```bash
# Train on synthetic vs real tumor datasets
python main.py --syn
```

---

## 📊 Data & Outputs

- **Input CT Scans:**
  - `CancerVerse_data/BDMAP_XXXX/ct.nii.gz`
  - `mp-factory/CancerVerse/CancerVerse/CV_XXXX/ct.nii.gz`
- **Generated Masks:** Saved to `mp-factory/results/` (`totalseg_gi_masks`, `totalseg_masks`, etc.)
