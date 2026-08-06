# mp-factory (Massive Processing Factory)

**mp-factory** is an HPC-scale medical image processing and deep learning factory built
around the **CancerVerse** and **BodyMaps/AbdomenAtlas** CT datasets. It delivers
**Project 9 (GI Organ Segmentation)** through a fully-automated, "under-annotation-tolerant"
pipeline: instead of hand-correcting noisy labels across ~22,000 scans, it mathematically
filters them and distills clean pseudo-labels into a compact student model.

**Pipeline in one line:** download → segment (TotalSeg + MedNeXt + Swin-UNETR) →
ensemble-cleanse → distill (Phase 2) → validate against radiologist ground truth.

## 🚀 Key Features

1. **Data Ingestion & HPC Downloading** — automated pulls of large-scale 3D CT datasets (`BodyMaps`/`CancerVerse`) via Hugging Face Hub.
2. **Automated 3D Organ Segmentation** — parallel TotalSegmentator inference plus trained **MedNeXt-B** and **Swin-UNETR** GI segmenters across `BDMAP_*` and `CV_*` cohorts.
3. **Multi-Model Ensemble & Automated Data Cleansing** — consensus pseudo-GT assembly, spatial uncertainty heatmaps, and topological/metric auditing (Dice, HD95, Betti-0, ARI, VOI, ECE) that triage every scan into `CLEAN_HIGH_CONFIDENCE`, `WEAK_COARSE`, or `NOISE_REJECT`.
4. **Teacher–Student Distillation (Phase 2)** — train MedNeXt-B on consensus pseudo-ground-truth, ignoring conflicting boundary pixels (`ignore_index=255`).
5. **Expert Validation** — GPU-accelerated evaluation against JHU radiologist-corrected masks, applying multi-phase mask-propagation rules.
6. **Synthetic Tumor Synthesis** — `SyntheticTumors` submodule for procedural liver/pancreatic tumor generation and 3D model training.

**GI organ label map:** `0` background · `1` stomach · `2` duodenum · `3` small bowel / intestine · `4` colon.

---

## 📂 Repository Layout

```
mp-factory/
├── AGENTS.md                       # Detailed architecture & agent guide
├── ENSEMBLE_FIX_PLAN.md            # Automated-cleansing & distillation design doc
├── requirements.txt / setup.py     # Packaging (environment.yml for conda)
├── CancerVerse/                    # Dataset repo, metadata, HF specs (data gitignored)
├── JHU_data_radiologist_corrected/ # Expert radiologist GT (*.seg.nrrd) + clinical notes
├── code/
│   ├── active_learning/            # Uncertainty estimation & pool orchestration (optional path)
│   ├── data_download/              # download.sbatch (Hugging Face pull)
│   ├── evaluation/                 # GI audit, ensemble cleansing, JHU eval, verification, monitor
│   ├── prediction/                 # MedNeXt / Swin-UNETR / TotalSegmentator batch inference
│   ├── slurm_scripts/              # All *.sbatch jobs + TotalSeg array runners
│   ├── training/                   # train_mednext[_phase2].py, train_swin_unetr.py
│   ├── utilities/                  # compare_models, merge_ensemble_chunks, audits, tests
│   └── SyntheticTumors/            # Synthetic tumor generator & trainer (gitignored)
├── graphify-out/                   # Generated knowledge-graph artifacts (GRAPH_REPORT.md)
├── logs/                           # SLURM logs (gitignored)
└── results/                        # Masks, consensus, CSV summaries (gitignored)
```

---

## ⚙️ Environment & Setup

**Requirements:** Python 3.11, PyTorch, MONAI, TotalSegmentator, `nnunet_mednext`,
`nibabel`, `pynrrd`, `scikit-image`, `scikit-learn` (see `requirements.txt`).

**Environment activation (HPC):**
```bash
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory
```

---

## 🛠️ Workflows & Execution

### 1. Data Ingestion
```bash
sbatch code/data_download/download.sbatch
# OR
bash CancerVerse/download_cancerverse.sh
```

### 2. Baseline 3D Segmentation (TotalSegmentator)
```bash
sbatch code/slurm_scripts/run_totalseg.sbatch          # CV cohort (multi-label)
sbatch code/slurm_scripts/run_totalseg_bdmap.sbatch    # BDMAP cohort (job array)
sbatch code/slurm_scripts/run_totalseg_gi_array.sbatch # GI organ extraction (array worker)
# Standalone:
python code/prediction/run_totalsegmentator_batch.py --fast
```

### 3. Train Deep Models (4-fold SLURM arrays)
```bash
sbatch code/slurm_scripts/run_mednext.sbatch      # MedNeXt-B (Phase 1)
sbatch code/slurm_scripts/run_swin_unetr.sbatch   # Swin-UNETR
```

### 4. Batch Inference
```bash
sbatch code/slurm_scripts/run_predict_mednext.sbatch
sbatch code/slurm_scripts/run_predict_swin_unetr.sbatch
```

### 5. Ensemble Consensus + Automated Cleansing
```bash
sbatch code/slurm_scripts/run_evaluation.sbatch        # CPU, 16 workers, 128GB
# or GPU:  sbatch code/slurm_scripts/run_ensemble_eval.sbatch
python code/utilities/merge_ensemble_chunks.py \
  --pattern "results/ensemble_audit_summary_*.csv" \
  --out_csv "results/ensemble_audit_summary.csv"
```
Produces per-subject `results/ensemble_out/<subject>_consensus.nii.gz` and an
`ensemble_audit_summary.csv` with the triage category and recommended action per scan.

### 6. Phase-2 Distillation (Student on Consensus Pseudo-GT)
```bash
sbatch code/slurm_scripts/run_mednext_phase2.sbatch
```
Selects `CLEAN_HIGH_CONFIDENCE` cases (optionally `WEAK_COARSE` with `ignore_index=255`)
from the audit CSV and trains MedNeXt-B with cosine-annealing LR.

### 7. Validation Against Expert Ground Truth
```bash
sbatch code/slurm_scripts/run_jhu_radiologist_eval_gpu.sbatch
# Compare architectures:
sbatch code/slurm_scripts/run_compare_models.sbatch
# Monitor / verify:
python code/evaluation/watch_pipeline.py
python code/evaluation/verify_pipeline_run.py
```

### 8. Synthetic Tumors & Model Training
```bash
cd code/SyntheticTumors && python main.py --syn
```

---

## 📊 Data & Outputs

- **Input CT scans:**
  - Default (training/inference): `CancerVerse_dbox/<subject>/ct.nii.gz`
  - BDMAP: `CancerVerse_data/BDMAP_XXXX/ct.nii.gz`
  - CV: `mp-factory/CancerVerse/CancerVerse/CV_XXXX/ct.nii.gz`
- **Outputs (in `results/`):** `mednext_predictions/`, `swin_unetr_predictions/`,
  `totalseg_gi_masks*/`, `ensemble_out/` (`*_consensus.nii.gz`),
  `ensemble_audit_summary.csv`, `model_comparison.csv`, and JHU audit CSVs.

> `results/`, `logs/`, `CancerVerse/` data, `*.nii.gz`, `envs/`, and `SyntheticTumors/`
> are gitignored. See `AGENTS.md` for the full architecture and per-script reference.
