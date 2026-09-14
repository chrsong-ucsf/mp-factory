# AGY Autonomous Execution Playbook: mp-factory Phase 2 Scaling & Refinement

> **Target Agent**: Google Antigravity (AGY) / Autonomous Subagent  
> **Workspace**: `/mnt/scratch/user/chrsong/mp-factory`  
> **Environment**: `/mnt/scratch/user/chrsong/envs/mp-factory`  
> **Execution Directive**: Execute the phases sequentially. Verify each step against the specified exit criteria before advancing. Do not stop until all verification tests pass.

---

## 0. Runtime Initialization & Environment Setup

Whenever executing shell commands in this workspace, the agent must initialize and activate the environment:
```bash
cd /mnt/scratch/user/chrsong/mp-factory
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory
```

---

## Phase 1: Evaluation Completion & Scaling Law Analysis

### Goal
Ensure all 18 checkpoints have been evaluated on both the 20-case clean consensus dataset (Option A) and the JHU radiologist-corrected gold standard dataset (Option B), and generate publication-ready plots.

### Step 1.1: Monitor & Verify Job B Completion
- **Action**: Check if SLURM job `1714487` (or any active `eval_sweep_B` job) is running:
  ```bash
  squeue -u chrsong -n eval_sweep_B
  ```
- **If running**: Tail the log to observe progress:
  ```bash
  tail -n 25 logs/eval_sweep_B_*.out
  ```
  Wait until the job completes.
- **Verification / Exit Criteria**:
  - `results/scaling_sweep/scaling_sweep_eval_goldstandard.csv` exists and contains 19 lines (1 header + 18 rows).
  - No fatal errors in `logs/eval_sweep_B_*.err`.

### Step 1.2: Generate Scaling Figures
- **Action**: Run the visualization script on both evaluation outputs:
  ```bash
  python code/evaluation/plot_scaling_curves.py \
      --csv results/scaling_sweep/scaling_sweep_eval_20cases.csv \
      --out_dir results/scaling_sweep/figures_20cases

  python code/evaluation/plot_scaling_curves.py \
      --csv results/scaling_sweep/scaling_sweep_eval_goldstandard.csv \
      --out_dir results/scaling_sweep/figures_goldstandard
  ```
- **Verification / Exit Criteria**:
  - `results/scaling_sweep/figures_20cases/scaling_curve_overall.png` exists (>0 bytes).
  - `results/scaling_sweep/figures_20cases/scaling_curve_per_organ.png` exists (>0 bytes).
  - `results/scaling_sweep/figures_goldstandard/scaling_curve_overall.png` exists (>0 bytes).

---

## Phase 2: Root-Cause Investigation of Small Bowel (Dice = 0.0)

### Problem Definition
In the 20-case evaluation, `dice_small_bowel` was `0.0000` across all 18 models, despite stomach and duodenum scaling up to ~0.07.

### Step 2.1: Run Label & Prediction Diagnostic
- **Action**: Create and run a diagnostic check `code/evaluation/diagnose_small_bowel.py` to inspect:
  1. Class index alignment:
     - Check if label index `3` in `CancerVerse_dbox` / `ensemble_out` / `JHU_data_radiologist_corrected` actually represents small bowel.
     - Check label voxel frequencies across 5 cases.
  2. Model output distribution:
     - Check `torch.argmax(out, dim=1)` on a single validation volume to see if class `3` is ever predicted or if its softmax logits are suppressed below background/colon.
  3. Crop sampling verification:
     - In `RandCropByPosNegLabeld`, check if small bowel patches are ever sampled during training.
- **Command**:
  ```bash
  python -c "
  import nibabel as nib, numpy as np, glob, os
  lbl_files = glob.glob('/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out/*_consensus.nii.gz')[:5]
  for f in lbl_files:
      data = nib.load(f).get_fdata()
      unique, counts = np.unique(data, return_counts=True)
      print(os.path.basename(f), dict(zip(unique.astype(int), counts)))
  "
  ```
- **Verification / Exit Criteria**:
  - Confirm whether label `3` exists in the consensus masks and note its relative voxel frequency.
  - Document whether the issue is data-side (label indexing) or model-side (loss penalty / foreground suppression).

---

## Phase 3: Train Full-Dataset Benchmark Model ($N=1,594$)

### Goal
Train a benchmark MedNeXt-B model on all available usable training data ($N=1,594$) to determine the upper-bound performance ceiling of Phase 2 pseudo-label fine-tuning.

### Step 3.1: Construct SLURM Training Script
- **Action**: Create `code/slurm_scripts/train_full_anchor.sh` configuring:
  - Partition: `gpu`
  - GPU: `1x nvidia_l40s` or `1x a100`
  - CPUs: 8
  - Mem: 32G
  - Time: `24:00:00`
  - Command:
    ```bash
    python code/training/train_mednext_phase2.py \
        --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
        --ensemble_out_dir /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out \
        --audit_csv /mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv \
        --autolabel_dir /mnt/scratch/user/chrsong/mp-factory/results/autolabel_out \
        --pretrained_ckpt /mnt/scratch/user/chrsong/mp-factory/results/mednext_models/fold_0/best_mednext_gi.pt \
        --output_dir /mnt/scratch/user/chrsong/mp-factory/results/anchor_full_n1594 \
        --epochs 150 \
        --batch_size 2 \
        --lr 1e-4 \
        --use_weak
    ```
### Step 3.2: Launch & Monitor
- **Action**: Submit via `sbatch code/slurm_scripts/train_full_anchor.sh`.
- **Verification / Exit Criteria**:
  - Job accepted by SLURM with valid JOBID.
  - Initial log shows `Epoch [001/150]` with healthy loss and no CUDA out-of-memory errors.

---

## Phase 4: Final Summary Report Generation

### Goal
Synthesize results from Option A, Option B, diagnostic analysis, and the scaling curve fit into a single markdown artifact.

### Step 4.1: Compute Power-Law Scaling Exponent
- Fit the empirical mean Dice scores to the power-law equation:
  $$\text{Dice}(N) = a - b \cdot N^{-\alpha}$$
  where $\alpha$ is the scaling exponent and $a$ is the estimated asymptotic performance ceiling.

### Step 4.2: Output Artifact
- Generate `results/scaling_sweep/SCALING_SWEEP_REPORT.md` documenting:
  - Aggregate table across all cohort sizes ($N \in \{4, 10, 15, 20, 30, 50\}$).
  - Comparison of held-out consensus vs. radiologist gold standard.
  - Small bowel diagnostic conclusions.
  - Final recommendations for Phase 3 active learning / human-in-the-loop expansion.
