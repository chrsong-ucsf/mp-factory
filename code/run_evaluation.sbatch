#!/bin/bash
#SBATCH --job-name=eval_ensemble
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/eval_ensemble_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/eval_ensemble_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# Load modules and activate environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

# Define Paths
# IMPORTANT: MedNeXt must be listed FIRST — it defines the reference voxel grid
# for affine-aware TotalSegmentator resampling.
DIR_MEDNEXT="/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions"
DIR_SWIN="/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions"
DIR_TOTALSEG="/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap"

OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out"
OUT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv"

mkdir -p "$OUT_DIR"

# Run 3-model ensemble: MedNeXt + Swin-UNETR + TotalSegmentator
# TotalSeg is affine-resampled into MedNeXt's GI-crop voxel grid using world coordinates.
python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluate_multi_model_ensemble.py \
    --pred_dirs "$DIR_MEDNEXT" "$DIR_SWIN" "$DIR_TOTALSEG" \
    --model_names MedNeXt Swin-UNETR TotalSegmentator \
    --out_dir "$OUT_DIR" \
    --num_workers 16 \
    --out_csv "$OUT_CSV"
