#!/bin/bash
#SBATCH --job-name=eval_ensemble
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/eval_ensemble_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/eval_ensemble_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# Load modules and activate environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

# Define Paths
DIR_MEDNEXT="/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions"
DIR_SWIN="/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions"

OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out"
OUT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv"
SPLITS_DIR="/mnt/scratch/user/chrsong/mp-factory/results/dataset_splits"

mkdir -p "$OUT_DIR"

# Run 2-model ensemble: MedNeXt-B + Swin-UNETR
# Ensembles our 2 primary trained deep models sharing identical 1.5x1.5x2.0mm RAS space.
python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/evaluate_multi_model_ensemble.py \
    --pred_dirs "$DIR_MEDNEXT" "$DIR_SWIN" \
    --model_names MedNeXt Swin-UNETR \
    --out_dir "$OUT_DIR" \
    --num_workers 16 \
    --out_csv "$OUT_CSV" \
    --splits_dir "$SPLITS_DIR"
