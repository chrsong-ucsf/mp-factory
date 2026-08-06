#!/bin/bash
#SBATCH --job-name=mednext_p2
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_p2_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_p2_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-3

# Load environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

# Install MedNeXt if not already installed
pip install --quiet git+https://github.com/MIC-DKFZ/MedNeXt.git

export PYTHONUNBUFFERED=1

FOLD=${SLURM_ARRAY_TASK_ID}

# Paths
DATA_DIR="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"
AUDIT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv"
ENSEMBLE_OUT="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out"
OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/mednext_phase2/fold_${FOLD}"

# Phase 1 pretrained checkpoint for warm-start (comment out if training from scratch)
PHASE1_CKPT="/mnt/scratch/user/chrsong/mp-factory/results/mednext_models/fold_${FOLD}/best_mednext_gi.pt"

mkdir -p "${OUT_DIR}"
mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs

echo "[Phase 2] Starting MedNeXt-B Fold ${FOLD} on GPU: ${CUDA_VISIBLE_DEVICES}"

python -u /mnt/scratch/user/chrsong/mp-factory/code/train_mednext_phase2.py \
    --data_dir        "${DATA_DIR}" \
    --audit_csv       "${AUDIT_CSV}" \
    --ensemble_out_dir "${ENSEMBLE_OUT}" \
    --out_dir         "${OUT_DIR}" \
    --model_id        "B" \
    --kernel_size     3 \
    --fold            "${FOLD}" \
    --num_folds       4 \
    --epochs          150 \
    --batch_size      1 \
    --lr              0.0001 \
    --val_interval    5 \
    --use_weak \
    --pretrained_ckpt "${PHASE1_CKPT}"
