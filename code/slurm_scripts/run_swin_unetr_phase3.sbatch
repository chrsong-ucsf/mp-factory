#!/bin/bash
#SBATCH --job-name=swin_p3
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/swin_p3_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/swin_p3_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

# Load environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

FOLD=${SLURM_ARRAY_TASK_ID}

# Paths
DATA_DIR="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"
AUDIT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase2_audit_summary.csv"
ENSEMBLE_OUT="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase2_out"
AUTOLABEL_DIR="/mnt/scratch/user/chrsong/mp-factory/results/autolabel_phase2_out"
GOLD_STD_DIR="/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected"
OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_phase3/fold_${FOLD}"

# Warm-start finetuning from best Phase 2 checkpoint
PHASE2_CKPT="/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_phase2/fold_${FOLD}/best_swin_unetr_phase2.pt"

mkdir -p "${OUT_DIR}"
mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs

echo "[Phase 3 Cycle 7] Starting Swin-UNETR Fold ${FOLD} on GPU: ${CUDA_VISIBLE_DEVICES}"

python -u /mnt/scratch/user/chrsong/mp-factory/code/training/train_swin_unetr_phase2.py \
    --data_dir          "${DATA_DIR}" \
    --audit_csv         "${AUDIT_CSV}" \
    --ensemble_out_dir  "${ENSEMBLE_OUT}" \
    --autolabel_dir     "${AUTOLABEL_DIR}" \
    --gold_standard_dir "${GOLD_STD_DIR}" \
    --out_dir           "${OUT_DIR}" \
    --fold              "${FOLD}" \
    --num_folds         4 \
    --epochs            150 \
    --batch_size        1 \
    --lr                0.00005 \
    --val_interval      5 \
    --use_weak \
    --pretrained_ckpt   "${PHASE2_CKPT}"
