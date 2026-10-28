#!/bin/bash
#SBATCH --job-name=mednext_pool873
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_pool873_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_pool873_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
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
AUDIT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase3_audit_summary.csv"
ENSEMBLE_OUT="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase3_out"
AUTOLABEL_DIR="/mnt/scratch/user/chrsong/mp-factory/results/autolabel_phase3_out"
GOLD_STD_DIR="/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected"
OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/mednext_pool873/fold_${FOLD}"

# Warm-start finetuning from best Phase 3 checkpoint
PHASE3_CKPT="/mnt/scratch/user/chrsong/mp-factory/results/mednext_phase3/fold_2/best_mednext_phase2.pt"

mkdir -p "${OUT_DIR}"
mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs

echo "====================================================================="
echo "  [Expanded Distillation] Starting MedNeXt-B on 873-Scan Golden Pool"
echo "  Fold:   ${FOLD}/4"
echo "  Node:   $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Output: ${OUT_DIR}"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/training/train_mednext_phase2.py \
    --data_dir         "${DATA_DIR}" \
    --audit_csv        "${AUDIT_CSV}" \
    --ensemble_out_dir "${ENSEMBLE_OUT}" \
    --autolabel_dir    "${AUTOLABEL_DIR}" \
    --gold_standard_dir "${GOLD_STD_DIR}" \
    --out_dir          "${OUT_DIR}" \
    --model_id         "B" \
    --kernel_size      3 \
    --fold             "${FOLD}" \
    --num_folds        4 \
    --epochs           150 \
    --batch_size       1 \
    --lr               0.00005 \
    --val_interval     5 \
    --use_weak \
    --pretrained_ckpt  "${PHASE3_CKPT}"
