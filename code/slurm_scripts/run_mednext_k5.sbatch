#!/bin/bash
#SBATCH --job-name=mednext_k5
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_k5_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_k5_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-1

# Load modules and activate environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/mednext_k5_models/fold_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUT_DIR}"

DATA_DIR="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"

echo "====================================================================="
echo "  MedNeXt-B (5x5x5 Large Kernel) Training — Fold ${SLURM_ARRAY_TASK_ID}"
echo "  Date: $(date)"
echo "  Architecture: MedNeXt-B with 5x5x5 Convolutions (Receptive Field Boost)"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/training/train_mednext.py \
    --data_dir "${DATA_DIR}" \
    --out_dir "${OUT_DIR}" \
    --model_id "B" \
    --kernel_size 5 \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    --num_folds 4 \
    --batch_size 1 \
    --lr 0.0001 \
    --val_interval 2

echo "====================================================================="
echo "  Fold ${SLURM_ARRAY_TASK_ID} complete at $(date)"
echo "====================================================================="
