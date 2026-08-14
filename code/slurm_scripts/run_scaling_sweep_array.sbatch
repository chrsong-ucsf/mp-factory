#!/bin/bash
#SBATCH --job-name=gi_scaling_sweep_arr
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/scaling_sweep_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/scaling_sweep_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%4

set -euo pipefail

# Sub-cohort mapping: array task 0->N=4, 1->N=10, 2->N=15, 3->N=20, 4->N=30, 5->N=50
SIZES=(4 10 15 20 30 50)
TARGET_N="${SIZES[$SLURM_ARRAY_TASK_ID]}"

# Cluster environment paths
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/user/chrsong/mp-factory}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/CancerVerse_dbox}"
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/logs}"
CODE_DIR="${CODE_DIR:-${SCRATCH_ROOT}/code}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/results/scaling_sweep}"
RAD_GT_DIR="${RAD_GT_DIR:-${SCRATCH_ROOT}/JHU_data_radiologist_corrected}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/scratch/user/chrsong/envs/mp-factory}"

# Hyperparameters
EPOCHS="${EPOCHS:-30}"
LOSS="${LOSS:-asymmetric}"
ALPHA="${ALPHA:-2.0}"
BETA="${BETA:-1.0}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# Environment setup
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_PREFIX}"

export PYTHONPATH="${SCRATCH_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=================================================================="
echo "Job Array Task   : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node             : $(hostname)"
echo "Target Cohort N  : ${TARGET_N}"
echo "Loss             : ${LOSS} (alpha=${ALPHA}, beta=${BETA})"
echo "Epochs           : ${EPOCHS}"
echo "=================================================================="

python -u "${CODE_DIR}/training/run_scaling_sweep.py" \
    --data_dir "${DATA_DIR}" \
    --mask_dir "${DATA_DIR}" \
    --cohort_sizes "${TARGET_N}" \
    --loss "${LOSS}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --epochs "${EPOCHS}" \
    --radiologist_gt_dir "${RAD_GT_DIR}" \
    --output_dir "${OUT_DIR}"
