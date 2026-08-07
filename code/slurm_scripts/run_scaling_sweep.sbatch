#!/bin/bash
#SBATCH --job-name=gi_scaling_sweep
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/scaling_sweep_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/scaling_sweep_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

# Cluster environment paths
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/user/chrsong/mp-factory}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/CancerVerse_dbox}"
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/logs}"
CODE_DIR="${CODE_DIR:-${SCRATCH_ROOT}/code}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/results/scaling_sweep}"
RAD_GT_DIR="${RAD_GT_DIR:-${SCRATCH_ROOT}/JHU_data_radiologist_corrected}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/scratch/user/chrsong/envs/mp-factory}"

# Hyperparameters
COHORT_SIZES="${COHORT_SIZES:-4 10 15 20 30 50}"
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

# Install MedNeXt if missing
python -c "import nnunet_mednext" 2>/dev/null \
  || pip install --quiet git+https://github.com/MIC-DKFZ/MedNeXt.git

export PYTHONPATH="${SCRATCH_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=================================================================="
echo "Job              : ${SLURM_JOB_ID:-local}"
echo "Node             : $(hostname)"
echo "Cohort Sizes     : ${COHORT_SIZES}"
echo "Loss             : ${LOSS} (alpha=${ALPHA}, beta=${BETA})"
echo "Epochs           : ${EPOCHS}"
echo "Data Dir         : ${DATA_DIR}"
echo "Radiologist GT   : ${RAD_GT_DIR}"
echo "Output Dir       : ${OUT_DIR}"
echo "=================================================================="

python -u "${CODE_DIR}/training/run_scaling_sweep.py" \
    --data_dir "${DATA_DIR}" \
    --mask_dir "${DATA_DIR}" \
    --cohort_sizes ${COHORT_SIZES} \
    --loss "${LOSS}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}" \
    --epochs "${EPOCHS}" \
    --radiologist_gt_dir "${RAD_GT_DIR}" \
    --output_dir "${OUT_DIR}"
