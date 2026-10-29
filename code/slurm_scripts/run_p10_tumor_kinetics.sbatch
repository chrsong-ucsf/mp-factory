#!/bin/bash
#SBATCH --job-name=p10_tumor_kin
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/p10_tumor_kin_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/p10_tumor_kin_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

# -----------------------------------------------------------------------------
# Module Loading & Environment Activation
# -----------------------------------------------------------------------------
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

BASE_DIR="/mnt/scratch/user/chrsong/mp-factory"
LOGS_DIR="${BASE_DIR}/logs"
OUT_DIR="${BASE_DIR}/results/project10/synthetic_tumors"

mkdir -p "${LOGS_DIR}"
mkdir -p "${OUT_DIR}"

echo "====================================================================="
echo "  Project 10 Multi-Phase Tumor Kinetics Synthesis"
echo "  Worker: ${SLURM_ARRAY_TASK_ID} / 6 parallel workers"
echo "  Date:   $(date)"
echo "  Output: ${OUT_DIR}"
echo "====================================================================="

python -u "${BASE_DIR}/code/project10/kinetics/generate_multiphase_tumors.py" \
    --volumes_dir "${BASE_DIR}/results/project10/synthetic_volumes" \
    --mask_dirs \
        "${BASE_DIR}/results/mednext_pool873_predictions" \
        "${BASE_DIR}/results/totalseg_gi_masks" \
        "${BASE_DIR}/results/totalseg_gi_masks_bdmap" \
        "${BASE_DIR}/results/totalseg_masks" \
    --out_dir "${OUT_DIR}" \
    --tumor_types hcc pdac hemangioma metastasis \
    --phases ncct arterial venous \
    --array_id "${SLURM_ARRAY_TASK_ID}" \
    --total_workers 4

echo "====================================================================="
echo "  Worker ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "====================================================================="
