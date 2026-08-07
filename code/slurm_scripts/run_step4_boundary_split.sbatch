#!/bin/bash
#SBATCH --job-name=gi_step4_split
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/step4_split_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/step4_split_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/user/chrsong/mp-factory}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/CancerVerse_dbox}"
CODE_DIR="${CODE_DIR:-${SCRATCH_ROOT}/code}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/scratch/user/chrsong/envs/mp-factory}"

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_PREFIX}"

export PYTHONPATH="${SCRATCH_ROOT}:${CODE_DIR}:${PYTHONPATH:-}"

echo "=================================================================="
echo "Job              : ${SLURM_JOB_ID:-local}"
echo "Step 4           : Centerline-Skeleton 5-Organ Boundary Splitting"
echo "Data Dir         : ${DATA_DIR}"
echo "=================================================================="

python -u "${CODE_DIR}/utilities/run_step4_boundary_split.py" \
    --data_dir "${DATA_DIR}" \
    --out_suffix gi_mask_5organ.nii.gz
