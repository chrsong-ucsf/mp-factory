#!/bin/bash
#SBATCH --job-name=gi_step5_extval
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/step5_extval_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/step5_extval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/user/chrsong/mp-factory}"
CODE_DIR="${CODE_DIR:-${SCRATCH_ROOT}/code}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRATCH_ROOT}/results/scaling_sweep/N0050/best_checkpoint.pt}"
OUT_DIR="${OUT_DIR:-${SCRATCH_ROOT}/results/cross_site_report}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/scratch/user/chrsong/envs/mp-factory}"

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_PREFIX}"

export PYTHONPATH="${SCRATCH_ROOT}:${CODE_DIR}:${PYTHONPATH:-}"

echo "=================================================================="
echo "Job              : ${SLURM_JOB_ID:-local}"
echo "Step 5           : Cross-Site Out-of-Domain External Validation"
echo "Checkpoint       : ${CHECKPOINT_PATH}"
echo "Output Dir       : ${OUT_DIR}"
echo "=================================================================="

# Run cross-site evaluation (falls back to dummy generator if test datasets are missing)
python -u "${SCRATCH_ROOT}/src/eval_cross_site.py" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --output_dir "${OUT_DIR}" \
    --generate_dummy
