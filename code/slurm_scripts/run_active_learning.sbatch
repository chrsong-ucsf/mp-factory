#!/bin/bash
#SBATCH --job-name=gi_active_learning
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/active_learning_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/active_learning_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/user/chrsong/mp-factory}"
CODE_DIR="${CODE_DIR:-${SCRATCH_ROOT}/code}"
RESULTS_DIR="${RESULTS_DIR:-${SCRATCH_ROOT}/results}"
AUDIT_CSV="${AUDIT_CSV:-${RESULTS_DIR}/ensemble_audit_summary.csv}"
TOP_K="${TOP_K:-10}"
OUT_JSON="${OUT_JSON:-${RESULTS_DIR}/radiologist_query_queue.json}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/scratch/user/chrsong/envs/mp-factory}"

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_PREFIX}"

export PYTHONPATH="${SCRATCH_ROOT}:${CODE_DIR}:${PYTHONPATH:-}"

echo "=================================================================="
echo "Job              : ${SLURM_JOB_ID:-local}"
echo "Node             : $(hostname)"
echo "Audit CSV        : ${AUDIT_CSV}"
echo "Top K            : ${TOP_K}"
echo "Out JSON         : ${OUT_JSON}"
echo "=================================================================="

python -u "${CODE_DIR}/utilities/run_active_learning.py" \
    --audit_csv "${AUDIT_CSV}" \
    --top_k "${TOP_K}" \
    --out_json "${OUT_JSON}"
