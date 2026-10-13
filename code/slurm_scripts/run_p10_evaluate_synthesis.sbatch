#!/bin/bash
#SBATCH --job-name=p10_eval_synth
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/p10_eval_synth_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/p10_eval_synth_%A_%a.err
#SBATCH --partition=gpu,large_gpu
#SBATCH --qos=hpc_core
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

BASE_DIR="/mnt/scratch/user/chrsong/mp-factory"
LOGS_DIR="${BASE_DIR}/logs"
REPORT_DIR="${BASE_DIR}/results/project10/evaluation_reports"

mkdir -p "${LOGS_DIR}"
mkdir -p "${REPORT_DIR}"

echo "====================================================================="
echo "  Project 10 Multiphase Synthesis Evaluation Pipeline (8 GPUs)"
echo "  Worker: ${SLURM_ARRAY_TASK_ID} / 8 GPUs"
echo "  Date:   $(date)"
echo "  Output: ${REPORT_DIR}"
echo "====================================================================="

python -u "${BASE_DIR}/code/project10/evaluation/evaluate_synthesis.py" \
    --fake_dir "${BASE_DIR}/results/project10/synthetic_volumes" \
    --real_dir "${BASE_DIR}/results/project10/registered_volumes" \
    --output_csv "${REPORT_DIR}/synthesis_metrics.csv" \
    --array_id "${SLURM_ARRAY_TASK_ID}" \
    --total_workers 8

echo "====================================================================="
echo "  Worker ${SLURM_ARRAY_TASK_ID} completed evaluation at $(date)"
echo "====================================================================="
