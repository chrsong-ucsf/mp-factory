#!/bin/bash
#SBATCH --job-name=p10_turing
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/p10_turing_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/p10_turing_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/project10/evaluation_reports

echo "====================================================================="
echo "  Executing Project 10 Clinical Turing Test (Worker ${SLURM_ARRAY_TASK_ID}/4)"
echo "  Date: $(date)"
echo "  Node: $(hostname) | GPU: ${CUDA_VISIBLE_DEVICES}"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/project10/evaluation/run_clinical_turing_test.py \
    --synthetic_dir /mnt/scratch/user/chrsong/mp-factory/results/project10/synthetic_volumes \
    --real_dir /mnt/scratch/user/chrsong/mp-factory/results/project10/registered_volumes \
    --model_path /mnt/scratch/user/chrsong/mp-factory/results/mednext_phase3/fold_2/best_mednext_phase2.pt \
    --out_csv /mnt/scratch/user/chrsong/mp-factory/results/project10/evaluation_reports/clinical_turing_test_chunk_${SLURM_ARRAY_TASK_ID}.csv \
    --num_cases 100 \
    --array_id "${SLURM_ARRAY_TASK_ID}" \
    --total_workers 4

echo "====================================================================="
echo "  Worker ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "====================================================================="
