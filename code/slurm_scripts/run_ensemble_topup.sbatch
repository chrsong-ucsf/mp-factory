#!/bin/bash
#SBATCH --job-name=ensemble_topup
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_topup_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_topup_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out

echo "====================================================================="
echo "  Ensemble Audit Top-Up  — Chunk ${SLURM_ARRAY_TASK_ID}/2"
echo "  Date: $(date)"
echo "  Purpose: Process 36 subjects missing from ensemble_audit_summary.csv"
echo "====================================================================="

# Smart resume: skips the 1699 already-audited subjects automatically.
# 2 chunks split the 36 remaining subjects for parallel completion.
python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/evaluate_multi_model_ensemble.py \
    --pred_dirs \
        "/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions" \
        "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions" \
    --model_names MedNeXt Swin-UNETR \
    --out_dir  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out" \
    --out_csv  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary_topup_${SLURM_ARRAY_TASK_ID}.csv" \
    --splits_dir "/mnt/scratch/user/chrsong/mp-factory/results/dataset_splits" \
    --num_chunks 2 \
    --chunk_idx "${SLURM_ARRAY_TASK_ID}"

echo "====================================================================="
echo "  Chunk ${SLURM_ARRAY_TASK_ID} complete at $(date)"
echo "====================================================================="
