#!/bin/bash
#SBATCH --job-name=ensemble_audit_array
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_audit_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_audit_%A_%a.err
#SBATCH --partition=community
#SBATCH --qos=cmnt
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-15

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out

echo "====================================================================="
echo "  Starting Ensemble Worker ${SLURM_ARRAY_TASK_ID}/16 on $(hostname)"
echo "  Date: $(date)"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/evaluate_multi_model_ensemble.py \
    --pred_dirs \
        "/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions" \
        "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions" \
    --model_names MedNeXt Swin-UNETR \
    --out_dir "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out" \
    --out_csv "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary_${SLURM_ARRAY_TASK_ID}.csv" \
    --splits_dir "/mnt/scratch/user/chrsong/mp-factory/results/dataset_splits" \
    --num_workers 1 \
    --num_chunks 16 \
    --chunk_idx "${SLURM_ARRAY_TASK_ID}"

echo "====================================================================="
echo "  Worker ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "====================================================================="
