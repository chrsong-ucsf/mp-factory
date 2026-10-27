#!/bin/bash
#SBATCH --job-name=ens_p3_eval
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_phase3_eval_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_phase3_eval_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase3_out
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/dataset_splits_phase3

echo "====================================================================="
echo "  Starting Phase 3 Ensemble Worker ${SLURM_ARRAY_TASK_ID}/4 on $(hostname)"
echo "  Date: $(date)"
echo "====================================================================="

# Sharded CUDA GPU 3D Ensemble Execution across MedNeXt Phase 3 + Swin-UNETR Phase 3
python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/evaluate_multi_model_ensemble.py \
    --pred_dirs \
        "/mnt/scratch/user/chrsong/mp-factory/results/mednext_phase3_predictions" \
        "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_phase3_predictions" \
    --model_names MedNeXt_Phase3 Swin-UNETR_Phase3 \
    --out_dir  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase3_out" \
    --out_csv  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase3_audit_summary_${SLURM_ARRAY_TASK_ID}.csv" \
    --splits_dir "/mnt/scratch/user/chrsong/mp-factory/results/dataset_splits_phase3" \
    --num_chunks 4 \
    --chunk_idx "${SLURM_ARRAY_TASK_ID}"
