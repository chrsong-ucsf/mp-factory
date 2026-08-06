#!/bin/bash
#SBATCH --job-name=ensemble_eval_gpu
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_eval_gpu_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_eval_gpu_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out

# Fast CUDA GPU 3D Ensemble Execution across MedNeXt + Swin-UNETR
python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/evaluate_multi_model_ensemble.py \
    --pred_dirs \
        "/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions" \
        "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions" \
    --model_names MedNeXt Swin-UNETR \
    --out_dir  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out" \
    --out_csv  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv" \
    --splits_dir "/mnt/scratch/user/chrsong/mp-factory/results/dataset_splits"
