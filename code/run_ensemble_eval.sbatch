#!/bin/bash
#SBATCH --job-name=ensemble_eval
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_eval_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out

python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluate_multi_model_ensemble.py \
    --pred_dirs \
        "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions" \
        "/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions" \
    --model_names Swin-UNETR MedNeXt \
    --out_dir  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out" \
    --out_csv  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv"
