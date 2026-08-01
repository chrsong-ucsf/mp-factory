#!/bin/bash
#SBATCH --job-name=model_compare
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/model_compare_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/model_compare_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results

python -u /mnt/scratch/user/chrsong/mp-factory/code/compare_models.py \
    --data_dir          "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox" \
    --swin_pred_dir     "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions" \
    --mednext_pred_dir  "/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions" \
    --out_csv           "/mnt/scratch/user/chrsong/mp-factory/results/model_comparison.csv" \
    --max_subjects 200
