#!/bin/bash
#SBATCH --job-name=swin_unetr_gi
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/swin_unetr_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/swin_unetr_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Load modules and activate environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

# Make directories for outputs and logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_models

DATA_DIR="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"
OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_models"

# Execute Swin-UNETR 3D Training Pipeline
python -u /mnt/scratch/user/chrsong/mp-factory/code/train_swin_unetr.py \
    --data_dir "${DATA_DIR}" \
    --out_dir "${OUT_DIR}" \
    --epochs 100 \
    --batch_size 1 \
    --lr 0.0001 \
    --val_interval 2
