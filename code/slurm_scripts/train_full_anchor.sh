#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=train_full_anchor
#SBATCH --output=logs/train_full_anchor_%j.out
#SBATCH --error=logs/train_full_anchor_%j.err

cd /mnt/scratch/user/chrsong/mp-factory
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

python code/training/train_mednext_phase2.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --ensemble_out_dir /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out \
    --audit_csv /mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv \
    --autolabel_dir /mnt/scratch/user/chrsong/mp-factory/results/autolabel_out \
    --pretrained_ckpt /mnt/scratch/user/chrsong/mp-factory/results/mednext_models/fold_0/best_mednext_gi.pt \
    --out_dir /mnt/scratch/user/chrsong/mp-factory/results/anchor_full_n1594 \
    --epochs 150 \
    --batch_size 2 \
    --lr 1e-4 \
    --use_weak
