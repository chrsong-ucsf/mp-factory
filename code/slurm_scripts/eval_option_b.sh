#!/usr/bin/env bash
#SBATCH --job-name=eval_sweep_B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/eval_sweep_B_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/eval_sweep_B_%j.err

set -euo pipefail

cd /mnt/scratch/user/chrsong/mp-factory

eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs

echo "Starting Evaluation B (Gold Standard JHU) on $(hostname) at $(date)"
python code/evaluation/eval_scaling_checkpoints.py \
    --device cuda \
    --gold_standard_dir /mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected \
    --num_val_cases 50 \
    --output_csv /mnt/scratch/user/chrsong/mp-factory/results/scaling_sweep/scaling_sweep_eval_goldstandard.csv
echo "Finished Evaluation B at $(date)"
