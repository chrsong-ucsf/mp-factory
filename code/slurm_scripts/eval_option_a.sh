#!/usr/bin/env bash
#SBATCH --job-name=eval_sweep_A
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/eval_sweep_A_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/eval_sweep_A_%j.err

set -euo pipefail

cd /mnt/scratch/user/chrsong/mp-factory

eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs

echo "Starting Evaluation A (20 Clean Cases) on $(hostname) at $(date)"
python code/evaluation/eval_scaling_checkpoints.py \
    --device cuda \
    --num_val_cases 20 \
    --output_csv /mnt/scratch/user/chrsong/mp-factory/results/scaling_sweep/scaling_sweep_eval_20cases.csv
echo "Finished Evaluation A at $(date)"
