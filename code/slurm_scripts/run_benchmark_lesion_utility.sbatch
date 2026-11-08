#!/bin/bash
#SBATCH --job-name=lesion_utility
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/lesion_utility_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/lesion_utility_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
cd /mnt/scratch/user/chrsong/mp-factory

mkdir -p logs results/project10/evaluation_reports

echo "Starting 3-Arm Lesion Augmentation Utility Benchmark at $(date)..."
python -u code/project10/evaluation/benchmark_lesion_augmentation_utility.py \
    --out_json results/project10/evaluation_reports/lesion_augmentation_utility_benchmark.json \
    --epochs 10

echo "Benchmark completed at $(date)."
