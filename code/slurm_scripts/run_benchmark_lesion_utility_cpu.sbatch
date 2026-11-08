#!/bin/bash
#SBATCH --job-name=lesion_cpu
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/lesion_cpu_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/lesion_cpu_%j.err
#SBATCH --partition=community,community_short,cpu
#SBATCH --qos=hpc_core
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
cd /mnt/scratch/user/chrsong/mp-factory

mkdir -p logs results/project10/evaluation_reports

echo "Running 3-Arm Lesion Augmentation Utility Benchmark on CPU at $(date)..."
python -u code/project10/evaluation/benchmark_lesion_augmentation_utility.py \
    --out_json results/project10/evaluation_reports/lesion_augmentation_utility_benchmark.json \
    --epochs 10

echo "Benchmark completed at $(date)."
