#!/bin/bash
#SBATCH --job-name=p9_study
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/p9_study_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/p9_study_%j.err
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

mkdir -p logs results/p9_independent_study

echo "Running P9 Independent Study 4-Arm Benchmark at $(date)..."
python -u code/evaluation/benchmark_p9_independent_study.py \
    --out_dir results/p9_independent_study

echo "Job finished at $(date) with exit code $?"
