#!/bin/bash
#SBATCH --job-name=compile_tables
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/compile_tables_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/compile_tables_%j.err
#SBATCH --partition=community,community_short,cpu
#SBATCH --qos=hpc_core
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
cd /mnt/scratch/user/chrsong/mp-factory

mkdir -p logs results/manuscript

echo "Compiling manuscript tables at $(date)..."
python -u code/manuscript/compile_manuscript_tables.py
echo "Table compilation completed at $(date)."
