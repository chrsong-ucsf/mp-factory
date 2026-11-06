#!/bin/bash
#SBATCH --job-name=pytest_suite
#SBATCH --partition=community,community_short,cpu
#SBATCH --qos=hpc_core
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/pytest_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/pytest_%j.err

set -eo pipefail

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

echo "====================================================================="
echo "  Starting Pytest Suite on $(hostname)"
echo "  Date: $(date)"
echo "====================================================================="

pytest -v

echo "====================================================================="
echo "  Pytest completed successfully at $(date)"
echo "====================================================================="
