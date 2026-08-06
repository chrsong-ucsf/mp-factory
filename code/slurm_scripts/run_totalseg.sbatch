#!/bin/bash
#SBATCH --job-name=totalseg_batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/totalseg_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/totalseg_%j.err

set -eo pipefail

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

python /mnt/scratch/user/chrsong/mp-factory/code/run_totalseg.py
