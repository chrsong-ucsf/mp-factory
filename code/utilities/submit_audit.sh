#!/bin/bash
#SBATCH --job-name=audit_gi_db
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/audit_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/audit_%j.err

# Load CBI and miniforge modules, then activate environment via mamba hook
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

python /mnt/scratch/user/chrsong/mp-factory/code/evaluate_gi_masks.py
