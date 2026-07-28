#!/bin/bash
#SBATCH --job-name=totalseg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/totalseg_gi_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/totalseg_gi_%A_%a.err

# Initialize mamba shell hook for non-interactive Slurm session
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

# Run worker script across array tasks
python /mnt/scratch/user/chrsong/mp-factory/code/run_totalseg_gi_array.py $SLURM_ARRAY_TASK_ID 4
