#!/bin/bash
#SBATCH --job-name=totalseg_bdmap
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/totalseg_bdmap_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/totalseg_bdmap_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

source activate /mnt/scratch/user/chrsong/envs/mp-factory

python -u /mnt/scratch/user/chrsong/mp-factory/code/run_totalseg_bdmap.py ${SLURM_ARRAY_TASK_ID} 4
