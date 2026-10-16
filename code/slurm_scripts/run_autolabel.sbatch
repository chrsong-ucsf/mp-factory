#!/bin/bash
#SBATCH --job-name=autolabel
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/autolabel_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/autolabel_%j.err
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00

# Load environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/autolabel_out

echo "Starting hard-threshold auto-labeling on $(hostname) with ${SLURM_CPUS_PER_TASK} workers..."

python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/hard_threshold_autolabel.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --consensus_dir /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out \
    --audit_csv /mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv \
    --out_dir /mnt/scratch/user/chrsong/mp-factory/results/autolabel_out \
    --num_workers "${SLURM_CPUS_PER_TASK:-16}"

echo "Auto-labeling completed successfully."
