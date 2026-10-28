#!/bin/bash
#SBATCH --job-name=autolabel_p3
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/autolabel_phase3_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/autolabel_phase3_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/autolabel_phase3_out

echo "====================================================================="
echo "  Starting Phase 3 Hard-Threshold Autolabeling (465 Weak Scans)"
echo "  Date: $(date)"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/hard_threshold_autolabel.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --consensus_dir /mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase3_out \
    --out_dir /mnt/scratch/user/chrsong/mp-factory/results/autolabel_phase3_out \
    --audit_csv /mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase3_audit_summary.csv \
    --num_workers 12

echo "====================================================================="
echo "  Autolabeling completed at $(date)"
echo "====================================================================="
