#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=verify_unified
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/verify_unified_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/verify_unified_%j.err

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/dataset_splits_phase3

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

echo "========================================="
echo "  Verifying Unified Gastrointestinal Dataset (873 Scans)"
echo "  Date: $(date)"
echo "========================================="

python -u code/evaluation/unified_gastro_dataset.py \
    --audit_csv results/ensemble_phase3_audit_summary.csv \
    --data_dir CancerVerse_dbox \
    --consensus_dir results/ensemble_phase3_out \
    --autolabel_dir results/autolabel_phase3_out \
    --use_weak \
    --verify \
    --out_dir results/dataset_splits_phase3

echo "========================================="
echo "  Verification complete at $(date)"
echo "========================================="
