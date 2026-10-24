#!/bin/bash
#SBATCH --job-name=autolabel_p2
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/autolabel_p2_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/autolabel_p2_%j.err
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
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/autolabel_phase2_out
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/dataset_splits_phase2

DATA_DIR="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"
CONSENSUS_DIR="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase2_out"
AUDIT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase2_audit_summary.csv"
SPLITS_FILE="/mnt/scratch/user/chrsong/mp-factory/results/dataset_splits_phase2/ensemble_split_WEAK_COARSE.txt"
AUTOLABEL_DIR="/mnt/scratch/user/chrsong/mp-factory/results/autolabel_phase2_out"
OUT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/autolabel_phase2_summary.csv"

echo "====================================================================="
echo "  Executing Phase 2 Hard-Threshold Auto-Labeling"
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "  Workers: ${SLURM_CPUS_PER_TASK:-16}"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/hard_threshold_autolabel.py \
    --data_dir "${DATA_DIR}" \
    --consensus_dir "${CONSENSUS_DIR}" \
    --audit_csv "${AUDIT_CSV}" \
    --splits_file "${SPLITS_FILE}" \
    --out_dir "${AUTOLABEL_DIR}" \
    --out_csv "${OUT_CSV}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-16}" \
    --overwrite

echo "====================================================================="
echo "  Building Unified Dataset Manifest & Verifying NIfTI Integrity"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/unified_gastro_dataset.py \
    --audit_csv "${AUDIT_CSV}" \
    --data_dir "${DATA_DIR}" \
    --consensus_dir "${CONSENSUS_DIR}" \
    --autolabel_dir "${AUTOLABEL_DIR}" \
    --use_weak \
    --verify \
    --out_dir "/mnt/scratch/user/chrsong/mp-factory/results/dataset_splits_phase2"

echo "====================================================================="
echo "  Phase 2 Auto-Labeling and Dataset Verification Completed: $(date)"
echo "====================================================================="
