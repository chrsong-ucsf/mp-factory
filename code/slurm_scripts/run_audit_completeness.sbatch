#!/bin/bash
#SBATCH --job-name=audit_comp
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/audit_comp_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/audit_comp_%j.err
#SBATCH --partition=community,community_short,cpu
#SBATCH --qos=hpc_core
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
cd /mnt/scratch/user/chrsong/mp-factory

mkdir -p logs results

python -u code/evaluation/audit_dataset_completeness.py \
    --data_dirs \
      /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
      /mnt/scratch/user/chrsong/CancerVerse_data \
      /mnt/scratch/user/chrsong/mp-factory/CancerVerse/CancerVerse \
    --out_csv results/dataset_annotation_completeness_audit.csv \
    --out_json results/dataset_annotation_completeness_summary.json
