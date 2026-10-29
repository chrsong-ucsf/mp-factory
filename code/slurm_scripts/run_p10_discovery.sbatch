#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=p10_discover
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/p10_discover_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/p10_discover_%j.err

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/project10/manifests

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

echo "========================================="
echo "  Project 10 Multi-Phase Cohort Discovery"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURMD_NODENAME"
echo "  Started:   $(date)"
echo "========================================="

python -u code/project10/datasets/discover_multiphase_cohort.py \
    --metadata CancerVerse/CancerVerse_dataset_metadata.csv \
    --data_root CancerVerse/CancerVerse \
    --out_dir results/project10/manifests \
    --min_phases 2

echo "========================================="
echo "  Finished at: $(date)"
echo "  Output manifests:"
ls -lh results/project10/manifests/
echo "========================================="
