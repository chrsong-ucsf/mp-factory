#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=p10_discover
#SBATCH --output=logs/p10_discover_%j.out
#SBATCH --error=logs/p10_discover_%j.err

# ---------------------------------------------------------------------------
# Project 10 Phase 1 Step 1.1: Multi-Phase Cohort Discovery & Manifest Build
# ---------------------------------------------------------------------------
# Runs discover_multiphase_cohort.py to scan the CancerVerse metadata,
# identify paired (NCCT + post-contrast) patients, verify on-disk existence,
# and output:
#   results/project10/manifests/multiphase_pairs.csv
#   results/project10/manifests/multiphase_triplets.csv
#   results/project10/manifests/discovery_stats.json
# ---------------------------------------------------------------------------

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/project10/manifests

eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

echo "========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Started:   $(date)"
echo "========================================="

python -u code/project10/datasets/discover_multiphase_cohort.py \
    --metadata CancerVerse/CancerVerse_dataset_metadata.csv \
    --data_root CancerVerse/CancerVerse \
    --out_dir results/project10/manifests \
    --min_phases 2 \
    --verify_disk

echo ""
echo "Finished at: $(date)"
echo "Output files:"
ls -lh results/project10/manifests/
