#!/bin/bash
#SBATCH --job-name=eval_totalseg
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/eval_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/eval_%j.err
#SBATCH --partition=cpu          # CPU partition (change if named differently on your cluster)
#SBATCH --cpus-per-task=32       # Request 32 CPU cores
#SBATCH --mem=64G                # RAM allocation
#SBATCH --time=02:00:00          # 2 hours max

# Activate virtual environment
source activate /mnt/scratch/user/chrsong/envs/mp-factory

# Define Paths
PRED_DIR="/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks"
GT_DIR="/mnt/scratch/user/chrsong/mp-factory/CancerVerse/ground_truth_masks"  # ADJUST THIS TO YOUR GT PATH
OUT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_metrics_summary.csv"

# Run evaluation script
python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluate_gi_masks.py \
    --pred_dir "$PRED_DIR" \
    --gt_dir "$GT_DIR" \
    --out_csv "$OUT_CSV" \
    --num_workers 32

