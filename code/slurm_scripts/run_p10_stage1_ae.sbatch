#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=p10_stage1_ae
#SBATCH --output=logs/p10_stage1_ae_%j.out
#SBATCH --error=logs/p10_stage1_ae_%j.err

# ---------------------------------------------------------------------------
# Project 10 Phase 3 Stage 1: Train 3D CT Compression Autoencoder (Single GPU)
# ---------------------------------------------------------------------------
# Trains the 3D VQ-VAE compression autoencoder (8x spatial compression, 4 channels)
# Consumes all discovered CancerVerse multi-phase scans from multiphase_pairs.csv.
#
# Outputs:
#   results/project10/checkpoints/stage1_ae/best_stage1_ae.pt
#   results/project10/checkpoints/stage1_ae/last_stage1_ae.pt
#   results/project10/checkpoints/stage1_ae/training_log.csv
# ---------------------------------------------------------------------------

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/project10/checkpoints/stage1_ae

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/scratch/user/chrsong/mp-factory:${PYTHONPATH}"
export OMP_NUM_THREADS=16

echo "========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "========================================="

# Resume from last checkpoint if available
CKPT=""
LAST_CKPT="results/project10/checkpoints/stage1_ae/last_stage1_ae.pt"
if [ -f "$LAST_CKPT" ]; then
    CKPT="--checkpoint $LAST_CKPT"
    echo "Resuming from: $LAST_CKPT"
fi

python -u code/project10/training/train_stage1_autoencoder.py \
    --pairs     results/project10/manifests/multiphase_pairs.csv \
    --out_dir   results/project10/checkpoints/stage1_ae \
    --epochs    100 \
    --batch_size 4 \
    --lr        1e-4 \
    --val_interval 5 \
    --target_size  128 \
    --spacing_mm   1.5 \
    $CKPT

echo ""
echo "Finished at: $(date)"
echo "Checkpoint files:"
ls -lh results/project10/checkpoints/stage1_ae/
