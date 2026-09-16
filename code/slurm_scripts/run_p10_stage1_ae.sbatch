#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=p10_stage1_ae
#SBATCH --output=logs/p10_stage1_ae_%j.out
#SBATCH --error=logs/p10_stage1_ae_%j.err

# ---------------------------------------------------------------------------
# Project 10 Phase 3 Stage 1: Train 3D CT Compression Autoencoder
# ---------------------------------------------------------------------------
# 4-GPU single-node DDP training of the VQ-VAE 3D compression autoencoder.
# Consumes all NCCT + post-contrast CT volumes from the multiphase manifest.
#
# Outputs:
#   results/project10/checkpoints/stage1_ae/best_stage1_ae.pt
#   results/project10/checkpoints/stage1_ae/last_stage1_ae.pt
#   results/project10/checkpoints/stage1_ae/training_log.csv
# ---------------------------------------------------------------------------

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/project10/checkpoints/stage1_ae

eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MASTER_ADDR=localhost
export MASTER_PORT=29600

echo "========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "========================================="

# Resume from last checkpoint if available
CKPT=""
LAST_CKPT="results/project10/checkpoints/stage1_ae/last_stage1_ae.pt"
if [ -f "$LAST_CKPT" ]; then
    CKPT="--checkpoint $LAST_CKPT"
    echo "Resuming from: $LAST_CKPT"
fi

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    code/project10/training/train_stage1_autoencoder.py \
        --pairs     results/project10/manifests/multiphase_pairs.csv \
        --out_dir   results/project10/checkpoints/stage1_ae \
        --epochs    100 \
        --batch_size 2 \
        --lr        1e-4 \
        --val_interval 5 \
        --target_size  128 \
        --spacing_mm   1.5 \
        $CKPT

echo ""
echo "Finished at: $(date)"
echo "Checkpoint files:"
ls -lh results/project10/checkpoints/stage1_ae/
