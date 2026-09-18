#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=p10_stage1_ablation
#SBATCH --output=logs/p10_stage1_ablation_%j.out
#SBATCH --error=logs/p10_stage1_ablation_%j.err

# ---------------------------------------------------------------------------
# Project 10 Phase 3 Stage 1: Continuous Latent Autoencoder Ablation (--no_vq)
# ---------------------------------------------------------------------------
# Architectural ablation against running job 1841972:
# Compares discrete VQ-VAE (512 codebook embeddings) vs continuous latent space
# to evaluate reconstruction fidelity on high-frequency vascular contrast peaks.
#
# Outputs:
#   results/project10/checkpoints/stage1_ae_ablation/best_stage1_ae.pt
#   results/project10/checkpoints/stage1_ae_ablation/last_stage1_ae.pt
#   results/project10/checkpoints/stage1_ae_ablation/training_log.csv
# ---------------------------------------------------------------------------

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/project10/checkpoints/stage1_ae_ablation

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/scratch/user/chrsong/mp-factory:${PYTHONPATH}"
export OMP_NUM_THREADS=16

echo "========================================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "========================================="

python -u code/project10/training/train_stage1_autoencoder.py \
    --pairs        results/project10/manifests/multiphase_pairs.csv \
    --out_dir      results/project10/checkpoints/stage1_ae_ablation \
    --epochs       100 \
    --batch_size   4 \
    --lr           1e-4 \
    --val_interval 5 \
    --target_size  128 \
    --spacing_mm   1.5 \
    --no_vq

echo ""
echo "Finished at: $(date)"
echo "Checkpoint files:"
ls -lh results/project10/checkpoints/stage1_ae_ablation/
