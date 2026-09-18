#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=p10_stage2_synth
#SBATCH --output=logs/p10_stage2_synth_%j.out
#SBATCH --error=logs/p10_stage2_synth_%j.err

# ---------------------------------------------------------------------------
# Project 10 Phase 3 Stage 2: Anatomy-Conditional Contrast Synthesis Training
# ---------------------------------------------------------------------------
# Trains the SPADE Latent Diffusion Synthesis Model (AnatomyConditionalSynthesisNet)
# to synthesize target contrast phase latents from NCCT source latents,
# conditioned on the Project 9 5-organ GI segmentation mask.
#
# Inputs:
#   - Registered multi-phase pairs: results/project10/manifests/registration_log.csv (~638 pairs)
#   - Stage 1 compressed latent autoencoder: results/project10/checkpoints/stage1_ae/best_stage1_ae.pt
#   - TotalSegmentator organ masks: results/totalseg_masks/
#
# Outputs:
#   - results/project10/checkpoints/stage2_synthesis/best_synthesis_net.pt
#   - results/project10/checkpoints/stage2_synthesis/last_synthesis_net.pt
#   - results/project10/checkpoints/stage2_synthesis/stage2_training_log.csv
# ---------------------------------------------------------------------------

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/project10/checkpoints/stage2_synthesis

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/scratch/user/chrsong/mp-factory:${PYTHONPATH}"
export OMP_NUM_THREADS=16

echo "========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURMD_NODENAME"
echo "GPU Device:    $CUDA_VISIBLE_DEVICES"
echo "Started At:    $(date)"
echo "========================================="

AE_CKPT="results/project10/checkpoints/stage1_ae/best_stage1_ae.pt"
if [ ! -f "$AE_CKPT" ]; then
    echo "Warning: $AE_CKPT not found yet. Falling back to last_stage1_ae.pt if available."
    AE_CKPT="results/project10/checkpoints/stage1_ae/last_stage1_ae.pt"
fi

RESUME_ARG=""
LAST_SYNTH_CKPT="results/project10/checkpoints/stage2_synthesis/last_synthesis_net.pt"
if [ -f "$LAST_SYNTH_CKPT" ]; then
    RESUME_ARG="--resume $LAST_SYNTH_CKPT"
    echo "Resuming synthesis training from: $LAST_SYNTH_CKPT"
fi

python -u code/project10/training/train_stage2_synthesis.py \
    --registration_csv results/project10/manifests/registration_log.csv \
    --base_dir         /mnt/scratch/user/chrsong/mp-factory \
    --mask_dir         results/totalseg_masks \
    --ae_checkpoint    "$AE_CKPT" \
    --out_dir          results/project10/checkpoints/stage2_synthesis \
    --epochs           100 \
    --batch_size       4 \
    --lr               1e-4 \
    --timesteps        1000 \
    --val_interval     5 \
    $RESUME_ARG

echo ""
echo "Stage 2 Synthesis Training finished at: $(date)"
echo "Saved checkpoints:"
ls -lh results/project10/checkpoints/stage2_synthesis/
