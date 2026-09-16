#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --job-name=p10_reg
#SBATCH --array=0-7
#SBATCH --output=logs/p10_reg_%A_%a.out
#SBATCH --error=logs/p10_reg_%A_%a.err

# ---------------------------------------------------------------------------
# Project 10 Phase 1 Step 1.2: 3D Deformable Multi-Phase CT Co-Registration
# ---------------------------------------------------------------------------
# SLURM array job: 8 workers sharing the multiphase_pairs.csv manifest.
# Each task processes its shard (chunk_idx = SLURM_ARRAY_TASK_ID).
# Total pairs ~438; each worker handles ~55 pairs.
#
# SimpleITK pipeline: Rigid -> Affine -> B-Spline (Mattes MI).
# Saves registered NIfTI + displacement vector field (DVF) per pair.
#
# Outputs:
#   results/project10/registered_volumes/<pair_id>_reg.nii.gz
#   results/project10/registered_volumes/<pair_id>_dvf.nii.gz
#   results/project10/manifests/registration_log.csv
# ---------------------------------------------------------------------------

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/project10/registered_volumes

eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
TOTAL_CHUNKS=8

echo "========================================="
echo "Job ID:         $SLURM_ARRAY_JOB_ID"
echo "Array Task:     $SLURM_ARRAY_TASK_ID / $TOTAL_CHUNKS"
echo "Node:           $SLURMD_NODENAME"
echo "Started:        $(date)"
echo "========================================="

python -u code/project10/datasets/register_multiphase_pairs.py \
    --pairs        results/project10/manifests/multiphase_pairs.csv \
    --out_dir      results/project10/registered_volumes \
    --log_csv      results/project10/manifests/registration_log.csv \
    --chunk_idx    $SLURM_ARRAY_TASK_ID \
    --total_chunks $TOTAL_CHUNKS \
    --nthreads     8 \
    --skip_existing

echo ""
echo "Chunk $SLURM_ARRAY_TASK_ID finished at: $(date)"
