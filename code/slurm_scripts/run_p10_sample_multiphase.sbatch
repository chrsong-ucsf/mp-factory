#!/bin/bash
#SBATCH --job-name=p10_diff_sample
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/p10_diff_sample_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/p10_diff_sample_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-7

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_DIR="/mnt/scratch/user/chrsong/mp-factory"
LOGS_DIR="${BASE_DIR}/logs"
OUT_DIR="${BASE_DIR}/results/project10/synthetic_volumes"

mkdir -p "${LOGS_DIR}"
mkdir -p "${OUT_DIR}"

echo "====================================================================="
echo "  Project 10 Multiphase Diffusion Sampling Inference"
echo "  Worker: ${SLURM_ARRAY_TASK_ID} / 8 GPUs"
echo "  Date:   $(date)"
echo "  Output: ${OUT_DIR}"
echo "====================================================================="

python -u "${BASE_DIR}/code/project10/evaluation/sample_multiphase_diffusion.py" \
    --registered_dir "${BASE_DIR}/results/project10/registered_volumes" \
    --ae_checkpoint "${BASE_DIR}/results/project10/checkpoints/stage1_ae/best_stage1_ae.pt" \
    --diffusion_checkpoint "${BASE_DIR}/results/project10/checkpoints/stage2_synthesis/best_synthesis_net.pt" \
    --out_dir "${OUT_DIR}" \
    --totalseg_task total \
    --num_steps 50 \
    --array_id "${SLURM_ARRAY_TASK_ID}" \
    --total_workers 8

echo "====================================================================="
echo "  Worker ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "====================================================================="
