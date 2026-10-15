#!/bin/bash
#SBATCH --job-name=mednext_p2_pred
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_p2_pred_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_p2_pred_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3

# Load environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/mednext_phase2_predictions

echo "[MedNeXt-B Phase 2 Pred] Starting Worker ${SLURM_ARRAY_TASK_ID}/4 on GPU: ${CUDA_VISIBLE_DEVICES}"

python -u /mnt/scratch/user/chrsong/mp-factory/code/prediction/predict_mednext.py \
    --data_dir   "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox" \
    --model_path "/mnt/scratch/user/chrsong/mp-factory/results/mednext_phase2/fold_0/best_mednext_phase2.pt" \
    --out_dir    "/mnt/scratch/user/chrsong/mp-factory/results/mednext_phase2_predictions" \
    --model_id   B \
    --kernel_size 3 \
    --roi_size   96 96 96 \
    --sw_batch_size 4 \
    --overlap    0.5 \
    --array_id   "${SLURM_ARRAY_TASK_ID}" \
    --total_workers 4 \
    --save_probs
