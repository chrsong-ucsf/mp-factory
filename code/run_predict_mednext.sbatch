#!/bin/bash
#SBATCH --job-name=mednext_pred
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_predict_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_predict_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions

python -u /mnt/scratch/user/chrsong/mp-factory/code/predict_mednext.py \
    --data_dir  "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox" \
    --model_path "/mnt/scratch/user/chrsong/mp-factory/results/mednext_models/fold_0/best_mednext_gi.pt" \
    --out_dir   "/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions" \
    --model_id B \
    --kernel_size 3 \
    --roi_size 96 96 96 \
    --sw_batch_size 4 \
    --overlap 0.5 \
    --array_id "${SLURM_ARRAY_TASK_ID}" \
    --total_workers 4
