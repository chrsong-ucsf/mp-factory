#!/bin/bash
#SBATCH --job-name=swin_predict
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/swin_predict_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/swin_predict_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions

python -u /mnt/scratch/user/chrsong/mp-factory/code/prediction/predict_swin_unetr.py \
    --data_dir "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox" \
    --model_path "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_models/fold_0/best_swin_unetr_gi.pt" \
    --out_dir "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions" \
    --roi_size 96 96 96 \
    --sw_batch_size 4 \
    --overlap 0.5 \
    --array_id "${SLURM_ARRAY_TASK_ID}" \
    --save_probs \
    --total_workers 4
