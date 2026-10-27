#!/bin/bash
#SBATCH --job-name=swin_p3_pred
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/swin_p3_predict_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/swin_p3_predict_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_phase3_predictions

echo "====================================================================="
echo "  Executing Phase 3 Swin-UNETR Batch Inference (Worker ${SLURM_ARRAY_TASK_ID}/4)"
echo "  Date: $(date)"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/prediction/predict_swin_unetr.py \
    --data_dir "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox" \
    --model_path "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_phase3/fold_0/last_swin_unetr_phase2.pt" \
    --out_dir "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_phase3_predictions" \
    --roi_size 96 96 96 \
    --sw_batch_size 4 \
    --overlap 0.5 \
    --array_id "${SLURM_ARRAY_TASK_ID}" \
    --save_probs \
    --total_workers 4
