#!/bin/bash
#SBATCH --job-name=p873_pred
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_pool873_predict_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_pool873_predict_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/mednext_pool873_predictions

# Select best available fold checkpoint
CKPT="/mnt/scratch/user/chrsong/mp-factory/results/mednext_pool873/fold_0/best_mednext_phase2.pt"
if [ ! -f "$CKPT" ]; then
    CKPT=$(find /mnt/scratch/user/chrsong/mp-factory/results/mednext_pool873/ -name "best_*.pt" | head -n 1)
fi

echo "====================================================================="
echo "  Executing MedNeXt-Pool873 Batch Inference (Worker ${SLURM_ARRAY_TASK_ID}/4)"
echo "  Date: $(date)"
echo "  Checkpoint: ${CKPT}"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/prediction/predict_mednext.py \
    --data_dir  "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox" \
    --model_path "${CKPT}" \
    --out_dir   "/mnt/scratch/user/chrsong/mp-factory/results/mednext_pool873_predictions" \
    --model_id B \
    --kernel_size 3 \
    --roi_size 96 96 96 \
    --sw_batch_size 4 \
    --overlap 0.5 \
    --array_id "${SLURM_ARRAY_TASK_ID}" \
    --save_probs \
    --total_workers 4

echo "====================================================================="
echo "  Worker ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "====================================================================="
