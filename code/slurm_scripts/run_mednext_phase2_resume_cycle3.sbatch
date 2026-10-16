#!/bin/bash
#SBATCH --job-name=mednext_p2_c3
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_p2_c3_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_p2_c3_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00

# Load environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

FOLD=${FOLD:-${SLURM_ARRAY_TASK_ID:-0}}

# Paths
DATA_DIR="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"
AUDIT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv"
ENSEMBLE_OUT="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out"
AUTOLABEL_DIR="/mnt/scratch/user/chrsong/mp-factory/results/autolabel_out"
GOLD_STD_DIR="/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected"
OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/mednext_phase2/fold_${FOLD}"

mkdir -p "${OUT_DIR}"
mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs

# Auto-detect start epoch from previous log if available
PREV_LOG="/mnt/scratch/user/chrsong/mp-factory/logs/mednext_p2_res_2271424_${FOLD}.out"
if [ -f "${PREV_LOG}" ]; then
    LAST_EPOCH=$(grep -o 'Epoch \[[0-9]*' "${PREV_LOG}" 2>/dev/null | tail -n 1 | awk -F'[' '{print $2}' | sed 's/^0*//')
    if [ -n "${LAST_EPOCH}" ]; then
        START_EPOCH=$((LAST_EPOCH + 1))
    else
        START_EPOCH=50
    fi
else
    START_EPOCH=50
fi

echo "[Phase 2 Cycle 3] Resuming MedNeXt-B Fold ${FOLD} from Epoch ${START_EPOCH} on GPU: ${CUDA_VISIBLE_DEVICES}"

python -u /mnt/scratch/user/chrsong/mp-factory/code/training/train_mednext_phase2.py \
    --data_dir          "${DATA_DIR}" \
    --audit_csv         "${AUDIT_CSV}" \
    --ensemble_out_dir  "${ENSEMBLE_OUT}" \
    --autolabel_dir     "${AUTOLABEL_DIR}" \
    --gold_standard_dir "${GOLD_STD_DIR}" \
    --out_dir           "${OUT_DIR}" \
    --model_id          "B" \
    --kernel_size       3 \
    --fold              "${FOLD}" \
    --num_folds         4 \
    --epochs            150 \
    --start_epoch       "${START_EPOCH}" \
    --batch_size        1 \
    --lr                0.0001 \
    --val_interval      5 \
    --use_weak
