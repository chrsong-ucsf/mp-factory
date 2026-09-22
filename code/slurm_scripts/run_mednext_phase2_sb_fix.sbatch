#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=mednext_p2_sb_fix
#SBATCH --output=logs/mednext_p2_sb_fix_%j.out
#SBATCH --error=logs/mednext_p2_sb_fix_%j.err

# ---------------------------------------------------------------------------
# Project 9 (Arm A): MedNeXt-B Phase 2 Student Distillation with Small Bowel Fix
# ---------------------------------------------------------------------------
# - Single GPU (L40S / A100 / RTX 6000 Ada 48GB)
# - 16 CPU cores with 8 DataLoader workers (pin_memory + persistent_workers)
# - Corrected Label Remapping: TotalSegmentator label 20 -> label 3 (small bowel)
# - Asymmetric Partial CE + Dice Loss (alpha=2.0 for recall optimization)
# - Evaluates on CLEAN (1,417) + WEAK_COARSE (177 with ignore_index=255)
# ---------------------------------------------------------------------------

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/mednext_phase2_sb_fix

# Environment setup
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

# Install / update MedNeXt if needed
pip install --quiet git+https://github.com/MIC-DKFZ/MedNeXt.git

export PYTHONUNBUFFERED=1
export PYTHONPATH="/mnt/scratch/user/chrsong/mp-factory:${PYTHONPATH}"
export OMP_NUM_THREADS=16

# Determine Checkpoint and Start Epoch
OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/mednext_phase2_sb_fix"
LAST_CKPT="${OUT_DIR}/last_mednext_phase2.pt"
PHASE1_CKPT="/mnt/scratch/user/chrsong/mp-factory/results/mednext_models/fold_0/best_mednext_gi.pt"

CKPT_ARG=""
START_EPOCH=1

if [ -f "$LAST_CKPT" ]; then
    echo "[Resumption] Found existing checkpoint: $LAST_CKPT"
    CKPT_ARG="--pretrained_ckpt ${LAST_CKPT}"
    # Extract last finished epoch from previous logs that actually contain epoch lines
    # Use grep -l to find files containing epoch lines, then sort by modification time
    PREV_LOG=$(for f in logs/mednext_p2_sb_fix_*.out /home/remote/chrsong/logs/mednext_p2_sb_fix_*.out; do
        [ -f "$f" ] && grep -qE "Epoch \[[0-9]{3}/[0-9]{3}\]" "$f" 2>/dev/null && echo "$f"
    done | xargs ls -t 2>/dev/null | head -n 1)
    if [ -n "$PREV_LOG" ]; then
        LAST_EPOCH=$(grep -oE "Epoch \[[0-9]{3}/[0-9]{3}\]" "$PREV_LOG" | tail -n 1 | grep -oE "[0-9]{3}" | head -n 1 | sed 's/^0*//')
        if [ -n "$LAST_EPOCH" ]; then
            START_EPOCH=$((LAST_EPOCH + 1))
        fi
    fi
    echo "[Resumption] Resuming from epoch $START_EPOCH (prev log: ${PREV_LOG:-none})"
elif [ -f "$PHASE1_CKPT" ]; then
    echo "[Warm-start] Starting from Phase 1 checkpoint: $PHASE1_CKPT"
    CKPT_ARG="--pretrained_ckpt ${PHASE1_CKPT}"
else
    echo "[Cold-start] Training from scratch (random initialization)"
fi

echo "=========================================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Node:             $SLURMD_NODENAME"
echo "GPU Allocated:    $CUDA_VISIBLE_DEVICES"
echo "CPUs Allocated:   $SLURM_CPUS_PER_TASK"
echo "Output Dir:       $OUT_DIR"
echo "Starting Epoch:   $START_EPOCH"
echo "Started At:       $(date)"
echo "=========================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/training/train_mednext_phase2.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --audit_csv /mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv \
    --ensemble_out_dir /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out \
    --autolabel_dir /mnt/scratch/user/chrsong/mp-factory/results/autolabel_out \
    --gold_standard_dir /mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected \
    --out_dir "${OUT_DIR}" \
    --model_id "B" \
    --kernel_size 3 \
    --epochs 150 \
    --start_epoch ${START_EPOCH} \
    --batch_size 2 \
    --num_workers 8 \
    --lr 1e-4 \
    --val_interval 5 \
    --use_weak \
    --loss_type asymmetric \
    --alpha 2.0 \
    --beta 1.0 \
    ${CKPT_ARG}

echo "=========================================================="
echo "Training finished at: $(date)"
echo "=========================================================="
