#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=train_full_anchor_4gpu
#SBATCH --output=logs/train_full_anchor_4gpu_%j.out
#SBATCH --error=logs/train_full_anchor_4gpu_%j.err

# ---------------------------------------------------------------------------
# Phase 2 MedNeXt-B Anchor Training: Multi-Node 4-GPU Distributed (DDP)
# ---------------------------------------------------------------------------
# - 2 Nodes x 2 GPUs per node = 4 GPUs total (L40S / RTX 6000 Ada 48GB)
# - 32 CPUs per node = 64 CPUs total
# - 14 DataLoader worker threads per GPU process = 56 CPU workers total
#   (completely eliminates 3D NIfTI loading & Spacingd resampling bottleneck)
# - PyTorch DistributedDataParallel (DDP) via torchrun + NCCL
# ---------------------------------------------------------------------------

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/anchor_full_n1594

# Environment activation
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
export NCCL_TIMEOUT=7200
ulimit -n 65535 2>/dev/null || true

# Multi-Node DDP Rendezvous Configuration
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))

# Auto-detect start epoch and checkpoint for resumption
CKPT="/mnt/scratch/user/chrsong/mp-factory/results/anchor_full_n1594/last_mednext_phase2.pt"
START_EPOCH=1

if [ -f "$CKPT" ]; then
    if [ -n "$RESUME_START_EPOCH" ]; then
        START_EPOCH=$RESUME_START_EPOCH
    else
        # Auto-detect latest completed epoch from previous anchor training log (excluding this job)
        PREV_LOG=$(ls -t logs/train_full_anchor_*.out logs/train_full_anchor_4gpu_*.out 2>/dev/null | grep -v "${SLURM_JOB_ID}" | head -n 1)
        if [ -n "$PREV_LOG" ]; then
            LAST_EPOCH=$(grep -oE "Epoch \[[0-9]{3}/[0-9]{3}\]" "$PREV_LOG" | tail -n 1 | grep -oE "[0-9]{3}" | head -n 1 | sed 's/^0*//')
            if [ -n "$LAST_EPOCH" ]; then
                START_EPOCH=$((LAST_EPOCH + 1))
            fi
        fi
    fi
    echo "[Resumption] Found checkpoint $CKPT. Detected prev log: $PREV_LOG (last completed epoch: ${LAST_EPOCH:-unknown}). Starting at epoch $START_EPOCH."
else
    CKPT="/mnt/scratch/user/chrsong/mp-factory/results/mednext_models/fold_0/best_mednext_gi.pt"
    echo "[Warm-start] Starting from Phase 1 checkpoint $CKPT (epoch 1)."
fi

echo "=========================================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Nodes allocated:  $SLURM_JOB_NODELIST (Count: $SLURM_NNODES)"
echo "Master node:      $MASTER_ADDR:$MASTER_PORT"
echo "GPUs per node:    2 (Total GPUs: 4)"
echo "CPUs per node:    $SLURM_CPUS_PER_TASK (Total CPUs: 64)"
echo "DataLoader CPUs:  14 workers/process (56 concurrent workers)"
echo "Resuming from:    Epoch $START_EPOCH -> 150"
echo "=========================================================="

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    code/training/train_mednext_phase2.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --ensemble_out_dir /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out \
    --audit_csv /mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv \
    --autolabel_dir /mnt/scratch/user/chrsong/mp-factory/results/autolabel_out \
    --pretrained_ckpt "$CKPT" \
    --out_dir /mnt/scratch/user/chrsong/mp-factory/results/anchor_full_n1594 \
    --epochs 150 \
    --start_epoch $START_EPOCH \
    --batch_size 2 \
    --num_workers 4 \
    --lr 2e-4 \
    --max_val_samples 25 \
    --use_weak
