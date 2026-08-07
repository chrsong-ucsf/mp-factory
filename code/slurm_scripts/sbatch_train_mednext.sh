#!/bin/bash
# =============================================================================
# Task B.3 — MedNeXt-B multi-GPU training launcher (nvidia_l40s)
#
# Submit:
#   sbatch code/slurm_scripts/sbatch_train_mednext.sh
#
# Common overrides (environment variables, no need to edit this file):
#   EPOCHS=200 ALPHA=3.0 sbatch code/slurm_scripts/sbatch_train_mednext.sh
#   LOSS_TYPE=dice_ce sbatch code/slurm_scripts/sbatch_train_mednext.sh
#
# GPU count is a Slurm directive, not an env var; change it with a CLI override:
#   sbatch --gres=gpu:nvidia_l40s:4 code/slurm_scripts/sbatch_train_mednext.sh
#
# Runs a 4-fold cross-validation array; each array task trains one fold on the
# allocated L40S GPUs via torch DataParallel.
# =============================================================================
#SBATCH --job-name=mednext_b
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_b_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_b_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-3

set -euo pipefail

# --- Fail fast on unset cluster paths -----------------------------------------
SCRATCH_ROOT="${SCRATCH_ROOT:-/mnt/scratch/user/chrsong/mp-factory}"
DATA_DIR="${DATA_DIR:-${SCRATCH_ROOT}/CancerVerse_dbox}"
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/logs}"
CODE_DIR="${CODE_DIR:-${SCRATCH_ROOT}/code}"
ENV_PREFIX="${ENV_PREFIX:-/mnt/scratch/user/chrsong/envs/mp-factory}"

# --- Hyperparameters (override via environment) -------------------------------
MODEL_ID="${MODEL_ID:-B}"
KERNEL_SIZE="${KERNEL_SIZE:-3}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-4}"
VAL_INTERVAL="${VAL_INTERVAL:-5}"
NUM_FOLDS="${NUM_FOLDS:-4}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-50}"
LOSS_TYPE="${LOSS_TYPE:-asymmetric}"
ALPHA="${ALPHA:-2.0}"          # false-negative weight (alpha > beta boosts recall)
BETA="${BETA:-1.0}"            # false-positive weight
ELASTIC_PROB="${ELASTIC_PROB:-0.15}"
GOLD_STANDARD_DIR="${GOLD_STANDARD_DIR:-}"

FOLD="${SLURM_ARRAY_TASK_ID:-0}"
OUT_DIR="${SCRATCH_ROOT}/results/mednext_b/fold_${FOLD}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

# --- Environment --------------------------------------------------------------
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate "${ENV_PREFIX}"

# MedNeXt is not on PyPI; install once if missing (no-op when present).
python -c "import nnunet_mednext" 2>/dev/null \
  || pip install --quiet git+https://github.com/MIC-DKFZ/MedNeXt.git

# Make the vault-root src/ importable so AsymmetricPDCELoss resolves.
export PYTHONPATH="${SCRATCH_ROOT}:${PYTHONPATH:-}"

# Multi-GPU / allocator tuning
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --- Provenance ---------------------------------------------------------------
echo "=================================================================="
echo "Job              : ${SLURM_JOB_ID:-local} (array task ${FOLD})"
echo "Node             : $(hostname)"
echo "Fold             : ${FOLD} / ${NUM_FOLDS}"
echo "Loss             : ${LOSS_TYPE} (alpha=${ALPHA}, beta=${BETA})"
echo "Elastic prob     : ${ELASTIC_PROB}"
echo "Output           : ${OUT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "=================================================================="

# Note: declared with a placeholder-free pattern and expanded with :- so that an
# empty array does not trip `set -u` under bash 3.x.
GOLD_ARGS=()
if [[ -n "${GOLD_STANDARD_DIR}" ]]; then
  GOLD_ARGS=(--gold_standard_dir "${GOLD_STANDARD_DIR}")
fi

set +e
srun python -u "${CODE_DIR}/training/train_mednext.py" \
    --data_dir        "${DATA_DIR}" \
    --out_dir         "${OUT_DIR}" \
    --model_id        "${MODEL_ID}" \
    --kernel_size     "${KERNEL_SIZE}" \
    --fold            "${FOLD}" \
    --num_folds       "${NUM_FOLDS}" \
    --epochs          "${EPOCHS}" \
    --batch_size      "${BATCH_SIZE}" \
    --lr              "${LR}" \
    --val_interval    "${VAL_INTERVAL}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" \
    --loss_type       "${LOSS_TYPE}" \
    --alpha           "${ALPHA}" \
    --beta            "${BETA}" \
    --elastic_prob    "${ELASTIC_PROB}" \
    ${GOLD_ARGS[@]+"${GOLD_ARGS[@]}"}
STATUS=$?
set -e

echo "Fold ${FOLD} finished with exit code ${STATUS}"
exit "${STATUS}"
