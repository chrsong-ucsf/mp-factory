#!/usr/bin/env bash
# submit_scaling_sweep.sh
#
# Scaling sweep orchestrator for mp-factory Phase 2.
# Submits one SLURM job per (cohort_size N, seed) pair, each of which
# trains a full MedNeXt-B Phase 2 model on a reproducible N-case subsample
# of the cleansed pool (CLEAN_HIGH_CONFIDENCE + WEAK_COARSE w/ ignore masking).
#
# Prerequisites:
#   1. hard_threshold_autolabel.py must have been run (results/autolabel_out/ populated).
#   2. The Phase 1 ensemble must have completed (results/ensemble_audit_summary.csv exists).
#   3. At least one Phase 1 checkpoint exists per fold for warm-start.
#
# Usage:
#   bash code/slurm_scripts/submit_scaling_sweep.sh [--dry-run]
#
# Pass --dry-run to print sbatch commands without submitting.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRATCH="/mnt/scratch/user/chrsong/mp-factory"
CODE_DIR="${SCRATCH}/code"
DATA_DIR="${SCRATCH}/CancerVerse_dbox"
AUDIT_CSV="${SCRATCH}/results/ensemble_audit_summary.csv"
ENSEMBLE_OUT="${SCRATCH}/results/ensemble_out"
AUTOLABEL_DIR="${SCRATCH}/results/autolabel_out"
GOLD_STD_DIR="${SCRATCH}/JHU_data_radiologist_corrected"
PHASE1_CKPT_DIR="${SCRATCH}/results/mednext_models"
LOG_DIR="${SCRATCH}/logs/scaling_sweep"
SWEEP_OUT="${SCRATCH}/results/scaling_sweep"

# Sweep parameters
COHORT_SIZES=(4 10 15 20 30 50)
SEEDS=(42 123 456)
EPOCHS=150
MODEL_ID="B"
KERNEL_SIZE=3
LR="0.0001"
LOSS_TYPE="asymmetric"
ALPHA="2.0"
BETA="1.0"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[submit_scaling_sweep] DRY RUN — commands will be printed but not submitted."
fi

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "=================================================================="
echo "  mp-factory Phase 2 Scaling Sweep Orchestrator"
echo "=================================================================="

if [[ ! -f "${AUDIT_CSV}" ]]; then
    echo "ERROR: Audit CSV not found: ${AUDIT_CSV}"
    echo "  Run the Phase 1 ensemble evaluation pipeline first."
    exit 1
fi

AUTOLABEL_COUNT=$(find "${AUTOLABEL_DIR}" -name "*_autolabel.nii.gz" 2>/dev/null | wc -l || echo 0)
if [[ "${AUTOLABEL_COUNT}" -lt 1 ]]; then
    echo "ERROR: No *_autolabel.nii.gz files found in: ${AUTOLABEL_DIR}"
    echo "  Run hard_threshold_autolabel.py first:"
    echo "    python ${CODE_DIR}/evaluation/hard_threshold_autolabel.py \\"
    echo "        --data_dir      ${DATA_DIR} \\"
    echo "        --consensus_dir ${ENSEMBLE_OUT} \\"
    echo "        --audit_csv     ${AUDIT_CSV} \\"
    echo "        --out_dir       ${AUTOLABEL_DIR} \\"
    echo "        --num_workers   16"
    exit 1
fi
echo "  Autolabels found: ${AUTOLABEL_COUNT} files in ${AUTOLABEL_DIR}"

mkdir -p "${LOG_DIR}"
mkdir -p "${SWEEP_OUT}"

# ---------------------------------------------------------------------------
# Submit one job per (N, seed)
# ---------------------------------------------------------------------------
SUBMITTED_JOBS=()

for N in "${COHORT_SIZES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        N_PADDED=$(printf "%04d" "${N}")
        OUT_DIR="${SWEEP_OUT}/N${N_PADDED}/seed${SEED}"
        mkdir -p "${OUT_DIR}"

        # Best Phase 1 checkpoint (fold 0 as default; sweep does not use k-fold)
        PHASE1_CKPT="${PHASE1_CKPT_DIR}/fold_0/best_mednext_gi.pt"

        # Build the inline training command
        TRAIN_CMD="set -euo pipefail
source /etc/profile.d/modules.sh || true
module load CBI miniforge3/26.3.2-3 2>/dev/null || true
eval \"\$(mamba shell hook --shell bash)\" && mamba activate /mnt/scratch/user/chrsong/envs/mp-factory
pip install --quiet git+https://github.com/MIC-DKFZ/MedNeXt.git 2>/dev/null || true
export PYTHONUNBUFFERED=1
echo \"[sweep N=${N} seed=${SEED}] Starting on \$(hostname)\"
python -u ${CODE_DIR}/training/train_mednext_phase2.py \
    --data_dir          '${DATA_DIR}'     \
    --audit_csv         '${AUDIT_CSV}'    \
    --ensemble_out_dir  '${ENSEMBLE_OUT}' \
    --autolabel_dir     '${AUTOLABEL_DIR}' \
    --gold_standard_dir '${GOLD_STD_DIR}' \
    --out_dir           '${OUT_DIR}'      \
    --model_id          ${MODEL_ID}       \
    --kernel_size       ${KERNEL_SIZE}    \
    --epochs            ${EPOCHS}         \
    --batch_size        1                 \
    --lr                ${LR}             \
    --loss_type         ${LOSS_TYPE}      \
    --alpha             ${ALPHA}          \
    --beta              ${BETA}           \
    --val_interval      5                 \
    --max_val_samples   50                \
    --max_cases         ${N}              \
    --seed              ${SEED}           \
    --use_weak                            \
    --pretrained_ckpt   '${PHASE1_CKPT}'
echo \"[sweep N=${N} seed=${SEED}] Done.\""

        if [[ "${DRY_RUN}" == "true" ]]; then
            echo ""
            echo "  [DRY RUN] Would submit: N=${N}, seed=${SEED}"
            echo "  Output: ${OUT_DIR}"
            echo "  sbatch --job-name=sweep_N${N_PADDED}_s${SEED} --time=48:00:00 ..."
        else
            JOB_ID=$(sbatch \
                --job-name="sweep_N${N_PADDED}_s${SEED}" \
                --output="${LOG_DIR}/sweep_N${N_PADDED}_s${SEED}_%j.out" \
                --error="${LOG_DIR}/sweep_N${N_PADDED}_s${SEED}_%j.err" \
                --partition=gpu \
                --gres="gpu:nvidia_l40s:1" \
                --cpus-per-task=8 \
                --mem=64G \
                --time=48:00:00 \
                --wrap="${TRAIN_CMD}" \
                --parsable)
            SUBMITTED_JOBS+=("N=${N_PADDED},seed=${SEED} -> JobID=${JOB_ID}")
            echo "  Submitted: N=${N_PADDED}, seed=${SEED} -> JobID=${JOB_ID}"
        fi
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_JOBS=$(( ${#COHORT_SIZES[@]} * ${#SEEDS[@]} ))
echo ""
echo "=================================================================="
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "  DRY RUN complete. ${TOTAL_JOBS} jobs would be submitted."
else
    echo "  Submitted ${TOTAL_JOBS} SLURM jobs."
    echo ""
    echo "  Monitor progress:"
    echo "    squeue -u \$(whoami) | grep sweep"
    echo ""
    for JOB in "${SUBMITTED_JOBS[@]}"; do
        echo "    ${JOB}"
    done
fi
echo ""
echo "  Results written to: ${SWEEP_OUT}/N{N}/seed{seed}/"
echo "  Each run saves: best_mednext_phase2.pt, last_mednext_phase2.pt"
echo "=================================================================="
