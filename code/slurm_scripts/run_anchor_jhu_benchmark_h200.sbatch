#!/bin/bash
#SBATCH --job-name=jhu_anchor_eval
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_anchor_eval_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_anchor_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# -----------------------------------------------------------------------------
# End-to-End Pipeline: Inference with Full Anchor Model + JHU Radiologist Benchmark Eval
# Target Partition: large_gpu (H200 / H200 NVL)
# -----------------------------------------------------------------------------

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

BASE_DIR="/mnt/scratch/user/chrsong/mp-factory"
LOGS_DIR="${BASE_DIR}/logs"
RESULTS_DIR="${BASE_DIR}/results"
PRED_DIR="${RESULTS_DIR}/anchor_full_predictions"
CHECKPOINT="${RESULTS_DIR}/anchor_full_n1594/best_mednext_phase2.pt"
DATA_DIR="${BASE_DIR}/CancerVerse_dbox"
JHU_DIR="${BASE_DIR}/JHU_data_radiologist_corrected"

mkdir -p "${LOGS_DIR}"
mkdir -p "${PRED_DIR}"

echo "====================================================================="
echo "  Executing Anchor Model Inference + JHU Radiologist Benchmark Eval"
echo "  Node:         ${SLURMD_NODENAME}"
echo "  GPU:          NVIDIA H200 (large_gpu partition)"
echo "  Checkpoint:   ${CHECKPOINT}"
echo "  Started at:   $(date)"
echo "====================================================================="

# Step 1: Run inference on the BDMAP evaluation cohort using the trained anchor weights
echo ""
echo ">>> STEP 1: Running MedNeXt inference on BDMAP scans..."
python -u "${BASE_DIR}/code/prediction/predict_mednext.py" \
    --data_dir "${DATA_DIR}" \
    --model_path "${CHECKPOINT}" \
    --out_dir "${PRED_DIR}" \
    --model_id B \
    --kernel_size 3 \
    --roi_size 96 96 96 \
    --sw_batch_size 4 \
    --overlap 0.5 \
    --overwrite

# Step 2: Run GPU-accelerated evaluation against expert radiologist ground-truth
echo ""
echo ">>> STEP 2: Running GPU Radiologist Benchmark Evaluation..."
python -u "${BASE_DIR}/code/evaluation/evaluate_jhu_radiologist_gpu.py" \
    --jhu_dir "${JHU_DIR}" \
    --pred_dirs \
        "${PRED_DIR}" \
        "${RESULTS_DIR}/totalseg_gi_masks_bdmap" \
    --model_names Anchor_MedNeXt_N1594 TotalSegmentator \
    --out_csv "${RESULTS_DIR}/anchor_n1594_jhu_radiologist_benchmark.csv"

echo ""
echo "====================================================================="
echo "  Benchmark finished at: $(date)"
echo "  Results CSV: ${RESULTS_DIR}/anchor_n1594_jhu_radiologist_benchmark.csv"
echo "====================================================================="
