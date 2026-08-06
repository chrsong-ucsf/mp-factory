#!/bin/bash
#SBATCH --job-name=jhu_rad_eval
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_rad_eval_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_rad_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00


# -----------------------------------------------------------------------------
# Module Loading & Environment Activation
# -----------------------------------------------------------------------------
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1


# -----------------------------------------------------------------------------
# Directory Setup & Dataset Discovery
# -----------------------------------------------------------------------------
BASE_DIR="/mnt/scratch/user/chrsong/mp-factory"
LOGS_DIR="${BASE_DIR}/logs"
RESULTS_DIR="${BASE_DIR}/results"

mkdir -p "${LOGS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Discover JHU dataset directory
if [ -d "${BASE_DIR}/JHU_data_radiologist_corrected" ]; then
    JHU_DIR="${BASE_DIR}/JHU_data_radiologist_corrected"
elif [ -d "${BASE_DIR}/03_Datasets/JHU_data_radiologist_corrected" ]; then
    JHU_DIR="${BASE_DIR}/03_Datasets/JHU_data_radiologist_corrected"
else
    JHU_DIR="${BASE_DIR}/JHU_data_radiologist_corrected"
fi

echo "====================================================================="
echo "  Executing GPU-Accelerated JHU Radiologist Evaluation"
echo "  Date: $(date)"
echo "  JHU Dataset Directory: ${JHU_DIR}"
echo "====================================================================="

# -----------------------------------------------------------------------------
# Run Multi-Model GPU Evaluation
# -----------------------------------------------------------------------------
python -u "${BASE_DIR}/code/evaluation/evaluate_jhu_radiologist_gpu.py" \
    --jhu_dir "${JHU_DIR}" \
    --pred_dirs \
        "${RESULTS_DIR}/mednext_predictions" \
        "${RESULTS_DIR}/swin_unetr_predictions" \
        "${RESULTS_DIR}/totalseg_gi_masks_bdmap" \
        "${RESULTS_DIR}/ensemble_out" \
    --model_names MedNeXt Swin-UNETR TotalSegmentator EnsembleConsensus \
    --out_csv "${RESULTS_DIR}/jhu_radiologist_gpu_audit_summary.csv"

echo "====================================================================="
echo "  Evaluation completed successfully at $(date)"
echo "  Summary saved to: ${RESULTS_DIR}/jhu_radiologist_gpu_audit_summary.csv"
echo "====================================================================="
