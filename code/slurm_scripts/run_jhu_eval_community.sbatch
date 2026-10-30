#!/bin/bash
#SBATCH --job-name=jhu_rad_eval_fast
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_rad_eval_fast_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_rad_eval_fast_%j.err
#SBATCH --partition=community
#SBATCH --qos=cmnt
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

BASE_DIR="/mnt/scratch/user/chrsong/mp-factory"
LOGS_DIR="${BASE_DIR}/logs"
RESULTS_DIR="${BASE_DIR}/results"
JHU_DIR="${BASE_DIR}/JHU_data_radiologist_corrected"

echo "====================================================================="
echo "  Executing Fast JHU Radiologist Evaluation on Community Compute"
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "====================================================================="

python -u "${BASE_DIR}/code/evaluation/evaluate_jhu_radiologist_gpu.py" \
    --jhu_dir "${JHU_DIR}" \
    --pred_dirs \
        "${RESULTS_DIR}/totalseg_gi_masks_bdmap" \
        "${RESULTS_DIR}/mednext_predictions" \
        "${RESULTS_DIR}/mednext_phase2_predictions" \
        "${RESULTS_DIR}/mednext_phase3_predictions" \
        "${RESULTS_DIR}/mednext_pool873_predictions" \
        "${RESULTS_DIR}/swin_unetr_predictions" \
        "${RESULTS_DIR}/swin_unetr_phase2_predictions" \
        "${RESULTS_DIR}/swin_unetr_phase3_predictions" \
        "${RESULTS_DIR}/ensemble_out" \
        "${RESULTS_DIR}/ensemble_phase3_out" \
    --model_names TotalSegmentator MedNeXt MedNeXt_Phase2 MedNeXt_Phase3 MedNeXt_Pool873 Swin-UNETR Swin-UNETR_Phase2 Swin-UNETR_Phase3 EnsembleConsensus EnsembleConsensus_Phase3 \
    --out_csv "${RESULTS_DIR}/jhu_radiologist_master_comparison.csv"

echo "====================================================================="
echo "  Evaluation completed successfully at $(date)"
echo "====================================================================="
