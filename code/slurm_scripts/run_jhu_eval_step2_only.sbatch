#!/bin/bash
#SBATCH --job-name=jhu_eval_fast
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_eval_fast_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_eval_fast_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

BASE_DIR="/mnt/scratch/user/chrsong/mp-factory"
PRED_DIR="${BASE_DIR}/results/anchor_full_predictions"
JHU_DIR="${BASE_DIR}/JHU_data_radiologist_corrected"
RESULTS_DIR="${BASE_DIR}/results"

echo ">>> Running GPU/CPU Radiologist Benchmark Evaluation with label remap fix..."
python -u "${BASE_DIR}/code/evaluation/evaluate_jhu_radiologist_gpu.py" \
    --jhu_dir "${JHU_DIR}" \
    --pred_dirs \
        "${PRED_DIR}" \
        "${RESULTS_DIR}/totalseg_gi_masks_bdmap" \
    --model_names Anchor_MedNeXt_N1594 TotalSegmentator \
    --out_csv "${RESULTS_DIR}/anchor_n1594_jhu_radiologist_benchmark.csv"

echo ">>> Benchmark evaluation finished at: $(date)"
