#!/bin/bash
#SBATCH --job-name=eval_p873
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/eval_pool873_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/eval_pool873_%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results

echo "====================================================================="
echo "  Executing 500-Subject Benchmark of MedNeXt-Pool873 on CPU"
echo "  Date: $(date)"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/utilities/compare_models.py \
    --data_dir          "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox" \
    --swin_pred_dir     "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_phase3_predictions" \
    --mednext_pred_dir  "/mnt/scratch/user/chrsong/mp-factory/results/mednext_pool873_predictions" \
    --out_csv           "/mnt/scratch/user/chrsong/mp-factory/results/model_comparison_pool873.csv" \
    --max_subjects 500

echo "====================================================================="
echo "  Benchmark finished at $(date)"
echo "====================================================================="
