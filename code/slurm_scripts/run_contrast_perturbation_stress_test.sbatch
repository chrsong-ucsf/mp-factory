#!/bin/bash
#SBATCH --job-name=p10_perturb
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/p10_contrast_perturbation_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/p10_contrast_perturbation_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/project10/evaluation_reports
mkdir -p /mnt/scratch/user/chrsong/mp-factory/docs/figures

echo "====================================================================="
echo "  Executing Project 10 Contrast Dose & Noise Perturbation Stress Test"
echo "  Date: $(date)"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "====================================================================="

python -u /mnt/scratch/user/chrsong/mp-factory/code/project10/evaluation/stress_test_contrast_perturbation.py \
    --synthetic_dir /mnt/scratch/user/chrsong/mp-factory/results/project10/synthetic_volumes \
    --model_path /mnt/scratch/user/chrsong/mp-factory/results/mednext_phase3/fold_2/best_mednext_phase2.pt \
    --out_csv /mnt/scratch/user/chrsong/mp-factory/results/project10/evaluation_reports/contrast_perturbation_stress_test.csv \
    --out_md /mnt/scratch/user/chrsong/mp-factory/results/project10/evaluation_reports/contrast_perturbation_summary.md \
    --out_fig /mnt/scratch/user/chrsong/mp-factory/docs/figures/project10_contrast_perturbation_robustness.png \
    --num_cases 50 \
    --device cuda

echo "====================================================================="
echo "  Stress test complete at $(date)"
echo "====================================================================="
