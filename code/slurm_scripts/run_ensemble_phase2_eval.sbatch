#!/bin/bash
#SBATCH --job-name=ens_p2_eval
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_phase2_eval_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/ensemble_phase2_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase2_out

# Fast CUDA GPU 3D Ensemble Execution across MedNeXt Phase 2 + Swin-UNETR Phase 2
python -u /mnt/scratch/user/chrsong/mp-factory/code/evaluation/evaluate_multi_model_ensemble.py \
    --pred_dirs \
        "/mnt/scratch/user/chrsong/mp-factory/results/mednext_phase2_predictions" \
        "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_phase2_predictions" \
    --model_names MedNeXt_Phase2 Swin-UNETR_Phase2 \
    --out_dir  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase2_out" \
    --out_csv  "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase2_audit_summary.csv" \
    --splits_dir "/mnt/scratch/user/chrsong/mp-factory/results/dataset_splits_phase2"
