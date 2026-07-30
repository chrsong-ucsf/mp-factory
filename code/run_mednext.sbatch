#!/bin/bash
#SBATCH --job-name=mednext_gi
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_%A_%a.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/mednext_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3

# Load modules and activate environment
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

# Ensure MedNeXt package is installed
pip install --quiet git+https://github.com/MIC-DKFZ/MedNeXt.git

# Make directories for outputs and logs
mkdir -p /mnt/scratch/user/chrsong/mp-factory/logs
OUT_DIR="/mnt/scratch/user/chrsong/mp-factory/results/mednext_models/fold_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${OUT_DIR}"

DATA_DIR="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"

# Execute 3D MedNeXt-B Training Pipeline across 4 parallel GPU array tasks (Folds 0..3)
python -u /mnt/scratch/user/chrsong/mp-factory/code/train_mednext.py \
    --data_dir "${DATA_DIR}" \
    --out_dir "${OUT_DIR}" \
    --model_id "B" \
    --kernel_size 3 \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    --num_folds 4 \
    --batch_size 1 \
    --lr 0.0001 \
    --val_interval 2
