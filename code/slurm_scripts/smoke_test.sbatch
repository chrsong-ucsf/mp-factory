#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/smoke_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/smoke_%j.err

set -eo pipefail

# Load the exact modules we validated
module load CBI
module load miniforge3/26.3.2-3

# Initialize shell and activate the environment
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

echo "=== Hostname: $(hostname) ==="
echo "=== Date: $(date) ==="
nvidia-smi

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU name:', torch.cuda.get_device_name(0))
print('GPU mem (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)
"

echo "=== Done ==="
