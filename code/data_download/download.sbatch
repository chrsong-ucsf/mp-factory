#!/bin/bash
#SBATCH --job-name=hf_download
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=05:00:00
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/download_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/download_%j.err

set -eo pipefail

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

# Paste your copied Hugging Face token here:
export HF_TOKEN="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"

hf download BodyMaps/CancerVerse --repo-type dataset --local-dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse
