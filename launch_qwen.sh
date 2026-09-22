#!/bin/bash
#SBATCH --job-name=vllm_qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm_qwen_%j.out
#SBATCH --error=logs/vllm_qwen_%j.err

module load miniforge3/26.3.2-3
module load Sali
module load cuda/12.8.1

eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

mkdir -p logs

export HF_HUB_CACHE="/mnt/scratch/user/chrsong/.cache/huggingface/hub"
export HF_HUB_OFFLINE=1

vllm serve "Qwen/Qwen3.8-27B" \
    --port 8000 \
    --download-dir /mnt/scratch/user/chrsong/.cache/huggingface/hub \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --trust-remote-code
