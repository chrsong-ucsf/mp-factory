#!/bin/bash
#SBATCH --job-name=qwen_harness
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=qwen_%j.log

# Load environment
source ~/.zshrc

# Launch vLLM pointing to local scratch
python -m vllm.entrypoints.openai.api_server \
    --model /scratch/chrsong/qwen_model \
    --port 8000
