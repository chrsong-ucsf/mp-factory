#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=p10_reg
#SBATCH --array=3-6
#SBATCH --output=logs/p10_reg_%A_%a.out
#SBATCH --error=logs/p10_reg_%A_%a.err

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/project10/registered_volumes

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
TOTAL_CHUNKS=8

echo "========================================="
echo "Job ID:         $SLURM_ARRAY_JOB_ID"
echo "Array Task:     $SLURM_ARRAY_TASK_ID / $TOTAL_CHUNKS"
echo "Node:           $SLURMD_NODENAME"
echo "Started:        $(date)"
echo "========================================="

python -u code/project10/datasets/register_multiphase_pairs.py \
    --pairs        results/project10/manifests/multiphase_pairs.csv \
    --out_dir      results/project10/registered_volumes \
    --log_csv      results/project10/manifests/registration_log.csv \
    --chunk_idx    $SLURM_ARRAY_TASK_ID \
    --total_chunks $TOTAL_CHUNKS \
    --nthreads     8 \
    --skip_existing

echo ""
echo "Chunk $SLURM_ARRAY_TASK_ID finished at: $(date)"
