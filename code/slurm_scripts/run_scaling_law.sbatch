#!/bin/bash
#SBATCH --job-name=scale_law
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/scale_law_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/scale_law_%j.err
#SBATCH --partition=community,community_short,cpu
#SBATCH --qos=hpc_core
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
cd /mnt/scratch/user/chrsong/mp-factory

mkdir -p logs results

CSV_IN="results/scaling_sweep/scaling_sweep_eval_20cases.csv"
if [ ! -f "$CSV_IN" ]; then
    CSV_IN=$(find results/ -name "*scaling*.csv" | head -n 1)
fi

echo "Fitting scaling laws on: ${CSV_IN}"
python -u code/evaluation/compute_scaling_law.py \
    --csv "${CSV_IN}" \
    --out_json results/scaling_law_statistical_summary.json
