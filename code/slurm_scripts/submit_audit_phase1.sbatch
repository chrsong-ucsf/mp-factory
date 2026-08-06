#!/bin/bash
#SBATCH --job-name=phase1_audit
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/audit_p1_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/audit_p1_%j.err

# Load exact environment modules
module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

# Set GT_DIR to the location of the external masks
GT_DIR="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"
TOTALSEG_DIR="/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap"
OUT_CSV="/mnt/scratch/user/chrsong/mp-factory/results/phase1_audit_summary.csv"

python /mnt/scratch/user/chrsong/mp-factory/code/audit_phase1_gi.py \
    --gt_dir "${GT_DIR}" \
    --totalseg_dir "${TOTALSEG_DIR}" \
    --out_csv "${OUT_CSV}" \
    --num_workers 32

python /mnt/scratch/user/chrsong/mp-factory/code/analyze_audit_results.py "${OUT_CSV}"