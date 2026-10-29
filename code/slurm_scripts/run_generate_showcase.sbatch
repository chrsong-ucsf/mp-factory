#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=gi_showcase
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/generate_showcase_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/generate_showcase_%j.err

cd /mnt/scratch/user/chrsong/mp-factory
mkdir -p logs results/showcase_phase3

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1

echo "====================================================================="
echo "  Generating Publication 3D Meshes, Orthoviews, and HTML Showcases"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURMD_NODENAME"
echo "  Date:      $(date)"
echo "====================================================================="

python -u code/utilities/generate_showcase.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --jhu_dir /mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected \
    --mask_dir /mnt/scratch/user/chrsong/mp-factory/results/ensemble_phase3_out \
    --out_dir /mnt/scratch/user/chrsong/mp-factory/results/showcase_phase3 \
    --subjects \
        BDMAP_00242388 \
        BDMAP_00242410 \
        BDMAP_00242414 \
        BDMAP_00242419 \
        BDMAP_00242436 \
        BDMAP_00242440 \
        BDMAP_00242456 \
        BDMAP_00242467 \
        BDMAP_00242482 \
        BDMAP_00242486 \
        BDMAP_00242131 \
        BDMAP_00242114 \
        BDMAP_00242136 \
        BDMAP_00394224

echo "====================================================================="
echo "  Finished at: $(date)"
echo "====================================================================="
