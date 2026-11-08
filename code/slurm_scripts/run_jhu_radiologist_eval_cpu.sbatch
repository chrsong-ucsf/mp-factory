#!/bin/bash
#SBATCH --job-name=jhu_rad_cpu
#SBATCH --output=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_rad_cpu_%j.out
#SBATCH --error=/mnt/scratch/user/chrsong/mp-factory/logs/jhu_rad_cpu_%j.err
#SBATCH --partition=community,community_short,cpu
#SBATCH --qos=hpc_core
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

module load CBI
module load miniforge3/26.3.2-3
eval "$(mamba shell hook --shell bash)"
mamba activate /mnt/scratch/user/chrsong/envs/mp-factory

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

cd /mnt/scratch/user/chrsong/mp-factory

mkdir -p logs results

echo "Executing JHU Radiologist Evaluation on compute node $(hostname) at $(date)..."
python -u code/evaluation/evaluate_jhu_radiologist_gpu.py \
    --jhu_dir JHU_data_radiologist_corrected \
    --pred_dirs \
        results/totalseg_gi_masks_bdmap \
        results/mednext_predictions \
        results/mednext_phase2_predictions \
        results/mednext_phase3_predictions \
        results/mednext_pool873_predictions \
        results/swin_unetr_predictions \
        results/swin_unetr_phase2_predictions \
        results/swin_unetr_phase3_predictions \
        results/ensemble_out \
        results/ensemble_phase3_out \
    --model_names TotalSegmentator MedNeXt MedNeXt_Phase2 MedNeXt_Phase3 MedNeXt_Pool873 Swin-UNETR Swin-UNETR_Phase2 Swin-UNETR_Phase3 EnsembleConsensus EnsembleConsensus_Phase3 \
    --out_csv results/jhu_radiologist_master_comparison.csv

echo "Completed JHU Evaluation at $(date)."
