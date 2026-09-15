import os
import glob
import numpy as np
import nibabel as nib
import torch

def diagnose():
    print("--- 1. Class Index Alignment ---")
    
    # Check ensemble_out (consensus)
    lbl_files = glob.glob('/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out/*_consensus.nii.gz')[:5]
    print("Consensus Labels:")
    for f in lbl_files:
        data = nib.load(f).get_fdata()
        unique, counts = np.unique(data, return_counts=True)
        print(f"  {os.path.basename(f)}:", dict(zip(unique.astype(int), counts)))

    # Check JHU corrected
    jhu_files = glob.glob('/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected/*/*.seg.nrrd')[:2]
    import nrrd
    print("\nJHU Corrected Labels:")
    for f in jhu_files:
        data, header = nrrd.read(f)
        unique, counts = np.unique(data, return_counts=True)
        print(f"  {os.path.basename(f)}:", dict(zip(unique.astype(int), counts)))

    # Check TotalSegmentator GI masks in results/totalseg_gi_masks
    totalseg_files = glob.glob('/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks*/*.nii.gz')[:5]
    if not totalseg_files:
        totalseg_files = glob.glob('/mnt/scratch/user/chrsong/mp-factory/results/totalseg*/**/*.nii.gz', recursive=True)[:5]
        
    print("\nTotalSegmentator Masks:")
    for f in totalseg_files:
        data = nib.load(f).get_fdata()
        unique, counts = np.unique(data, return_counts=True)
        print(f"  {os.path.basename(f)}:", dict(zip(unique.astype(int), counts)))

    print("\n--- 2. Model Output Distribution ---")
    # Let's check a prediction file if available
    pred_files = glob.glob('/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions/*/*.nii.gz')[:2]
    if pred_files:
        print("MedNeXt Predictions:")
        for f in pred_files:
            data = nib.load(f).get_fdata()
            unique, counts = np.unique(data, return_counts=True)
            print(f"  {os.path.basename(f)}:", dict(zip(unique.astype(int), counts)))
    else:
        print("No prediction masks found to check.")

if __name__ == "__main__":
    diagnose()
