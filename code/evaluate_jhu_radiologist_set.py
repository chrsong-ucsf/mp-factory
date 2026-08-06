"""
evaluate_jhu_radiologist_set.py

Audits & evaluates model predictions (TotalSegmentator, MedNeXt, Swin-UNETR)
against expert radiologist-corrected ground truth masks in JHU_data_radiologist_corrected.

Applies radiologist propagation rules from radiologist_clinical_feedback.md:
  - BDMAP_00242113.seg.nrrd -> 113, 115, 116
  - BDMAP_00242114.seg.nrrd -> 114
  - BDMAP_00242131.seg.nrrd -> 131, 132, 133
  - BDMAP_00242134.seg.nrrd -> 134
  - BDMAP_00242136.seg.nrrd -> 135, 136
  - BDMAP_00394224.seg.nrrd -> 224
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import nrrd
import nibabel as nib
from scipy.ndimage import label, binary_erosion, distance_transform_edt

ORGAN_MAP = {
    1: 'esophagus',
    2: 'stomach',
    3: 'duodenum',
    4: 'jejunum',
    5: 'ileum',
    6: 'colon'
}

PROPAGATION_MAP = {
    'BDMAP_00242113': 'BDMAP_00242113.seg.nrrd',
    'BDMAP_00242114': 'BDMAP_00242114.seg.nrrd',
    'BDMAP_00242115': 'BDMAP_00242113.seg.nrrd',
    'BDMAP_00242116': 'BDMAP_00242113.seg.nrrd',
    'BDMAP_00242131': 'BDMAP_00242131.seg.nrrd',
    'BDMAP_00242132': 'BDMAP_00242131.seg.nrrd',
    'BDMAP_00242133': 'BDMAP_00242131.seg.nrrd',
    'BDMAP_00242134': 'BDMAP_00242134.seg.nrrd',
    'BDMAP_00242135': 'BDMAP_00242136.seg.nrrd',
    'BDMAP_00242136': 'BDMAP_00242136.seg.nrrd',
    'BDMAP_00394224': 'BDMAP_00394224.seg.nrrd'
}


def dice_score(gt, pred, label_id):
    g = (gt == label_id).astype(np.uint8)
    p = (pred == label_id).astype(np.uint8)
    denom = g.sum() + p.sum()
    if denom == 0:
        return np.nan
    return 2.0 * (g * p).sum() / float(denom)


def iou_score(gt, pred, label_id):
    g = (gt == label_id).astype(np.uint8)
    p = (pred == label_id).astype(np.uint8)
    denom = (g | p).sum()
    if denom == 0:
        return np.nan
    return float((g & p).sum()) / float(denom)


def compute_betti_0(mask):
    if not np.any(mask):
        return 0
    _, n = label(mask)
    return int(n)


def evaluate_jhu_set(jhu_dir, pred_dir, out_csv):
    results = []

    print(f"Loading radiologist corrected masks from {jhu_dir}...")
    
    for subject_id, nrrd_file in PROPAGATION_MAP.items():
        nrrd_path = os.path.join(jhu_dir, nrrd_file)
        if not os.path.exists(nrrd_path):
            print(f"[SKIP] NRRD file missing for {subject_id}: {nrrd_path}")
            continue

        gt_data, _ = nrrd.read(nrrd_path)

        # Look for matching prediction file
        pred_pattern = os.path.join(pred_dir, f"*{subject_id}*.nii.gz")
        pred_files = glob.glob(pred_pattern)
        
        if not pred_files:
            print(f"[MISSING PRED] No prediction file found for {subject_id} in {pred_dir}")
            continue

        pred_path = pred_files[0]
        pred_data = nib.load(pred_path).get_fdata()

        if gt_data.shape != pred_data.shape:
            print(f"[SHAPE MISMATCH] {subject_id}: GT shape {gt_data.shape} vs Pred shape {pred_data.shape}")
            continue

        row = {'subject_id': subject_id, 'source_nrrd': nrrd_file}
        mean_dices = []

        for label_id, organ_name in ORGAN_MAP.items():
            d = dice_score(gt_data, pred_data, label_id)
            iou = iou_score(gt_data, pred_data, label_id)
            
            gt_b0 = compute_betti_0(gt_data == label_id)
            pred_b0 = compute_betti_0(pred_data == label_id)
            b0_diff = abs(gt_b0 - pred_b0)

            row[f'{organ_name}_dice'] = d
            row[f'{organ_name}_iou'] = iou
            row[f'{organ_name}_b0_diff'] = b0_diff

            if not np.isnan(d):
                mean_dices.append(d)

        row['mean_dice'] = np.mean(mean_dices) if mean_dices else np.nan
        results.append(row)
        print(f"Evaluated {subject_id}: Mean Dice = {row['mean_dice']:.4f}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        print(f"\nSaved evaluation summary to {out_csv}")
        print("\n--- MEAN PERFORMANCE SUMMARY ---")
        print(df.mean(numeric_only=True))
    else:
        print("\nNo matching prediction files were found to evaluate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against JHU Radiologist Corrected NRRD set.")
    parser.add_argument("--jhu_dir", default="/Users/chrissong/research_su26/03_Datasets/JHU_data_radiologist_corrected", help="Path to JHU radiologist corrected NRRD directory")
    parser.add_argument("--pred_dir", required=True, help="Path to directory containing model prediction NIfTI files")
    parser.add_argument("--out_csv", default="jhu_radiologist_audit_results.csv", help="Path to save evaluation summary CSV")
    args = parser.parse_args()

    evaluate_jhu_set(args.jhu_dir, args.pred_dir, args.out_csv)
