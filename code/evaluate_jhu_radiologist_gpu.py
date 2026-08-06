"""
evaluate_jhu_radiologist_gpu.py

GPU-Accelerated Evaluation & Audit Pipeline against Expert Radiologist Annotations.

Evaluates multi-organ 3D segmentation predictions (MedNeXt, Swin-UNETR, TotalSegmentator,
and Multi-Model Consensus) against expert radiologist-corrected ground truth masks
from the JHU BDMAP dataset.

Features:
  - Full PyTorch CUDA Acceleration for 3D tensor metrics (Dice, IoU, Precision, Sensitivity, Specificity).
  - 3D Morphological Erosion & Fast Distance Transform for HD95.
  - Topological Betti-0 (connected components count difference |Δβ0|).
  - Multi-phase annotation propagation rules from radiologist_clinical_feedback.md.
  - Multi-model evaluation reporting in a consolidated CSV.

Usage:
  python evaluate_jhu_radiologist_gpu.py \
    --jhu_dir /mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected \
    --pred_dirs \
      /mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions \
      /mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions \
      /mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap \
      /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out \
    --model_names MedNeXt Swin-UNETR TotalSegmentator EnsembleConsensus \
    --out_csv /mnt/scratch/user/chrsong/mp-factory/results/jhu_radiologist_gpu_audit_summary.csv
"""

import os
import sys
import glob
import argparse
import gc
import numpy as np
import pandas as pd
import nrrd
import nibabel as nib
import torch
from scipy.ndimage import label, binary_erosion, distance_transform_edt

# Organ Class Definition
ORGAN_MAP = {
    1: 'esophagus',
    2: 'stomach',
    3: 'duodenum',
    4: 'jejunum',
    5: 'ileum',
    6: 'colon'
}

# Expert Radiologist Multi-Phase Mask Propagation Rules
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


def compute_gpu_tensor_metrics(gt_tensor, pred_tensor, label_id):
    """
    Computes Dice, IoU (Jaccard), Precision, Sensitivity, and Specificity
    using PyTorch CUDA tensors for maximum speed.
    """
    gt_b = (gt_tensor == label_id)
    pred_b = (pred_tensor == label_id)

    tp = torch.sum(gt_b & pred_b).float()
    fp = torch.sum((~gt_b) & pred_b).float()
    fn = torch.sum(gt_b & (~pred_b)).float()
    tn = torch.sum((~gt_b) & (~pred_b)).float()

    gt_sum = torch.sum(gt_b).float()
    pred_sum = torch.sum(pred_b).float()

    if gt_sum == 0 and pred_sum == 0:
        return {'dice': np.nan, 'iou': np.nan, 'precision': np.nan, 'sensitivity': np.nan, 'specificity': 1.0}

    denom_dice = gt_sum + pred_sum
    dice = float((2.0 * tp / denom_dice).cpu().item()) if denom_dice > 0 else np.nan

    denom_iou = tp + fp + fn
    iou = float((tp / denom_iou).cpu().item()) if denom_iou > 0 else np.nan

    precision = float((tp / (tp + fp)).cpu().item()) if (tp + fp) > 0 else 0.0
    sensitivity = float((tp / (tp + fn)).cpu().item()) if (tp + fn) > 0 else 0.0
    specificity = float((tn / (tn + fp)).cpu().item()) if (tn + fp) > 0 else 1.0

    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def compute_betti_0(binary_mask_np):
    """Compute 0th Betti number (number of 3D connected components)."""
    if not np.any(binary_mask_np):
        return 0
    _, num_features = label(binary_mask_np)
    return int(num_features)


def compute_hd95_fast(gt_mask_np, pred_mask_np, voxel_spacing=(1.5, 1.5, 2.0)):
    """Fast EDT-based 95th Percentile Hausdorff Distance."""
    if not np.any(gt_mask_np) or not np.any(pred_mask_np):
        return np.nan
    if np.array_equal(gt_mask_np, pred_mask_np):
        return 0.0

    gt_border = gt_mask_np ^ binary_erosion(gt_mask_np)
    pred_border = pred_mask_np ^ binary_erosion(pred_mask_np)

    if not np.any(gt_border) or not np.any(pred_border):
        return np.nan

    dt_gt = distance_transform_edt(~gt_border, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)

    hd_gt_to_pred = dt_pred[gt_border]
    hd_pred_to_gt = dt_gt[pred_border]

    if len(hd_gt_to_pred) == 0 or len(hd_pred_to_gt) == 0:
        return np.nan

    return float(max(np.percentile(hd_gt_to_pred, 95), np.percentile(hd_pred_to_gt, 95)))


def find_jhu_dir(given_path):
    """Fallback search for JHU_data_radiologist_corrected in common directories."""
    candidates = [
        given_path,
        "JHU_data_radiologist_corrected",
        "03_Datasets/JHU_data_radiologist_corrected",
        "/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected",
        "/mnt/scratch/user/chrsong/mp-factory/03_Datasets/JHU_data_radiologist_corrected"
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.path.isdir(c):
            return c
    return given_path


def run_gpu_evaluation(jhu_dir, pred_dirs, model_names, out_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"      GPU-ACCELERATED RADIOLOGIST EVALUATION PIPELINE")
    print(f"      Execution Device: {device}")
    print(f"      JHU Dataset Dir:  {jhu_dir}")
    print("=" * 70)

    jhu_dir = find_jhu_dir(jhu_dir)
    if not os.path.exists(jhu_dir):
        print(f"[CRITICAL ERROR] JHU Radiologist Directory not found: {jhu_dir}")
        sys.exit(1)

    results = []

    for model_name, pred_dir in zip(model_names, pred_dirs):
        if not os.path.exists(pred_dir):
            print(f"[WARNING] Prediction directory for {model_name} does not exist: {pred_dir}. Skipping.")
            continue

        print(f"\n--- Evaluating Model: {model_name} ({pred_dir}) ---")

        for subject_id, nrrd_file in PROPAGATION_MAP.items():
            nrrd_path = os.path.join(jhu_dir, nrrd_file)
            if not os.path.exists(nrrd_path):
                print(f"  [SKIP] Ground truth NRRD missing for {subject_id}: {nrrd_path}")
                continue

            # Locate prediction file
            patterns = [
                os.path.join(pred_dir, f"{subject_id}_gi_seg.nii.gz"),
                os.path.join(pred_dir, f"{subject_id}_consensus.nii.gz"),
                os.path.join(pred_dir, f"*{subject_id}*.nii.gz")
            ]
            pred_path = None
            for p in patterns:
                matches = glob.glob(p)
                if matches:
                    pred_path = matches[0]
                    break

            if not pred_path or not os.path.exists(pred_path):
                print(f"  [MISSING] No prediction volume found for {subject_id}")
                continue

            try:
                gt_data, _ = nrrd.read(nrrd_path)
                pred_nii = nib.load(pred_path)
                pred_data = pred_nii.get_fdata()

                if gt_data.shape != pred_data.shape:
                    print(f"  [SHAPE MISMATCH] {subject_id}: GT {gt_data.shape} vs Pred {pred_data.shape}")
                    continue

                # Load into PyTorch CUDA Tensor
                gt_tensor = torch.from_numpy(gt_data.astype(np.int64)).to(device)
                pred_tensor = torch.from_numpy(pred_data.astype(np.int64)).to(device)

                row = {
                    'model_name': model_name,
                    'subject_id': subject_id,
                    'source_nrrd': nrrd_file,
                    'pred_path': os.path.basename(pred_path)
                }

                mean_dices = []
                mean_hd95s = []

                for label_id, organ_name in ORGAN_MAP.items():
                    # GPU Tensor Metrics
                    m = compute_gpu_tensor_metrics(gt_tensor, pred_tensor, label_id)
                    
                    gt_mask_np = (gt_data == label_id)
                    pred_mask_np = (pred_data == label_id)

                    # Betti-0 and HD95
                    gt_b0 = compute_betti_0(gt_mask_np)
                    pred_b0 = compute_betti_0(pred_mask_np)
                    b0_diff = abs(gt_b0 - pred_b0)
                    hd95 = compute_hd95_fast(gt_mask_np, pred_mask_np)

                    row[f'{organ_name}_dice'] = m['dice']
                    row[f'{organ_name}_iou'] = m['iou']
                    row[f'{organ_name}_sensitivity'] = m['sensitivity']
                    row[f'{organ_name}_hd95'] = hd95
                    row[f'{organ_name}_b0_diff'] = b0_diff

                    if not np.isnan(m['dice']):
                        mean_dices.append(m['dice'])
                    if not np.isnan(hd95):
                        mean_hd95s.append(hd95)

                row['mean_dice'] = np.mean(mean_dices) if mean_dices else np.nan
                row['mean_hd95'] = np.mean(mean_hd95s) if mean_hd95s else np.nan
                results.append(row)

                print(f"  [{subject_id}] Mean Dice: {row['mean_dice']:.4f} | Mean HD95: {row['mean_hd95']:.2f}mm")

                # Clean GPU VRAM
                del gt_tensor, pred_tensor
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [ERROR] Processing {subject_id} failed: {e}")

    if results:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)
        print("\n" + "=" * 70)
        print(f"      AUDIT COMPLETE! Results saved to: {out_csv}")
        print("=" * 70)
        print("\n--- MODEL PERFORMANCE SUMMARY AGAINST RADIOLOGIST GROUND TRUTH ---")
        summary = df.groupby('model_name')[['mean_dice', 'mean_hd95']].mean()
        print(summary)
    else:
        print("\n[WARNING] No predictions were successfully evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-Accelerated Evaluation against JHU Radiologist Corrected Set")
    parser.add_argument("--jhu_dir", default="/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected", help="Path to JHU_data_radiologist_corrected directory")
    parser.add_argument("--pred_dirs", nargs="+", required=True, help="List of prediction directories")
    parser.add_argument("--model_names", nargs="+", required=True, help="List of model names corresponding to pred_dirs")
    parser.add_argument("--out_csv", default="/mnt/scratch/user/chrsong/mp-factory/results/jhu_radiologist_gpu_audit_summary.csv", help="Output summary CSV path")
    args = parser.parse_args()

    run_gpu_evaluation(args.jhu_dir, args.pred_dirs, args.model_names, args.out_csv)
