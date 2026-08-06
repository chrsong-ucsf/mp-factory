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

# GT BDMAP Label Definition (radiologist NRRD ground truth)
# Label IDs from the BDMAP dataset annotation standard
GT_ORGAN_MAP = {
    2: 'stomach',
    3: 'duodenum',
    4: 'jejunum',
    5: 'ileum',
    6: 'colon'
}
# Note: GT label 1 = esophagus — excluded since models were NOT trained on esophagus

# Per-model label remapping: maps prediction label IDs -> GT BDMAP label IDs
# MedNeXt / Swin-UNETR / Ensemble training labels:
#   1=stomach, 2=duodenum, 3=small_bowel (jejunum+ileum), 4=colon
# TotalSegmentator v2 full-body labels (relevant GI organs):
#   18=stomach, 19=duodenum, 20=small_bowel, 57=colon
MODEL_LABEL_REMAP = {
    'MedNeXt':             {1: 2, 2: 3, 3: [4, 5], 4: 6},
    'Swin-UNETR':          {1: 2, 2: 3, 3: [4, 5], 4: 6},
    'EnsembleConsensus':   {1: 2, 2: 3, 3: [4, 5], 4: 6},
    'TotalSegmentator':    {18: 2, 19: 3, 20: [4, 5], 57: 6},
}

# Keep ORGAN_MAP as alias for compatibility
ORGAN_MAP = GT_ORGAN_MAP

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
    """Fast EDT-based 95th Percentile Hausdorff Distance with bounding box cropping."""
    if not np.any(gt_mask_np) or not np.any(pred_mask_np):
        return np.nan
    if np.array_equal(gt_mask_np, pred_mask_np):
        return 0.0

    # Crop to bounding box of organ masks to speed up distance transform 20x
    combined = gt_mask_np | pred_mask_np
    coords = np.argwhere(combined)
    min_c = np.maximum(0, coords.min(axis=0) - 20)
    max_c = np.minimum(gt_mask_np.shape, coords.max(axis=0) + 21)
    bbox = tuple(slice(min_c[i], max_c[i]) for i in range(3))

    gt_crop = gt_mask_np[bbox]
    pred_crop = pred_mask_np[bbox]

    gt_border = gt_crop ^ binary_erosion(gt_crop)
    pred_border = pred_crop ^ binary_erosion(pred_crop)

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
    # Check CUDA availability AND compatibility with installed PyTorch
    if torch.cuda.is_available():
        # PyTorch supports up to sm_90 (Ada Lovelace). Blackwell is sm_120+.
        major, minor = torch.cuda.get_device_capability(0)
        sm = major * 10 + minor
        supported_sms = {50, 60, 70, 75, 80, 86, 90}
        if sm not in supported_sms:
            print(f"[WARNING] GPU sm_{sm} is not compatible with this PyTorch build (supports up to sm_90). Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

            # Derive candidate subject ID strings (e.g., 'BDMAP_00242131', '00242131', '242131')
            clean_id = subject_id.replace('BDMAP_', '')
            short_id = clean_id.lstrip('0')
            id_aliases = [subject_id, clean_id, short_id]

            pred_path = None
            for key in id_aliases:
                if not key:
                    continue
                matches = (
                    glob.glob(os.path.join(pred_dir, f"*{key}*.nii.gz")) +
                    glob.glob(os.path.join(pred_dir, "**", f"*{key}*.nii.gz"), recursive=True) +
                    glob.glob(os.path.join(pred_dir, f"*{key}*.nrrd")) +
                    glob.glob(os.path.join(pred_dir, "**", f"*{key}*.nrrd"), recursive=True)
                )
                if matches:
                    pred_path = matches[0]
                    break

            # For ensemble: prefer _consensus over _uncertainty files
            if model_name == 'EnsembleConsensus' and pred_path:
                consensus_candidates = [p for p in matches if 'consensus' in os.path.basename(p).lower()]
                if consensus_candidates:
                    pred_path = consensus_candidates[0]

            if not pred_path or not os.path.exists(pred_path):
                print(f"  [MISSING] No prediction volume found for {subject_id} in {pred_dir}")
                continue

            try:
                gt_data, gt_header = nrrd.read(nrrd_path)
                pred_nii = nib.load(pred_path)
                pred_data = pred_nii.get_fdata()

                # Squeeze channel/batch singleton dimensions (e.g. (1, 290, 290, 322) -> (290, 290, 322))
                pred_data = np.squeeze(pred_data)

                # ----------------------------------------------------------------
                # World-space resampling: resample prediction onto the GT's exact
                # spatial grid using the NRRD affine (LPS) and the NIfTI affine.
                # Naive array resize ignores origin and axis-direction differences
                # (NRRD LPS vs NIfTI RAS), causing systematic axis flips.
                # ----------------------------------------------------------------
                try:
                    from nibabel.processing import resample_from_to

                    # Build RAS affine from GT NRRD header
                    sd = gt_header.get('space directions')   # (3,3) in LPS voxel->mm
                    so = gt_header.get('space origin', np.zeros(3))
                    if sd is not None and len(sd) >= 3:
                        sd_3x3 = np.array([v for v in sd if v is not None])[:3, :3]
                        so_3d  = np.array(so)[:3]
                        lps_aff = np.eye(4)
                        lps_aff[:3, :3] = sd_3x3.T
                        lps_aff[:3,  3] = so_3d
                        # LPS -> RAS: flip x and y signs
                        lps_to_ras = np.diag([-1., -1., 1., 1.])
                        gt_ras_aff = lps_to_ras @ lps_aff
                    else:
                        gt_ras_aff = np.eye(4)

                    gt_nii_img  = nib.Nifti1Image(gt_data.astype(np.int16), gt_ras_aff)

                    # If prediction has a dummy/reset affine (e.g. origin at 0,0,0), copy GT RAS affine
                    pred_aff = pred_nii.affine[:4, :4].copy()
                    if np.allclose(pred_aff[:3, 3], 0):
                        print(f"  [AFFINE FIX] {subject_id} {model_name}: replacing dummy origin (0,0,0) with GT origin")
                        pred_aff = gt_ras_aff.copy()

                    pred_nii_clean = nib.Nifti1Image(pred_data.astype(np.int16), pred_aff)
                    pred_nii_rs = resample_from_to(pred_nii_clean, gt_nii_img, order=0)  # nearest-neighbour
                    pred_data   = np.round(pred_nii_rs.get_fdata()).astype(np.int64)
                    print(f"  [RESAMPLE] {subject_id}: world-space resample OK → {pred_data.shape}")

                except Exception as rs_err:
                    # Fallback to naive CPU resize if nibabel resample fails
                    print(f"  [RESAMPLE WARN] {subject_id}: world-space resample failed ({rs_err}), using naive resize")
                    pred_data = np.squeeze(pred_data)
                    if gt_data.shape != pred_data.shape:
                        pred_t = torch.from_numpy(pred_data.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                        pred_t = torch.nn.functional.interpolate(pred_t, size=gt_data.shape, mode='nearest').squeeze(0).squeeze(0)
                        pred_data = pred_t.numpy().astype(np.int64)

                # Remap prediction labels to match GT BDMAP label IDs
                remap = MODEL_LABEL_REMAP.get(model_name, {})
                if remap:
                    remapped = np.zeros_like(pred_data, dtype=np.int64)
                    for pred_label, gt_label in remap.items():
                        mask = (pred_data == pred_label)
                        if isinstance(gt_label, list):
                            # small_bowel (pred) maps to both jejunum (4) and ileum (5) in GT
                            for gl in gt_label:
                                remapped[mask] = gl  # last write wins; they're evaluated separately
                            # Split evenly by z-slice: top half -> jejunum, bottom half -> ileum
                            z_mid = pred_data.shape[2] // 2
                            mask_top = np.zeros_like(mask); mask_top[:, :, :z_mid] = mask[:, :, :z_mid]
                            mask_bot = np.zeros_like(mask); mask_bot[:, :, z_mid:] = mask[:, :, z_mid:]
                            remapped[mask_top] = gt_label[0]  # jejunum = superior
                            remapped[mask_bot] = gt_label[1]  # ileum = inferior
                        else:
                            remapped[mask] = gt_label
                    pred_data = remapped
                    print(f"  [REMAP] Applied {model_name} label remapping")

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

                for label_id, organ_name in GT_ORGAN_MAP.items():
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
        out_dir = os.path.dirname(os.path.abspath(out_csv))
        os.makedirs(out_dir, exist_ok=True)
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
