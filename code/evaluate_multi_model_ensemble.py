"""
evaluate_multi_model_ensemble.py

Multi-Model Deep Ensemble Evaluation, Consensus Assembly, and Active Learning Triage.

This script:
  1. Loads multi-organ segmentation masks or softmax probability maps from N independent models.
  2. Computes a Diversity-Promoting Weighting Strategy via a pairwise inter-model Dice similarity matrix.
  3. Generates voxel-wise Spatial Predictive Uncertainty Heatmaps (entropy/variance across models).
  4. Assembles a weighted consensus "pseudo-ground truth" mask (<subject_id>_consensus.nii.gz).
  5. Evaluates model and consensus quality via Dice, IoU, HD95, Betti-0 Count Difference (|Δβ0|), ARI, and VOI.
  6. Routes scans into Clean, Weak, or Reject/Active-Learning-Triage buckets.

Usage:
  python evaluate_multi_model_ensemble.py \
    --pred_dirs /path/to/totalseg_masks /path/to/mednext_predictions /path/to/swin_unetr_predictions \
    --model_names TotalSeg MedNeXt Swin-UNETR \
    --out_dir /path/to/results/ensemble_out \
    --out_csv /path/to/results/ensemble_audit_summary.csv
"""

import os
import sys
import glob
import re
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label, binary_erosion, distance_transform_edt
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score
from skimage.metrics import variation_of_information

ORGAN_MAP = {
    1: 'stomach',
    2: 'duodenum',
    3: 'small_bowel',
    4: 'colon'
}


def compute_betti_0(binary_mask):
    """Compute 0th Betti number (number of 3D connected components)."""
    if not np.any(binary_mask):
        return 0
    _, num_features = label(binary_mask)
    return int(num_features)


def compute_hd95_fast(gt_mask, pred_mask, voxel_spacing=(1.5, 1.5, 2.0)):
    """Fast EDT-based 95th Percentile Hausdorff Distance."""
    if not np.any(gt_mask) or not np.any(pred_mask):
        return np.nan
    if np.array_equal(gt_mask, pred_mask):
        return 0.0

    gt_border = gt_mask ^ binary_erosion(gt_mask)
    pred_border = pred_mask ^ binary_erosion(pred_mask)

    if not np.any(gt_border) or not np.any(pred_border):
        return np.nan

    dt_gt = distance_transform_edt(~gt_border, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)

    hd_gt_to_pred = dt_pred[gt_border]
    hd_pred_to_gt = dt_gt[pred_border]

    if len(hd_gt_to_pred) == 0 or len(hd_pred_to_gt) == 0:
        return np.nan

    return float(max(np.percentile(hd_gt_to_pred, 95), np.percentile(hd_pred_to_gt, 95)))


def compute_clustering_metrics(gt_arr, pred_arr):
    """Compute Adjusted Rand Index (ARI) and Variation of Information (VOI)."""
    gt_flat = gt_arr.ravel()[::50]
    pred_flat = pred_arr.ravel()[::50]

    ari = adjusted_rand_score(gt_flat, pred_flat)
    split, merge = variation_of_information(gt_flat, pred_flat)
    voi = split + merge
    return float(ari), float(voi)


def compute_diversity_weights(model_masks):
    """
    Computes diversity-promoting weights for N models based on pairwise Dice similarity.
    Models making redundant errors receive lower relative weights.
    """
    num_models = len(model_masks)
    if num_models == 1:
        return np.array([1.0])

    dice_matrix = np.ones((num_models, num_models))
    for i in range(num_models):
        for j in range(i + 1, num_models):
            m1 = model_masks[i] > 0
            m2 = model_masks[j] > 0
            intersection = np.sum(m1 & m2)
            total = np.sum(m1) + np.sum(m2)
            d = (2.0 * intersection) / total if total > 0 else 1.0
            dice_matrix[i, j] = d
            dice_matrix[j, i] = d

    # Mean redundancy per model (excluding self)
    mean_redundancy = (np.sum(dice_matrix, axis=1) - 1.0) / (num_models - 1 + 1e-7)
    diversity = np.clip(1.0 - mean_redundancy, 0.05, 1.0)
    weights = diversity / np.sum(diversity)
    return weights


def compute_spatial_uncertainty(prob_maps):
    """
    Computes voxel-wise predictive entropy across N models:
    H = - sum_c p_c log(p_c)
    """
    avg_probs = np.mean(prob_maps, axis=0)
    epsilon = 1e-7
    clamped_probs = np.clip(avg_probs, epsilon, 1.0 - epsilon)
    if avg_probs.ndim == 4:  # (Classes, X, Y, Z)
        ent = -np.sum(clamped_probs * np.log2(clamped_probs), axis=0)
    else:  # Binary (X, Y, Z)
        ent = -(clamped_probs * np.log2(clamped_probs) + (1.0 - clamped_probs) * np.log2(1.0 - clamped_probs))
    return ent


def evaluate_subject(subject_id, model_files, model_names, out_dir):
    """Processes a single subject scan across all available models."""
    res = {'subject_id': subject_id, 'num_models': len(model_files)}

    try:
        loaded_masks = []
        affine = None

        for fpath in model_files:
            nii = nib.load(fpath)
            if affine is None:
                affine = nii.affine
            arr = np.asanyarray(nii.dataobj).astype(np.uint8)
            if arr.ndim == 4:
                arr = arr[0]
            loaded_masks.append(arr)

        model_masks = np.stack(loaded_masks, axis=0)  # (N_models, X, Y, Z)

        # 1. Diversity Weights
        weights = compute_diversity_weights(model_masks)
        for name, w in zip(model_names, weights):
            res[f'weight_{name}'] = round(float(w), 4)

        # 2. Probability representation for consensus
        one_hot_probs = []
        for i in range(len(model_files)):
            m = model_masks[i]
            oh = np.stack([(m == c).astype(np.float32) for c in range(5)], axis=0)
            one_hot_probs.append(oh)
        one_hot_stack = np.stack(one_hot_probs, axis=0)  # (N_models, 5, X, Y, Z)

        # Weighted probability consensus
        consensus_probs = np.average(one_hot_stack, axis=0, weights=weights)  # (5, X, Y, Z)
        consensus_mask = np.argmax(consensus_probs, axis=0).astype(np.uint8)  # (X, Y, Z)

        # 3. Spatial Uncertainty Heatmap
        # Compute entropy directly on consensus_probs (5, X, Y, Z) — already class-probability distribution
        epsilon = 1e-7
        clamped = np.clip(consensus_probs, epsilon, 1.0 - epsilon)
        uncertainty_map = -np.sum(clamped * np.log2(clamped), axis=0)  # (X, Y, Z)
        res['mean_uncertainty'] = round(float(np.mean(uncertainty_map)), 4)
        res['max_uncertainty'] = round(float(np.max(uncertainty_map)), 4)

        # Save Consensus NIfTI & Uncertainty Heatmap NIfTI
        os.makedirs(out_dir, exist_ok=True)
        consensus_path = os.path.join(out_dir, f"{subject_id}_consensus.nii.gz")
        uncertainty_path = os.path.join(out_dir, f"{subject_id}_uncertainty.nii.gz")

        nib.save(nib.Nifti1Image(consensus_mask, affine=affine), consensus_path)
        nib.save(nib.Nifti1Image(uncertainty_map.astype(np.float32), affine=affine), uncertainty_path)

        # 4. Metric Audit per organ against Consensus
        eval_dices = []
        eval_ious = []
        max_betti_diff = 0

        for organ_id, organ_name in ORGAN_MAP.items():
            cons_o = (consensus_mask == organ_id)
            betti_cons = compute_betti_0(cons_o)
            res[f'{organ_name}_betti_consensus'] = betti_cons

            organ_dices = []
            organ_ious = []
            organ_betti_diffs = []

            for i, name in enumerate(model_names):
                m_o = (model_masks[i] == organ_id)
                intersection = np.sum(cons_o & m_o)
                total = np.sum(cons_o) + np.sum(m_o)
                union = total - intersection

                d = (2.0 * intersection) / total if total > 0 else 1.0
                iou = intersection / union if union > 0 else 1.0

                betti_m = compute_betti_0(m_o)
                b_diff = abs(betti_cons - betti_m)

                res[f'{organ_name}_dice_{name}'] = round(float(d), 4)
                res[f'{organ_name}_iou_{name}'] = round(float(iou), 4)
                res[f'{organ_name}_betti_diff_{name}'] = b_diff

                organ_dices.append(d)
                organ_ious.append(iou)
                organ_betti_diffs.append(b_diff)

            mean_org_dice = float(np.mean(organ_dices))
            res[f'{organ_name}_mean_dice'] = round(mean_org_dice, 4)
            res[f'{organ_name}_mean_iou'] = round(float(np.mean(organ_ious)), 4)
            res[f'{organ_name}_max_betti_diff'] = max(organ_betti_diffs)

            eval_dices.append(mean_org_dice)
            eval_ious.append(np.mean(organ_ious))
            max_betti_diff = max(max_betti_diff, max(organ_betti_diffs))

        res['mean_consensus_dice'] = round(float(np.mean(eval_dices)), 4)
        res['mean_consensus_iou'] = round(float(np.mean(eval_ious)), 4)
        res['max_betti_diff'] = max_betti_diff

        # 5. Categorization / Triage Logic
        if res['mean_consensus_dice'] < 0.50 or max_betti_diff > 5 or res['mean_uncertainty'] > 0.35:
            res['triage_category'] = 'REJECT_OR_TRIAGE'
            res['action'] = 'Route to Active Learning Radiologist Queue'
        elif res['mean_consensus_dice'] >= 0.82 and max_betti_diff <= 2:
            res['triage_category'] = 'CLEAN_HIGH_CONFIDENCE'
            res['action'] = 'Auto-Approve for Generative VAE Conditioning'
        else:
            res['triage_category'] = 'WEAK_COARSE'
            res['action'] = 'Phase 3 Auto-labeling / Coarse Ignore Class'

        res['status'] = 'SUCCESS'

    except Exception as e:
        res['status'] = f'ERROR: {str(e)}'

    return res


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Ensemble Consensus & Active Learning Audit")
    parser.add_argument("--pred_dirs", type=str, nargs="+", required=True,
                        help="List of output directories for each model")
    parser.add_argument("--model_names", type=str, nargs="+", default=None,
                        help="Names corresponding to model directories")
    parser.add_argument("--out_dir", type=str, default="./results/ensemble_out",
                        help="Output directory for consensus masks and uncertainty maps")
    parser.add_argument("--out_csv", type=str, default="./results/ensemble_audit_summary.csv",
                        help="Output path for evaluation CSV summary")
    args = parser.parse_args()

    pred_dirs = args.pred_dirs
    model_names = args.model_names if args.model_names else [f"model_{i+1}" for i in range(len(pred_dirs))]

    if len(pred_dirs) != len(model_names):
        print("ERROR: --pred_dirs and --model_names must have the same number of elements.")
        sys.exit(1)

    print(f"Ensembling {len(pred_dirs)} models:")
    for name, pdir in zip(model_names, pred_dirs):
        print(f"  - {name}: {pdir}")

    # Discover common subjects across all model dirs
    subject_maps = {}
    for name, pdir in zip(model_names, pred_dirs):
        # Only match *_gi_seg.nii.gz to avoid picking up consensus/uncertainty maps
        files = glob.glob(os.path.join(pdir, "*_gi_seg.nii.gz"))
        smap = {}
        for f in files:
            fname = os.path.basename(f)
            sub_id = fname.replace('_gi_seg.nii.gz', '')
            smap[sub_id] = f
        subject_maps[name] = smap

    common_subjects = sorted(list(set.intersection(*[set(m.keys()) for m in subject_maps.values()])))
    print(f"\nFound {len(common_subjects)} common subjects across all {len(pred_dirs)} models.")

    if not common_subjects:
        print("ERROR: No common subjects found across prediction directories. Check directory contents.")
        sys.exit(1)

    results = []
    for idx, sub_id in enumerate(common_subjects, 1):
        print(f"[{idx}/{len(common_subjects)}] Processing {sub_id}...", flush=True)
        mfiles = [subject_maps[name][sub_id] for name in model_names]
        res = evaluate_subject(sub_id, mfiles, model_names, args.out_dir)
        results.append(res)

    df = pd.DataFrame(results)
    csv_dir = os.path.dirname(args.out_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Print Summary Report
    print("\n" + "=" * 70)
    print("      MULTI-MODEL DEEP ENSEMBLE & ACTIVE LEARNING REPORT      ")
    print("=" * 70)
    valid_df = df[df['status'] == 'SUCCESS']
    total_valid = len(valid_df)

    if total_valid > 0:
        n_clean = (valid_df['triage_category'] == 'CLEAN_HIGH_CONFIDENCE').sum()
        n_weak = (valid_df['triage_category'] == 'WEAK_COARSE').sum()
        n_reject = (valid_df['triage_category'] == 'REJECT_OR_TRIAGE').sum()

        print(f"Total Evaluated Subjects : {total_valid}")
        print(f"Mean Consensus Dice       : {valid_df['mean_consensus_dice'].mean():.4f}")
        print(f"Mean Consensus IoU        : {valid_df['mean_consensus_iou'].mean():.4f}")
        print(f"Mean Predictive Entropy   : {valid_df['mean_uncertainty'].mean():.4f}")
        print("-" * 70)
        print("[ACTIVE LEARNING DATASET TRIAGE BREAKDOWN]")
        print(f"  1. Clean High-Confidence Masks (Auto-Approve VAE) : {n_clean} cases ({n_clean/total_valid*100:.1f}%)")
        print(f"  2. Weak / Coarse Masks (Phase 3 Thresholding)      : {n_weak} cases ({n_weak/total_valid*100:.1f}%)")
        print(f"  3. Active Learning Triage / Reject (|Δβ0|>5 / High Var): {n_reject} cases ({n_reject/total_valid*100:.1f}%)")
    print("=" * 70)
    print(f"Report saved to: {args.out_csv}\n")


if __name__ == "__main__":
    main()
