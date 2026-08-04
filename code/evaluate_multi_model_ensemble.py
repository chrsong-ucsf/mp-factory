"""
evaluate_multi_model_ensemble.py

Multi-Model Deep Ensemble Evaluation, Consensus Assembly, and Automated Data Cleansing.

This script:
  1. Loads multi-organ segmentation masks or softmax probability maps from N independent models.
  2. Computes a Diversity-Promoting Weighting Strategy via a pairwise inter-model Dice similarity matrix.
  3. Generates voxel-wise Spatial Predictive Uncertainty Heatmaps (entropy/variance across models).
  4. Assembles a weighted consensus "pseudo-ground truth" mask (<subject_id>_consensus.nii.gz).
  5. Evaluates model and consensus quality via Dice, IoU, HD95, Betti-0 Count Difference (|Δβ0|), ARI, and VOI.
  6. Routes scans into CLEAN_HIGH_CONFIDENCE, WEAK_COARSE, or NOISE_REJECT buckets
     for fully-automated data cleansing (no human-in-the-loop radiologist review).

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
import gc
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from scipy.ndimage import label, binary_erosion, distance_transform_edt, zoom
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


def compute_classification_metrics(gt_mask, pred_mask):
    """Compute Precision, Sensitivity (Recall), and Specificity."""
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum((~gt_mask) & pred_mask)
    fn = np.sum(gt_mask & (~pred_mask))
    tn = np.sum((~gt_mask) & (~pred_mask))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0 else 0.0)
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else (1.0 if np.sum(gt_mask) == 0 and np.sum(pred_mask) == 0 else 0.0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0

    return precision, sensitivity, specificity


def compute_volumetric_metrics(gt_mask, pred_mask, voxel_spacing=(1.5, 1.5, 2.0)):
    """Compute Volumetric Overlap Error (VOE) and Relative Voxel Difference (RVD)."""
    v_gt = float(np.sum(gt_mask)) * np.prod(voxel_spacing) / 1000.0  # in mL
    v_pred = float(np.sum(pred_mask)) * np.prod(voxel_spacing) / 1000.0  # in mL

    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)

    voe = float(1.0 - (intersection / union)) if union > 0 else (0.0 if v_gt == 0 and v_pred == 0 else 1.0)
    rvd = float((v_pred - v_gt) / v_gt) if v_gt > 0 else (0.0 if v_pred == 0 else np.nan)

    return voe, rvd, v_gt, v_pred


def compute_surface_metrics(gt_mask, pred_mask, voxel_spacing=(1.5, 1.5, 2.0), tolerance_mm=2.0):
    """Compute Average Surface Distance (ASD) and Normalized Surface Distance (NSD)."""
    if not np.any(gt_mask) or not np.any(pred_mask):
        return np.nan, np.nan
    if np.array_equal(gt_mask, pred_mask):
        return 0.0, 1.0

    gt_border = gt_mask ^ binary_erosion(gt_mask)
    pred_border = pred_mask ^ binary_erosion(pred_mask)

    if not np.any(gt_border) or not np.any(pred_border):
        return np.nan, np.nan

    dt_gt = distance_transform_edt(~gt_border, sampling=voxel_spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)

    dist_gt_to_pred = dt_pred[gt_border]
    dist_pred_to_gt = dt_gt[pred_border]

    asd = float(0.5 * (np.mean(dist_gt_to_pred) + np.mean(dist_pred_to_gt)))

    # NSD: fraction of surface points within tolerance_mm
    nsd_gt = np.mean(dist_gt_to_pred <= tolerance_mm)
    nsd_pred = np.mean(dist_pred_to_gt <= tolerance_mm)
    nsd = float(0.5 * (nsd_gt + nsd_pred))

    return asd, nsd


def compute_composite_score(dice, hd95, max_hd_cap=50.0):
    """
    UW-Madison GI Tract Benchmark Composite Score:
    Score = 0.4 * Dice + 0.6 * max(0, 1.0 - HD95 / max_hd_cap)
    """
    if np.isnan(hd95):
        hd_part = 0.0
    else:
        hd_part = max(0.0, 1.0 - (hd95 / max_hd_cap))
    return float(0.4 * dice + 0.6 * hd_part)


def compute_ece(probs, true_labels, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()

    confidences = np.max(probs, axis=0).ravel()
    predictions = np.argmax(probs, axis=0).ravel()
    accuracies = (predictions == true_labels.ravel())

    # Subsample 1 in 50 for speed
    confidences = confidences[::50]
    accuracies = accuracies[::50]

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i+1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


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

    consensus_path = os.path.join(out_dir, f"{subject_id}_consensus.nii.gz")
    uncertainty_path = os.path.join(out_dir, f"{subject_id}_uncertainty.nii.gz")
    skip_saving = os.path.exists(consensus_path) and os.path.exists(uncertainty_path)

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

        # Ensure all model masks share the same 3D spatial shape (e.g. handle TotalSegmentator
        # native-resolution masks vs MedNeXt/Swin-UNETR 1.5x1.5x2.0mm resampled masks).
        target_shape = loaded_masks[0].shape
        resampled_masks = []
        for arr in loaded_masks:
            if arr.shape != target_shape:
                zoom_factors = [t / s for t, s in zip(target_shape, arr.shape)]
                arr = zoom(arr, zoom_factors, order=0).astype(np.uint8)
            resampled_masks.append(arr)

        model_masks = np.stack(resampled_masks, axis=0)  # (N_models, X, Y, Z)

        # 1. Diversity Weights
        weights = compute_diversity_weights(model_masks)
        for name, w in zip(model_names, weights):
            res[f'weight_{name}'] = round(float(w), 4)

        # 2. Always build per-model probability tensors — required for ECE even when
        #    consensus/uncertainty files already exist (skip_saving=True on resume).
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

        one_hot_tensors = []
        for i in range(len(model_files)):
            m = model_masks[i]
            fpath = model_files[i]
            prob_path = fpath.replace('_gi_seg.nii.gz', '_gi_probs.npz') if fpath.endswith('_gi_seg.nii.gz') else None
            if prob_path and os.path.exists(prob_path):
                probs = np.load(prob_path, allow_pickle=True)['probs'].astype(np.float32)
                if probs.shape[1:] != target_shape:
                    zoom_factors = [1.0] + [t / s for t, s in zip(target_shape, probs.shape[1:])]
                    probs = zoom(probs, zoom_factors, order=1).astype(np.float32)
                t = torch.tensor(probs, dtype=torch.float32, device=device)
            else:
                m_t = torch.tensor(m, dtype=torch.long, device=device)
                t = torch.zeros((5,) + m.shape, dtype=torch.float32, device=device)
                t.scatter_(0, m_t.unsqueeze(0), 1.0)
            one_hot_tensors.append(t)

        one_hot_stack = torch.stack(one_hot_tensors, dim=0)  # (N_models, 5, X, Y, Z)

        # GPU Weighted Probability Consensus
        weights_reshaped = weights_t.view(-1, 1, 1, 1, 1)
        consensus_probs_t = (one_hot_stack * weights_reshaped).sum(dim=0)  # (5, X, Y, Z)
        consensus_mask_t = torch.argmax(consensus_probs_t, dim=0).to(torch.uint8)  # (X, Y, Z)

        # 3. GPU Spatial Uncertainty Heatmap (Shannon Entropy)
        epsilon = 1e-7
        clamped = torch.clamp(consensus_probs_t, epsilon, 1.0 - epsilon)
        uncertainty_t = -(clamped * torch.log2(clamped)).sum(dim=0)  # (X, Y, Z)

        if skip_saving:
            # Load saved files as the authoritative source for reporting metrics
            # (the saved consensus is from the original run and is the ground truth).
            consensus_mask = nib.load(consensus_path).get_fdata().astype(np.uint8)
            uncertainty_map = nib.load(uncertainty_path).get_fdata()
        else:
            consensus_mask = consensus_mask_t.cpu().numpy()
            uncertainty_map = uncertainty_t.cpu().numpy()
            os.makedirs(out_dir, exist_ok=True)
            nib.save(nib.Nifti1Image(consensus_mask, affine=affine), consensus_path)
            nib.save(nib.Nifti1Image(uncertainty_map.astype(np.float32), affine=affine), uncertainty_path)

        res['mean_uncertainty'] = round(float(np.mean(uncertainty_map)), 4)
        res['max_uncertainty'] = round(float(np.max(uncertainty_map)), 4)

        # 4. Metric Audit per organ against Consensus
        eval_dices = []
        eval_ious = []
        eval_inter_model_dices = []
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

                hd95 = compute_hd95_fast(cons_o, m_o)
                betti_m = compute_betti_0(m_o)
                b_diff = abs(betti_cons - betti_m)
                prec, sens, spec = compute_classification_metrics(cons_o, m_o)
                voe, rvd, v_cons, v_m = compute_volumetric_metrics(cons_o, m_o)
                asd, nsd = compute_surface_metrics(cons_o, m_o)
                composite = compute_composite_score(d, hd95)

                res[f'{organ_name}_dice_{name}'] = round(float(d), 4)
                res[f'{organ_name}_iou_{name}'] = round(float(iou), 4)
                res[f'{organ_name}_hd95_{name}'] = round(float(hd95), 2) if not np.isnan(hd95) else np.nan
                res[f'{organ_name}_precision_{name}'] = round(float(prec), 4)
                res[f'{organ_name}_sensitivity_{name}'] = round(float(sens), 4)
                res[f'{organ_name}_specificity_{name}'] = round(float(spec), 4)
                res[f'{organ_name}_voe_{name}'] = round(float(voe), 4)
                res[f'{organ_name}_rvd_{name}'] = round(float(rvd), 4) if not np.isnan(rvd) else np.nan
                res[f'{organ_name}_asd_{name}'] = round(float(asd), 2) if not np.isnan(asd) else np.nan
                res[f'{organ_name}_nsd_{name}'] = round(float(nsd), 4) if not np.isnan(nsd) else np.nan
                res[f'{organ_name}_composite_{name}'] = round(float(composite), 4)
                res[f'{organ_name}_betti_diff_{name}'] = b_diff

                organ_dices.append(d)
                organ_ious.append(iou)
                organ_betti_diffs.append(b_diff)

            # Pairwise inter-model Dice (eliminates circular model-vs-consensus bias)
            organ_inter_model_dices = []
            if len(model_names) > 1:
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        m_i = (model_masks[i] == organ_id)
                        m_j = (model_masks[j] == organ_id)
                        intersection_ij = np.sum(m_i & m_j)
                        total_ij = np.sum(m_i) + np.sum(m_j)
                        d_ij = (2.0 * intersection_ij) / total_ij if total_ij > 0 else 1.0
                        organ_inter_model_dices.append(d_ij)
            else:
                organ_inter_model_dices = [1.0]

            mean_organ_inter_model_dice = float(np.mean(organ_inter_model_dices))
            res[f'{organ_name}_inter_model_dice'] = round(mean_organ_inter_model_dice, 4)

            mean_org_dice = float(np.mean(organ_dices))
            res[f'{organ_name}_mean_dice'] = round(mean_org_dice, 4)
            res[f'{organ_name}_mean_iou'] = round(float(np.mean(organ_ious)), 4)
            res[f'{organ_name}_max_betti_diff'] = max(organ_betti_diffs)

            eval_dices.append(mean_org_dice)
            eval_ious.append(np.mean(organ_ious))
            eval_inter_model_dices.append(mean_organ_inter_model_dice)
            max_betti_diff = max(max_betti_diff, max(organ_betti_diffs))

        # Overall partition clustering metrics and ECE calibration across models vs consensus
        for i, name in enumerate(model_names):
            ari, voi = compute_clustering_metrics(consensus_mask, model_masks[i])
            ece = compute_ece(one_hot_stack[i], consensus_mask)
            res[f'ari_{name}'] = round(ari, 4)
            res[f'voi_{name}'] = round(voi, 4)
            res[f'ece_{name}'] = round(ece, 4)

        res['mean_consensus_dice'] = round(float(np.mean(eval_dices)), 4)
        res['mean_consensus_iou'] = round(float(np.mean(eval_ious)), 4)
        res['mean_inter_model_dice'] = round(float(np.mean(eval_inter_model_dices)), 4)
        res['max_betti_diff'] = max_betti_diff

        # 5. Categorization / Triage Logic (Automated Data Cleansing)
        # REJECT uses pairwise inter-model Dice (unbiased) as primary gate,
        # avoiding circular model-vs-consensus bias in mean_consensus_dice.
        # Betti threshold tightened: > 3 (was > 5) to prevent broken topology
        # leaking into the WEAK_COARSE pool.
        if (res['mean_consensus_dice'] < 0.82
                or res['mean_inter_model_dice'] < 0.70
                or max_betti_diff > 3
                or res['mean_uncertainty'] > 0.15):
            res['triage_category'] = 'NOISE_REJECT'
            res['action'] = 'Auto-Exclude (Discard from Training Pool)'
        elif (res['mean_consensus_dice'] >= 0.82
                and max_betti_diff <= 2
                and res['mean_inter_model_dice'] >= 0.85):
            res['triage_category'] = 'CLEAN_HIGH_CONFIDENCE'
            res['action'] = 'Auto-Approve for GKD Distillation & VAE'
        else:
            res['triage_category'] = 'WEAK_COARSE'
            res['action'] = 'Apply Hard Thresholding (Set Conflicting Pixels to Ignore Class)'

        res['status'] = 'SUCCESS'

    except Exception as e:
        res['status'] = f'ERROR: {str(e)}'

    gc.collect()
    return res


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Ensemble Consensus & Automated Data Cleansing")
    parser.add_argument("--pred_dirs", type=str, nargs="+", required=True,
                        help="List of output directories for each model")
    parser.add_argument("--model_names", type=str, nargs="+", default=None,
                        help="Names corresponding to model directories")
    parser.add_argument("--out_dir", type=str, default="./results/ensemble_out",
                        help="Output directory for consensus masks and uncertainty maps")
    parser.add_argument("--out_csv", type=str, default="./results/ensemble_audit_summary.csv",
                        help="Output path for evaluation CSV summary")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel CPU worker processes (default: 16)")
    parser.add_argument("--num_chunks", type=int, default=1,
                        help="Total number of parallel array chunks (default: 1)")
    parser.add_argument("--chunk_idx", type=int, default=0,
                        help="Index of current array chunk (0-indexed)")
    args = parser.parse_args()

    pred_dirs = args.pred_dirs
    model_names = args.model_names if args.model_names else [f"model_{i+1}" for i in range(len(pred_dirs))]

    if len(pred_dirs) != len(model_names):
        print("ERROR: --pred_dirs and --model_names must have the same number of elements.")
        sys.exit(1)

    print(f"Ensembling {len(pred_dirs)} models across {args.num_workers} parallel workers:")
    for name, pdir in zip(model_names, pred_dirs):
        print(f"  - {name}: {pdir}")

    # Discover common subjects across all model dirs (supports both _gi_seg and _gi_mask filenames)
    subject_maps = {}
    for name, pdir in zip(model_names, pred_dirs):
        files = glob.glob(os.path.join(pdir, "*_gi_seg.nii.gz"))
        if not files:
            files = glob.glob(os.path.join(pdir, "*_gi_mask.nii.gz"))
        if not files:
            # General fallback for any .nii.gz files if neither pattern matches directly
            files = glob.glob(os.path.join(pdir, "*.nii.gz"))
        smap = {}
        for f in files:
            fname = os.path.basename(f)
            sub_id = fname.replace('_gi_seg.nii.gz', '').replace('_gi_mask.nii.gz', '').replace('.nii.gz', '')
            smap[sub_id] = f
        subject_maps[name] = smap

    common_subjects = sorted(list(set.intersection(*[set(m.keys()) for m in subject_maps.values()])))
    print(f"\nFound {len(common_subjects)} total common subjects across all {len(pred_dirs)} models.")

    if not common_subjects:
        print("ERROR: No common subjects found across prediction directories. Check directory contents.")
        sys.exit(1)

    if args.num_chunks > 1:
        chunk_subjects = [sub for i, sub in enumerate(common_subjects) if i % args.num_chunks == args.chunk_idx]
        print(f"GPU Worker [{args.chunk_idx}/{args.num_chunks}]: Processing {len(chunk_subjects)} / {len(common_subjects)} assigned scans.")
        common_subjects = chunk_subjects

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Deep Ensemble Audit on Device: {device} (Total Subjects: {len(common_subjects)})", flush=True)

    results = []
    total = len(common_subjects)

    if device.type == "cuda":
        # GPU execution: Run in main process to keep CUDA tensors on GPU VRAM
        for completed, sub_id in enumerate(common_subjects, 1):
            mfiles = [subject_maps[name][sub_id] for name in model_names]
            try:
                res = evaluate_subject(sub_id, mfiles, model_names, args.out_dir)
                results.append(res)
            except Exception as e:
                print(f"  [ERROR] Subject {sub_id} failed: {e}", flush=True)

            if completed % 50 == 0 or completed == total:
                print(f"[{completed}/{total}] GPU Progress: {completed/total*100:.1f}% complete", flush=True)
    else:
        # CPU execution: Use ProcessPoolExecutor for multi-core parallelism
        tasks = [(sub_id, [subject_maps[name][sub_id] for name in model_names], model_names, args.out_dir) for sub_id in common_subjects]
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(evaluate_subject, sub_id, mfiles, model_names, out_dir): sub_id
                for (sub_id, mfiles, model_names, out_dir) in tasks
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                sub_id = futures[future]
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"  [ERROR] Subject {sub_id} failed: {e}", flush=True)
                if completed % 50 == 0 or completed == total:
                    print(f"[{completed}/{total}] CPU Progress: {completed/total*100:.1f}% complete", flush=True)

    # Build DataFrame and save CSV — runs for both GPU and CPU paths.
    # Previously this was inside the CPU-only else branch, causing NameError
    # on GPU and silently producing no output file.
    df = pd.DataFrame(results)
    csv_dir = os.path.dirname(args.out_csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nCSV saved: {args.out_csv} ({len(df):,} rows)", flush=True)

    # Print Summary Report
    print("\n" + "=" * 70)
    print("      MULTI-MODEL DEEP ENSEMBLE & AUTOMATED DATA CLEANSING REPORT      ")
    print("=" * 70)
    valid_df = df[df['status'] == 'SUCCESS']
    total_valid = len(valid_df)

    if total_valid > 0:
        n_clean = (valid_df['triage_category'] == 'CLEAN_HIGH_CONFIDENCE').sum()
        n_weak = (valid_df['triage_category'] == 'WEAK_COARSE').sum()
        n_reject = (valid_df['triage_category'] == 'NOISE_REJECT').sum()

        print(f"Total Evaluated Subjects   : {total_valid}")
        print(f"Mean Consensus Dice         : {valid_df['mean_consensus_dice'].mean():.4f}")
        print(f"Mean Inter-Model Dice       : {valid_df['mean_inter_model_dice'].mean():.4f}")
        print(f"Mean Consensus IoU          : {valid_df['mean_consensus_iou'].mean():.4f}")
        print(f"Mean Predictive Entropy     : {valid_df['mean_uncertainty'].mean():.4f}")
        print("-" * 70)
        print("[AUTOMATED DATA CLEANSING BREAKDOWN]")
        print(f"  1. Clean High-Confidence (GKD Distillation + VAE)     : {n_clean} cases ({n_clean/total_valid*100:.1f}%)")
        print(f"  2. Weak / Coarse (Hard Threshold Ignore Class)         : {n_weak} cases ({n_weak/total_valid*100:.1f}%)")
        print(f"  3. Noise Reject (Auto-Excluded from Training Pool)     : {n_reject} cases ({n_reject/total_valid*100:.1f}%)")
    print("=" * 70)
    print(f"Report saved to: {args.out_csv}\n")


if __name__ == "__main__":
    main()
