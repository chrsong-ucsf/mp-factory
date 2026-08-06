import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import argparse
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label, euler_number
from sklearn.metrics import adjusted_rand_score
from skimage.metrics import variation_of_information, adapted_rand_error

# Organ label mapping for GI tract
ORGAN_MAP = {
    1: 'stomach',
    2: 'duodenum',
    3: 'small_bowel',
    4: 'colon'
}

def compute_betti_0(mask):
    """Compute Betti-0 (number of connected components)."""
    if not np.any(mask):
        return 0
    labeled_array, num_features = label(mask, return_num=True)
    return num_features

def compute_clustering_metrics(gt_arr, pred_arr):
    gt_flat = gt_arr.ravel()
    pred_flat = pred_arr.ravel()
    subsample_idx = slice(None, None, 50)
    gt_sub = gt_flat[subsample_idx]
    pred_sub = pred_flat[subsample_idx]
    ari = adjusted_rand_score(gt_sub, pred_sub)
    split, merge = variation_of_information(gt_sub, pred_sub)
    voi = split + merge
    return float(ari), float(voi)

def compute_hd95(mask_gt, mask_pred, voxel_spacing=(1.0, 1.0, 1.0)):
    """Compute 95th percentile Hausdorff Distance (HD95)."""
    if not np.any(mask_gt) or not np.any(mask_pred):
        return np.nan
    
    pts_gt = np.argwhere(mask_gt) * np.array(voxel_spacing)
    pts_pred = np.argwhere(mask_pred) * np.array(voxel_spacing)
    
    d_gt_pred = [np.min(np.linalg.norm(pts_gt - p, axis=1)) for p in pts_pred[::10]] # Subsampled for speed
    d_pred_gt = [np.min(np.linalg.norm(pts_pred - p, axis=1)) for p in pts_gt[::10]]
    
    return max(np.percentile(d_gt_pred, 95), np.percentile(d_pred_gt, 95))

def evaluate_single_subject(args):
    gt_path, pred_path = args
    subject_id = os.path.basename(gt_path).replace('.nii.gz', '')
    
    results = {'subject_id': subject_id}
    
    if not os.path.exists(pred_path):
        results['status'] = 'missing_prediction'
        return results
        
    try:
        gt_img = nib.load(gt_path)
        pred_img = nib.load(pred_path)
        
        gt_data = gt_img.get_fdata().astype(np.uint8)
        pred_data = pred_img.get_fdata().astype(np.uint8)
        voxel_spacing = gt_img.header.get_zooms()[:3]
        
        for organ_idx, organ_name in ORGAN_MAP.items():
            gt_o = (gt_data == organ_idx)
            pred_o = (pred_data == organ_idx)
            
            intersection = np.sum(gt_o & pred_o)
            total = np.sum(gt_o) + np.sum(pred_o)
            union = total - intersection
            
            dice = (2.0 * intersection) / (total) if total > 0 else (1.0 if not np.any(gt_o) and not np.any(pred_o) else 0.0)
            iou = intersection / union if union > 0 else (1.0 if not np.any(gt_o) and not np.any(pred_o) else 0.0)
            hd95 = compute_hd95(gt_o, pred_o, voxel_spacing)
            betti_gt = compute_betti_0(gt_o)
            betti_pred = compute_betti_0(pred_o)
            betti_diff = abs(betti_gt - betti_pred)
            
            results[f'{organ_name}_dice'] = dice
            results[f'{organ_name}_iou'] = iou
            results[f'{organ_name}_hd95'] = hd95
            results[f'{organ_name}_betti_gt'] = betti_gt
            results[f'{organ_name}_betti_pred'] = betti_pred
            results[f'{organ_name}_betti_diff'] = betti_diff
            
        ari, voi = compute_clustering_metrics(gt_data, pred_data)
        results['overall_ari'] = ari
        results['overall_voi'] = voi
            
        results['status'] = 'SUCCESS'
    except Exception as e:
        results['status'] = f'ERROR: {str(e)}'
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation masks against ground truth")
    parser.add_argument("--gt_dir", type=str,
                        default="/mnt/scratch/user/chrsong/CancerVerse_data/new_database_masks",
                        help="Directory containing ground truth masks")
    parser.add_argument("--pred_dir", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap",
                        help="Directory containing predicted masks")
    parser.add_argument("--output_csv", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/audit_summary.csv",
                        help="Output CSV file for evaluation results")
    parser.add_argument("--gold_standard_dir", type=str, default=None,
                        help="Directory containing gold standard manual annotations for True-Dice evaluation")

    args = parser.parse_args()

    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "*.nii.gz")))
    pair_list = []

    for gt_f in gt_files:
        filename = os.path.basename(gt_f)
        pred_f = os.path.join(args.pred_dir, filename)
        pair_list.append((gt_f, pred_f))

    print(f"Starting audit evaluation for {len(pair_list)} scan pairs...")

    num_workers = min(cpu_count(), 32)
    with Pool(num_workers) as pool:
        eval_results = pool.map(evaluate_single_subject, pair_list)

    df = pd.DataFrame(eval_results)
    df.to_csv(args.output_csv, index=False)
    print(f"Audit completed! Summary report saved to {args.output_csv}")

    # If gold standard directory is provided, evaluate against those manual annotations
    if args.gold_standard_dir:
        print(f"\n=== Evaluating against Gold Standard Manual Annotations ===")
        print(f"Gold standard directory: {args.gold_standard_dir}")

        # Build pairs for gold standard evaluation
        gs_pair_list = []
        for gt_f in gt_files:  # Use same ground truth files (CT scans)
            filename = os.path.basename(gt_f)
            gs_f = os.path.join(args.gold_standard_dir, filename)
            if os.path.exists(gs_f):  # Only include if gold standard exists
                gs_pair_list.append((gt_f, gs_f))
            else:
                print(f"Warning: Gold standard not found for {filename}")

        if gs_pair_list:
            print(f"Evaluating {len(gs_pair_list)} gold standard cases...")
            with Pool(num_workers) as pool:
                gs_results = pool.map(evaluate_single_subject, gs_pair_list)

            gs_df = pd.DataFrame(gs_results)
            gs_output_csv = args.output_csv.replace('.csv', '_gold_standard.csv')
            gs_df.to_csv(gs_output_csv, index=False)
            print(f"Gold standard evaluation completed! Results saved to {gs_output_csv}")

            # Calculate and print summary statistics
            success_mask = gs_df['status'] == 'SUCCESS'
            if success_mask.any():
                success_df = gs_df[success_mask]
                print(f"\n=== Gold Standard Evaluation Summary (True-Dice Metrics) ===")
                print(f"Successfully evaluated: {len(success_df)}/{len(gs_pair_list)} cases")

                # Calculate mean Dice scores for each organ
                for organ_name in ORGAN_MAP.values():
                    dice_col = f'{organ_name}_dice'
                    if dice_col in success_df.columns:
                        mean_dice = success_df[dice_col].mean()
                        std_dice = success_df[dice_col].std()
                        print(f"{organ_name.capitalize()} Dice: {mean_dice:.4f} ± {std_dice:.4f}")

                # Overall metrics
                if 'overall_ari' in success_df.columns:
                    mean_ari = success_df['overall_ari'].mean()
                    std_ari = success_df['overall_ari'].std()
                    print(f"Adjusted Rand Index: {mean_ari:.4f} ± {std_ari:.4f}")

                if 'overall_voi' in success_df.columns:
                    mean_voi = success_df['overall_voi'].mean()
                    std_voi = success_df['overall_voi'].std()
                    print(f"Variation of Information: {mean_voi:.4f} ± {std_voi:.4f}")

                # HD95 metrics
                for organ_name in ORGAN_MAP.values():
                    hd95_col = f'{organ_name}_hd95'
                    if hd95_col in success_df.columns:
                        # Filter out NaN values for HD95
                        valid_hd95 = success_df[hd95_col].dropna()
                        if len(valid_hd95) > 0:
                            mean_hd95 = valid_hd95.mean()
                            std_hd95 = valid_hd95.std()
                            print(f"{organ_name.capitalize()} HD95: {mean_hd95:.2f} ± {std_hd95:.2f}")

                # Betti differences
                for organ_name in ORGAN_MAP.values():
                    betti_col = f'{organ_name}_betti_diff'
                    if betti_col in success_df.columns:
                        mean_betti = success_df[betti_col].mean()
                        std_betti = success_df[betti_col].std()
                        print(f"{organ_name.capitalize()} Betti-0 Diff: {mean_betti:.2f} ± {std_betti:.2f}")
        else:
            print("No gold standard files found for evaluation.")

if __name__ == "__main__":
    main()
