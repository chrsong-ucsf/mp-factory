import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool, cpu_count
from scipy.ndimage import label, distance_transform_edt, binary_erosion
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

def compute_hd95_fast(gt_mask, pred_mask, voxel_spacing=(1.0, 1.0, 1.0)):
    """Fast EDT-based 95th Percentile Hausdorff Distance with proper morphological surface extraction."""
    if not np.any(gt_mask) or not np.any(pred_mask):
        return np.nan
    if np.array_equal(gt_mask, pred_mask):
        return 0.0

    # Extract 1-voxel thick boundary shell using 3D binary erosion
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
    gt_flat = gt_arr.ravel()
    pred_flat = pred_arr.ravel()
    
    # Subsample 1 out of every 50 voxels to optimize execution speed across large 3D arrays
    subsample_idx = slice(None, None, 50)
    gt_sub = gt_flat[subsample_idx]
    pred_sub = pred_flat[subsample_idx]
    
    ari = adjusted_rand_score(gt_sub, pred_sub)
    split, merge = variation_of_information(gt_sub, pred_sub)
    voi = split + merge
    
    return float(ari), float(voi)

def evaluate_case_pair(args):
    gt_path, pred_path = args
    base_name = os.path.basename(gt_path)
    subject_id = re.sub(r'(\.nii\.gz|\.nii)$', '', base_name)

    result = {'subject_id': subject_id, 'gt_path': gt_path, 'pred_path': pred_path}

    try:
        pred_nifti = nib.load(pred_path)
        pred_arr = np.asanyarray(pred_nifti.dataobj).astype(np.uint8)
        spacing = pred_nifti.header.get_zooms()[:3]

        # Load GT array (either single NIfTI or assemble from subfolder segmentations)
        if os.path.isdir(gt_path):
            gt_arr = np.zeros_like(pred_arr, dtype=np.uint8)
            seg_dir = os.path.join(gt_path, "segmentations")
            search_dir = seg_dir if os.path.exists(seg_dir) else gt_path
            
            for organ_id, organ_name in ORGAN_MAP.items():
                organ_file = os.path.join(search_dir, f"{organ_name}.nii.gz")
                if os.path.exists(organ_file):
                    o_nii = nib.load(organ_file)
                    o_arr = np.asanyarray(o_nii.dataobj) > 0
                    gt_arr[o_arr] = organ_id
        else:
            gt_nifti = nib.load(gt_path)
            gt_arr = np.asanyarray(gt_nifti.dataobj).astype(np.uint8)

        # Multi-organ global clustering metrics
        ari, voi = compute_clustering_metrics(gt_arr, pred_arr)
        result['overall_ari'] = round(ari, 4)
        result['overall_voi'] = round(voi, 4)

        eval_dice_list = []

        for organ_id, organ_name in ORGAN_MAP.items():
            gt_o = (gt_arr == organ_id)
            pred_o = (pred_arr == organ_id)

            sum_gt = np.sum(gt_o)
            sum_pred = np.sum(pred_o)
            intersection = np.sum(gt_o & pred_o)

            if sum_gt == 0 and sum_pred == 0:
                dice = 1.0
            else:
                dice = (2.0 * intersection) / (sum_gt + sum_pred)

            betti_gt = compute_betti_0(gt_o)
            betti_pred = compute_betti_0(pred_o)
            betti_diff = abs(betti_gt - betti_pred)

            hd95 = compute_hd95_fast(gt_o, pred_o, spacing)

            union = sum_gt + sum_pred - intersection
            iou = intersection / union if union > 0 else (1.0 if sum_gt == 0 and sum_pred == 0 else 0.0)

            result[f'{organ_name}_dice'] = round(dice, 4)
            result[f'{organ_name}_iou'] = round(iou, 4)
            result[f'{organ_name}_hd95'] = round(hd95, 2) if not np.isnan(hd95) else np.nan
            result[f'{organ_name}_betti_gt'] = betti_gt
            result[f'{organ_name}_betti_pred'] = betti_pred
            result[f'{organ_name}_betti_diff'] = betti_diff

            eval_dice_list.append(dice)

        result['mean_gi_dice'] = round(float(np.mean(eval_dice_list)), 4)
        result['status'] = 'SUCCESS'

    except Exception as e:
        result['status'] = f'ERROR: {str(e)}'

    return result

def discover_pairs(gt_dir, totalseg_dir):
    totalseg_files = glob.glob(os.path.join(totalseg_dir, "*.nii.gz"))
    totalseg_map = {}
    for p in totalseg_files:
        filename = os.path.basename(p)
        case_id = filename.replace('_gi_mask.nii.gz', '').replace('_gi_mask.nii', '')
        totalseg_map[case_id] = p
        # Backup numerical key
        match = re.search(r'(\d{6,8})', case_id)
        if match:
            totalseg_map[match.group(1)] = p

    pairs = []
    # Check for direct case subfolders in gt_dir (e.g. CancerVerse_dbox/BDMAP_XXXXXX)
    subfolders = [os.path.join(gt_dir, d) for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))]
    
    if subfolders:
        for sub in subfolders:
            cid = os.path.basename(sub)
            match = re.search(r'(\d{6,8})', cid)
            num_key = match.group(1) if match else cid
            
            if cid in totalseg_map:
                pairs.append((sub, totalseg_map[cid]))
            elif num_key in totalseg_map:
                pairs.append((sub, totalseg_map[num_key]))
    else:
        # Fallback to direct GT NIfTI files
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "**", "*.nii.gz"), recursive=True))
        for gt_f in gt_files:
            fn = os.path.basename(gt_f)
            cid = fn.replace('.nii.gz', '')
            match = re.search(r'(\d{6,8})', cid)
            num_key = match.group(1) if match else cid
            if cid in totalseg_map:
                pairs.append((gt_f, totalseg_map[cid]))
            elif num_key in totalseg_map:
                pairs.append((gt_f, totalseg_map[num_key]))

    return pairs

def main():
    parser = argparse.ArgumentParser(description="Audit GI Mask Quality against TotalSegmentator Consensus")
    parser.add_argument("--gt_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox", help="Path to new/external database annotations")
    parser.add_argument("--totalseg_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap")
    parser.add_argument("--out_csv", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/phase1_audit_summary.csv")
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    print(f"Scanning for file pairs between:\n GT: {args.gt_dir}\n Pred: {args.totalseg_dir}")
    pairs = discover_pairs(args.gt_dir, args.totalseg_dir)
    print(f"Matched {len(pairs)} scan pairs for evaluation.")

    if len(pairs) == 0:
        print("ERROR: No pairs found! Check directory paths or file naming conventions.")
        return

    workers = min(cpu_count(), args.num_workers)
    print(f"Starting parallel metric computation using {workers} CPU workers...")

    with Pool(workers) as pool:
        results = pool.map(evaluate_case_pair, pairs)

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Audit completed successfully! Detailed report saved to: {args.out_csv}")

if __name__ == "__main__":
    main()