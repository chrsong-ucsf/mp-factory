import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label, euler_number
from sklearn.metrics import adjusted_rand_score, v_measure_score

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
            
            dice = (2.0 * intersection) / (total) if total > 0 else (1.0 if not np.any(gt_o) and not np.any(pred_o) else 0.0)
            hd95 = compute_hd95(gt_o, pred_o, voxel_spacing)
            betti_gt = compute_betti_0(gt_o)
            betti_pred = compute_betti_0(pred_o)
            betti_diff = abs(betti_gt - betti_pred)
            
            results[f'{organ_name}_dice'] = dice
            results[f'{organ_name}_hd95'] = hd95
            results[f'{organ_name}_betti_gt'] = betti_gt
            results[f'{organ_name}_betti_pred'] = betti_pred
            results[f'{organ_name}_betti_diff'] = betti_diff
            
        results['status'] = 'success'
    except Exception as e:
        results['status'] = f'error: {str(e)}'
        
    return results

def main():
    gt_dir = "/mnt/scratch/user/chrsong/CancerVerse_data/new_database_masks"
    totalseg_dir = "/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap"
    output_csv = "/mnt/scratch/user/chrsong/mp-factory/results/audit_summary.csv"
    
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.nii.gz")))
    pair_list = []
    
    for gt_f in gt_files:
        filename = os.path.basename(gt_f)
        pred_f = os.path.join(totalseg_dir, filename)
        pair_list.append((gt_f, pred_f))
        
    print(f"Starting audit evaluation for {len(pair_list)} scan pairs...")
    
    num_workers = min(cpu_count(), 32)
    with Pool(num_workers) as pool:
        eval_results = pool.map(evaluate_single_subject, pair_list)
        
    df = pd.DataFrame(eval_results)
    df.to_csv(output_csv, index=False)
    print(f"Audit completed! Summary report saved to {output_csv}")

if __name__ == "__main__":
    main()
