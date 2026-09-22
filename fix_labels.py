import os
import glob
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
import multiprocessing

def fix_gi_mask(sub):
    gi_path = os.path.join(sub, "gi_mask_temp.nii.gz")
    if os.path.exists(gi_path):
        seg_dir = os.path.join(sub, "segmentations")
        if not os.path.exists(seg_dir):
            return
            
        gi_nii = nib.load(gi_path)
        gt_arr = np.asanyarray(gi_nii.dataobj).copy()
        
        changed = False
        # Small bowel aliases
        for alias in ['small_bowel', 'intestine', 'small_intestine']:
            organ_path = os.path.join(seg_dir, f"{alias}.nii.gz")
            if os.path.exists(organ_path):
                i_nii = nib.load(organ_path)
                i_arr = np.asanyarray(i_nii.dataobj) > 0
                if np.any(i_arr & (gt_arr != 3)):
                    gt_arr[i_arr] = 3
                    changed = True
                
        if changed:
            new_gi = nib.Nifti1Image(gt_arr, gi_nii.affine, gi_nii.header)
            nib.save(new_gi, gi_path)
            # print(f"Fixed {gi_path}")

def fix_consensus(c_file):
    sub_id = os.path.basename(c_file).replace('_consensus.nii.gz', '')
    
    totalseg_dirs = ["/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks", "/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap"]
    
    ts_mask_path = None
    for td in totalseg_dirs:
        p = os.path.join(td, f"{sub_id}_gi_mask.nii.gz")
        if os.path.exists(p):
            ts_mask_path = p
            break
            
    if ts_mask_path is None:
        return
        
    c_nii = nib.load(c_file)
    c_arr = np.asanyarray(c_nii.dataobj).copy()
    
    # Small optimization: check if class 3 is already prominent
    if np.sum(c_arr == 3) > 1000:
        return
    
    ts_nii = nib.load(ts_mask_path)
    ts_arr = np.asanyarray(ts_nii.dataobj)
    sb_mask = ((ts_arr == 20) | (ts_arr == 52) | (ts_arr == 57)).astype(np.float32)
    
    if sb_mask.sum() > 0:
        sb_nii = nib.Nifti1Image(sb_mask, ts_nii.affine, ts_nii.header)
        resampled_sb = resample_to_img(sb_nii, c_nii, interpolation='nearest', copy=False)
        res_arr = np.asanyarray(resampled_sb.dataobj).squeeze() > 0
        
        if np.any(res_arr & (c_arr != 3)):
            c_arr[res_arr] = 3
            new_c = nib.Nifti1Image(c_arr.astype(np.uint8), c_nii.affine, c_nii.header)
            nib.save(new_c, c_file)
            # print(f"Fixed {c_file}")

if __name__ == "__main__":
    data_dir = "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"
    sub_dirs = glob.glob(os.path.join(data_dir, "BDMAP_*")) + glob.glob(os.path.join(data_dir, "CV_*"))
    
    print("Fixing gi_mask_temp.nii.gz...")
    with multiprocessing.Pool(16) as p:
        p.map(fix_gi_mask, sub_dirs)
        
    print("Fixing consensus.nii.gz...")
    consensus_dir = "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out"
    consensus_files = glob.glob(os.path.join(consensus_dir, "*_consensus.nii.gz"))
    with multiprocessing.Pool(16) as p:
        p.map(fix_consensus, consensus_files)
        
    print("Done!")
