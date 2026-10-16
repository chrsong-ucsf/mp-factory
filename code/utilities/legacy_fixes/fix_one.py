import os, nibabel as nib, numpy as np
sub = '/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox/BDMAP_00242113'
gi_path = os.path.join(sub, "gi_mask_temp.nii.gz")
seg_dir = os.path.join(sub, "segmentations")
gi_nii = nib.load(gi_path)
gt_arr = np.asanyarray(gi_nii.dataobj).copy()

for alias in ['small_bowel', 'intestine', 'small_intestine']:
    organ_path = os.path.join(seg_dir, f"{alias}.nii.gz")
    if os.path.exists(organ_path):
        print(f"Found {organ_path}")
        i_nii = nib.load(organ_path)
        i_arr = np.asanyarray(i_nii.dataobj) > 0
        print(f"i_arr sum: {i_arr.sum()}")
        print(f"gt_arr sum before: {np.sum(gt_arr == 3)}")
        mask = i_arr & (gt_arr != 3)
        print(f"mask sum: {mask.sum()}")
        if np.any(mask):
            gt_arr[i_arr] = 3
            print(f"gt_arr sum after: {np.sum(gt_arr == 3)}")
