import os, nibabel as nib, numpy as np
from nilearn.image import resample_to_img

c_file = '/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out/BDMAP_00242113_consensus.nii.gz'
ts_mask_path = '/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap/BDMAP_00242113_gi_mask.nii.gz'

c_nii = nib.load(c_file)
c_arr = np.asanyarray(c_nii.dataobj).copy()
print(f"Initial 3s: {np.sum(c_arr == 3)}")

ts_nii = nib.load(ts_mask_path)
ts_arr = np.asanyarray(ts_nii.dataobj)
sb_mask = ((ts_arr == 20) | (ts_arr == 52) | (ts_arr == 57)).astype(np.float32)
print(f"sb_mask sum: {sb_mask.sum()}")

sb_nii = nib.Nifti1Image(sb_mask, ts_nii.affine, ts_nii.header)
resampled_sb = resample_to_img(sb_nii, c_nii, interpolation='nearest', copy=False)
res_arr = np.asanyarray(resampled_sb.dataobj).squeeze() > 0
print(f"resampled_sb sum: {res_arr.sum()}")

c_arr[res_arr] = 3
print(f"Final 3s: {np.sum(c_arr == 3)}")
new_c = nib.Nifti1Image(c_arr.astype(np.uint8), c_nii.affine, c_nii.header)
nib.save(new_c, c_file)
