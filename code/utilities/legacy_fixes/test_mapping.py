import nibabel as nib, numpy as np, os

ORGAN_ALIASES = {
    1: ['stomach'],
    2: ['duodenum'],
    3: ['small_bowel', 'intestine', 'small_intestine'],
    4: ['colon']
}

img_path = '/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox/BDMAP_00242113/ct.nii.gz'
search_dir = '/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox/BDMAP_00242113/segmentations'

ct_nii = nib.load(img_path)
ct_arr = np.asanyarray(ct_nii.dataobj)
gt_arr = np.zeros_like(ct_arr, dtype=np.uint8)

for organ_id, aliases in ORGAN_ALIASES.items():
    for alias in aliases:
        organ_file = os.path.join(search_dir, f"{alias}.nii.gz")
        if os.path.exists(organ_file):
            print(f"Found {organ_file}")
            o_nii = nib.load(organ_file)
            o_arr = np.asanyarray(o_nii.dataobj) > 0
            gt_arr[o_arr] = organ_id
            break

unique, counts = np.unique(gt_arr, return_counts=True)
print(dict(zip(unique.astype(int), counts)))
