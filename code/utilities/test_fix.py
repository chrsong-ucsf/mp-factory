import sys, os
SCRATCH = "/mnt/scratch/user/chrsong/mp-factory"
sys.path.insert(0, os.path.join(SCRATCH, "code", "training"))
from train_mednext_phase2 import get_transforms
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, MapTransform

class FixLiarLabelAffined(MapTransform):
    def __init__(self, keys=["label"], image_key="image", target_spacing=(1.5, 1.5, 2.0)):
        super().__init__(keys)
        self.image_key = image_key
        self.target_spacing = target_spacing
    def __call__(self, data):
        d = dict(data)
        img = d[self.image_key]
        for key in self.keys:
            if key in d:
                lbl = d[key]
                if lbl.shape != img.shape:
                    from monai.transforms import Spacing
                    spacer = Spacing(pixdim=self.target_spacing)
                    img_spaced = spacer(img)
                    lbl.affine = img_spaced.affine
        return d

val_transforms_fixed = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    FixLiarLabelAffined(keys=["label"], image_key="image"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
])

ct_path = "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox/BDMAP_00242142/ct.nii.gz"
lbl_path = "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out/BDMAP_00242142_consensus.nii.gz"

print("Running original val_transforms...")
_, orig_tf = get_transforms()
ds_orig = Dataset(data=[{"image": ct_path, "label": lbl_path}], transform=orig_tf)
item_orig = ds_orig[0]
print("Orig Label Shape:", tuple(item_orig["label"].shape))
import numpy as np
fg_orig = np.sum((item_orig["label"].numpy() >= 1) & (item_orig["label"].numpy() <= 4))
print("Orig Label Foreground:", fg_orig)

print("\nRunning fixed val_transforms...")
ds_fixed = Dataset(data=[{"image": ct_path, "label": lbl_path}], transform=val_transforms_fixed)
item_fixed = ds_fixed[0]
print("Fixed Image Shape:", tuple(item_fixed["image"].shape))
print("Fixed Label Shape:", tuple(item_fixed["label"].shape))
fg_fixed = np.sum((item_fixed["label"].numpy() >= 1) & (item_fixed["label"].numpy() <= 4))
print("Fixed Label Foreground:", fg_fixed)
