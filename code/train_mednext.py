import os
import sys
import glob
import re
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn

# CUDA Optimizations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import monai
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandRotated,
    RandFlipd,
    RandGaussianNoised,
    EnsureTyped,
    AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch

# Import MedNeXt from nnunet_mednext package
try:
    from nnunet_mednext import create_mednext_v1
except ImportError:
    print("WARNING: 'nnunet_mednext' is not installed! Installing MedNeXt package automatically...")
    subprocess_cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/MIC-DKFZ/MedNeXt.git"]
    import subprocess
    subprocess.check_call(subprocess_cmd)
    from nnunet_mednext import create_mednext_v1

ORGAN_MAP = {
    1: 'stomach',
    2: 'duodenum',
    3: 'small_bowel',
    4: 'colon'
}

def discover_dataset(data_dir):
    """Find scans and assemble image/label pairs from CancerVerse subfolder structure or paired files."""
    data_pairs = []
    
    # Method 1: Subfolder per subject with ct.nii.gz and segmentations/ organ files
    subfolders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for sub in sorted(subfolders):
        ct_file = os.path.join(sub, "ct.nii.gz")
        if not os.path.exists(ct_file):
            ct_candidates = glob.glob(os.path.join(sub, "*.nii.gz"))
            ct_file = ct_candidates[0] if ct_candidates else None
            
        if not ct_file or not os.path.exists(ct_file):
            continue
            
        # Check for multi-label mask or individual organ segmentations
        seg_dir = os.path.join(sub, "segmentations")
        search_dir = seg_dir if os.path.exists(seg_dir) else sub
        
        # Check if single multi-organ mask exists
        combined_mask = os.path.join(sub, "gi_mask.nii.gz")
        if os.path.exists(combined_mask):
            data_pairs.append({"image": ct_file, "label": combined_mask})
        else:
            # Check if individual organ files exist
            has_organs = any(os.path.exists(os.path.join(search_dir, f"{name}.nii.gz")) for name in ORGAN_MAP.values())
            if has_organs:
                data_pairs.append({"image": ct_file, "label_dir": search_dir})

    # Method 2: Direct paired image/mask files in directory
    if not data_pairs:
        ct_files = sorted(glob.glob(os.path.join(data_dir, "**", "*_ct.nii.gz"), recursive=True))
        for ct_f in ct_files:
            mask_f = ct_f.replace("_ct.nii.gz", "_mask.nii.gz")
            if os.path.exists(mask_f):
                data_pairs.append({"image": ct_f, "label": mask_f})
                
    return data_pairs

class GIDataset(Dataset):
    """Custom Dataset handler to merge separate organ NIfTI files on-the-fly if needed."""
    def __init__(self, data_pairs, transform=None):
        self.data_pairs = data_pairs
        self.transform = transform

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        img_path = item["image"]

        if "label" in item:
            lbl_path = item["label"]
            data_dict = {"image": img_path, "label": lbl_path}
        else:
            ct_nii = nib.load(img_path)
            ct_arr = np.asanyarray(ct_nii.dataobj)
            gt_arr = np.zeros_like(ct_arr, dtype=np.uint8)

            search_dir = item["label_dir"]
            for organ_id, organ_name in ORGAN_MAP.items():
                organ_file = os.path.join(search_dir, f"{organ_name}.nii.gz")
                if os.path.exists(organ_file):
                    o_nii = nib.load(organ_file)
                    o_arr = np.asanyarray(o_nii.dataobj) > 0
                    gt_arr[o_arr] = organ_id

            temp_lbl_path = img_path.replace("ct.nii.gz", "gi_mask_temp.nii.gz")
            if not os.path.exists(temp_lbl_path):
                lbl_nii = nib.Nifti1Image(gt_arr, ct_nii.affine, ct_nii.header)
                nib.save(lbl_nii, temp_lbl_path)
            data_dict = {"image": img_path, "label": temp_lbl_path}

        if self.transform:
            data_dict = self.transform(data_dict)
            
        return data_dict

def get_transforms(roi_size=(96, 96, 96)):
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandRotated(keys=["image", "label"], range_x=0.3, range_y=0.3, range_z=0.3, mode=("bilinear", "nearest"), prob=0.3),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    return train_transforms, val_transforms

def main():
    parser = argparse.ArgumentParser(description="Train 3D MedNeXt-B on GI Tract Segmentation (Project 9)")
    parser.add_argument("--data_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument("--out_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/mednext_models")
    parser.add_argument("--model_id", type=str, default="B", help="MedNeXt variant: S, B, M, or L (default: B)")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size: 3 (3x3x3) or 5 (5x5x5)")
    parser.add_argument("--fold", type=int, default=-1, help="Specific fold index (0..num_folds-1) for parallel SLURM array jobs")
    parser.add_argument("--num_folds", type=int, default=4, help="Total number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} | Total GPU Cores Detected: {num_gpus}")

    # Discover Dataset
    data_pairs = discover_dataset(args.data_dir)
    print(f"Discovered {len(data_pairs)} scanning pairs for training/validation.")

    if len(data_pairs) == 0:
        print("Error: No data pairs found!")
        return

    # Train / Val Split
    np.random.seed(42)
    indices = np.arange(len(data_pairs))
    np.random.shuffle(indices)

    if args.fold >= 0:
        folds = np.array_split(indices, args.num_folds)
        val_indices = folds[args.fold]
        train_indices = np.setdiff1d(indices, val_indices)
        train_pairs = [data_pairs[i] for i in train_indices]
        val_pairs = [data_pairs[i] for i in val_indices]
        print(f"[Fold {args.fold}/{args.num_folds}] Train scans: {len(train_pairs)} | Val scans: {len(val_pairs)}")
    else:
        split_idx = int(0.8 * len(data_pairs))
        train_pairs = [data_pairs[i] for i in indices[:split_idx]]
        val_pairs = [data_pairs[i] for i in indices[split_idx:]]
        print(f"[Standard 80/20 Split] Train scans: {len(train_pairs)} | Val scans: {len(val_pairs)}")

    train_tf, val_tf = get_transforms(roi_size=(96, 96, 96))

    train_ds = GIDataset(train_pairs, transform=train_tf)
    val_ds = GIDataset(val_pairs, transform=val_tf)

    num_workers = min(16, 4 * max(1, num_gpus))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, num_gpus), num_workers=4)

    # Instantiate MedNeXt-B Architecture
    print(f"Initializing MedNeXt (Variant: MedNeXt-{args.model_id}, Kernel: {args.kernel_size}x{args.kernel_size}x{args.kernel_size})...")
    model = create_mednext_v1(
        num_channels=1,
        num_classes=5,  # Background + 4 GI organs
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        deep_supervision=False
    ).to(device)

    # Wrap with DataParallel if multi-GPU is detected
    if num_gpus > 1:
        print(f"[Multi-GPU] Wrapping MedNeXt with DataParallel across {num_gpus} GPUs!")
        model = nn.DataParallel(model)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=5)
    post_label = AsDiscrete(to_onehot=5)

    best_val_dice = -1.0
    best_model_path = os.path.join(args.out_dir, "best_mednext_gi.pt")

    print("\nStarting MedNeXt-B Training Loop...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= max(step, 1)
        print(f"Epoch [{epoch}/{args.epochs}] Loss: {epoch_loss:.4f}")

        # Validation Step
        if epoch % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    with torch.cuda.amp.autocast():
                        val_outputs = model(val_inputs)

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs, y=val_labels)

                mean_val_dice = dice_metric.aggregate().item()
                dice_metric.reset()

                print(f"  --> Validation Mean Dice: {mean_val_dice:.4f}")

                if mean_val_dice > best_val_dice:
                    best_val_dice = mean_val_dice
                    raw_model = model.module if hasattr(model, "module") else model
                    torch.save(raw_model.state_dict(), best_model_path)
                    print(f"  [+] New Best MedNeXt Model Saved! Dice: {best_val_dice:.4f} -> {best_model_path}")

    print(f"\nTraining Complete! Best Validation Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()
