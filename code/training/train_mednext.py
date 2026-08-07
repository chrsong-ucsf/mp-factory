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
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandRotated,
    Rand3DElasticd,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureTyped,
    AsDiscrete
)
from monai.data import Dataset, DataLoader, decollate_batch

import subprocess

# Import MedNeXt with fallbacks
try:
    from nnunet_mednext import create_mednext_v1
except ImportError:
    try:
        from mednext.create_mednext_v1 import create_mednext_v1
    except ImportError:
        create_mednext_v1 = None

# Asymmetric PDCE Loss (recall-optimized partial CE + Dice) from the vault-root src/
try:
    from src.segmentation.losses.asymmetric_loss import AsymmetricPDCELoss
except ImportError:
    try:
        # __file__ = <vault>/02_Projects/mp-factory/code/training/train_mednext.py
        # dirname x3 -> mp-factory (repo root); x5 -> the vault root holding src/.
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _vault_root = os.path.dirname(os.path.dirname(_repo_root))
        if _vault_root not in sys.path:
            sys.path.insert(0, _vault_root)
        from src.segmentation.losses.asymmetric_loss import AsymmetricPDCELoss
    except ImportError:
        AsymmetricPDCELoss = None

NUM_CLASSES = 5      # Background + Stomach + Duodenum + Small Bowel + Colon
IGNORE_INDEX = 255   # Sentinel written by the Task A.2 consensus/auto-label stage

ORGAN_ALIASES = {
    1: ['stomach'],
    2: ['duodenum'],
    3: ['small_bowel', 'intestine', 'small_intestine'],
    4: ['colon']
}

def discover_dataset(data_dir, gold_standard_dir=None):
    """Find scans and assemble image/label pairs from CancerVerse subfolder structure strictly using ct.nii.gz as input."""
    data_pairs = []

    # Pre-compute set of gold standard subject IDs for efficient lookup
    gold_standard_ids = set()
    if gold_standard_dir and os.path.exists(gold_standard_dir):
        gs_files = glob.glob(os.path.join(gold_standard_dir, "*.nii.gz"))
        for gs_file in gs_files:
            # Extract subject ID from filename (assuming format like SUBJECT_ID.nii.gz or similar)
            filename = os.path.basename(gs_file)
            # Remove .nii.gz extension to get potential subject ID
            subject_id = filename.replace('.nii.gz', '')
            gold_standard_ids.add(subject_id)

    # Subfolder per subject with ct.nii.gz as the input image
    subfolders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for sub in sorted(subfolders):
        ct_file = os.path.join(sub, "ct.nii.gz")

        # STRICT REQUIREMENT: Input image must be ct.nii.gz
        if not os.path.exists(ct_file):
            continue

        # Extract subject ID from the folder path
        subject_id = os.path.basename(sub)

        # Check if this is a gold standard sample
        is_gold_standard = subject_id in gold_standard_ids

        # Check for multi-label mask or individual organ segmentations
        seg_dir = os.path.join(sub, "segmentations")
        search_dir = seg_dir if os.path.exists(seg_dir) else sub

        # Check if single multi-organ mask exists
        combined_mask = os.path.join(sub, "gi_mask.nii.gz")
        if os.path.exists(combined_mask):
            data_pairs.append({
                "image": ct_file,
                "label": combined_mask,
                "is_gold_standard": is_gold_standard
            })
        else:
            # Check if individual organ files exist using aliases
            has_organs = False
            for organ_id, aliases in ORGAN_ALIASES.items():
                if any(os.path.exists(os.path.join(search_dir, f"{alias}.nii.gz")) for alias in aliases):
                    has_organs = True
                    break
            if has_organs:
                data_pairs.append({
                    "image": ct_file,
                    "label_dir": search_dir,
                    "is_gold_standard": is_gold_standard
                })

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
            # Load CT image to get spatial reference shape
            ct_nii = nib.load(img_path)
            ct_arr = np.asanyarray(ct_nii.dataobj)
            gt_arr = np.zeros_like(ct_arr, dtype=np.uint8)

            search_dir = item["label_dir"]
            for organ_id, aliases in ORGAN_ALIASES.items():
                for alias in aliases:
                    organ_file = os.path.join(search_dir, f"{alias}.nii.gz")
                    if os.path.exists(organ_file):
                        o_nii = nib.load(organ_file)
                        o_arr = np.asanyarray(o_nii.dataobj) > 0
                        gt_arr[o_arr] = organ_id
                        break

            # Save temporary merged mask in memory or pass as volume
            temp_lbl_path = os.path.join(os.path.dirname(img_path), "gi_mask_temp.nii.gz")
            if not os.path.exists(temp_lbl_path):
                lbl_nii = nib.Nifti1Image(gt_arr, ct_nii.affine, ct_nii.header)
                nib.save(lbl_nii, temp_lbl_path)
            data_dict = {"image": img_path, "label": temp_lbl_path}

        # Add the gold standard flag to the data dict
        if "is_gold_standard" in item:
            data_dict["is_gold_standard"] = item["is_gold_standard"]
        else:
            data_dict["is_gold_standard"] = False

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict

def get_intensity_transform(percentile_clip=True):
    """Intensity normalization for CT.

    Task B.3 specifies clipping to the 1st-99th percentiles, which adapts to
    each scan's own histogram and is robust to scanner/protocol variation and
    to metal or contrast outliers. The fixed HU window (-175, 250) is retained
    as an opt-out for reproducing earlier runs.
    """
    if percentile_clip:
        return ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1.0, upper=99.0,
            b_min=0.0, b_max=1.0,
            clip=True,
            relative=False,
        )
    return ScaleIntensityRanged(
        keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
    )


def get_transforms(roi_size=(96, 96, 96), percentile_clip=True, elastic_prob=0.15):
    """Build MONAI train/val pipelines.

    Augmentations (Task B.3):
      - intensity clipping to the 1st-99th percentiles
      - 3D rotations about all three axes
      - elastic deformations (Rand3DElasticd)
      - flips plus mild intensity jitter and Gaussian noise

    Note every spatial augmentation applies ``nearest`` interpolation to the
    label so integer class indices and the ignore sentinel are never blended
    into invalid intermediate values.
    """
    intensity_tf = get_intensity_transform(percentile_clip)

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        intensity_tf,
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
        # --- 3D rotations about all three axes ---
        RandRotated(
            keys=["image", "label"],
            range_x=0.3, range_y=0.3, range_z=0.3,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            prob=0.3,
        ),
        # --- Elastic deformation: models peristaltic//postural GI variation ---
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 8),
            magnitude_range=(50, 120),
            prob=elastic_prob,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.15),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.15),
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        intensity_tf,
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
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--max_val_samples", type=int, default=50, help="Maximum number of validation scans to evaluate per check for fast validation")
    parser.add_argument("--gold_standard_dir", type=str, default=None,
                        help="Directory containing gold standard manual annotations for 10x loss weighting")
    parser.add_argument("--loss_type", type=str, default="asymmetric",
                        choices=["asymmetric", "dice_ce"],
                        help="Loss: 'asymmetric' (AsymmetricPDCELoss, default) or 'dice_ce' (legacy DiceCELoss)")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Asymmetric false-negative weight (alpha > beta boosts recall)")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Asymmetric false-positive weight (typically 1.0)")
    parser.add_argument("--no_percentile_clip", action="store_true",
                        help="Use the fixed HU window (-175, 250) instead of 1st-99th percentile clipping")
    parser.add_argument("--elastic_prob", type=float, default=0.15,
                        help="Probability of applying 3D elastic deformation (0 disables)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} | Total GPU Cores Detected: {num_gpus}")

    # Discover Dataset
    data_pairs = discover_dataset(args.data_dir, args.gold_standard_dir)
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
        print(f"[Fold {args.fold}/{args.num_folds}] Train scans: {len(train_pairs)} | Full Val pool: {len(val_pairs)}")
    else:
        split_idx = int(0.8 * len(data_pairs))
        train_pairs = [data_pairs[i] for i in indices[:split_idx]]
        val_pairs = [data_pairs[i] for i in indices[split_idx:]]
        print(f"[Standard 80/20 Split] Train scans: {len(train_pairs)} | Full Val pool: {len(val_pairs)}")

    # Subsample Validation scans for fast evaluation
    if args.max_val_samples > 0 and len(val_pairs) > args.max_val_samples:
        np.random.seed(42)
        val_eval_indices = np.random.choice(len(val_pairs), args.max_val_samples, replace=False)
        val_pairs_eval = [val_pairs[i] for i in val_eval_indices]
        print(f"  [Fast Val] Subsampled {len(val_pairs_eval)} validation scans for evaluation per check.")
    else:
        val_pairs_eval = val_pairs

    train_tf, val_tf = get_transforms(
        roi_size=(96, 96, 96),
        percentile_clip=not args.no_percentile_clip,
        elastic_prob=args.elastic_prob,
    )

    train_ds = GIDataset(train_pairs, transform=train_tf)
    val_ds = GIDataset(val_pairs_eval, transform=val_tf)

    num_workers = min(16, 4 * max(1, num_gpus))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, num_gpus), num_workers=4)

    # Instantiate MedNeXt-B Architecture
    if create_mednext_v1 is None:
        print("ERROR: MedNeXt (nnunet_mednext) is not installed in this environment.")
        print("Please install it manually on the login node: pip install git+https://github.com/MIC-DKFZ/MedNeXt.git")
        sys.exit(1)

    print(f"Initializing MedNeXt (Variant: MedNeXt-{args.model_id}, Kernel: {args.kernel_size}x{args.kernel_size}x{args.kernel_size})...")
    model = create_mednext_v1(
        num_input_channels=1,   # correct MedNeXt API argument name
        num_classes=5,          # Background + 4 GI organs
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        deep_supervision=False
    ).to(device)

    # Wrap with DataParallel if multi-GPU is detected
    if num_gpus > 1:
        print(f"[Multi-GPU] Wrapping MedNeXt with DataParallel across {num_gpus} GPUs!")
        model = nn.DataParallel(model)

    # --- Loss selection ---
    if args.loss_type == "asymmetric" and AsymmetricPDCELoss is not None:
        print(f"[Loss] AsymmetricPDCELoss(alpha={args.alpha}, beta={args.beta}, "
              f"ignore_index={IGNORE_INDEX})")
        loss_function = AsymmetricPDCELoss(
            apply_softmax=True,
            ce_weight=0.5,
            dice_weight=1.0,
            alpha=args.alpha,
            beta=args.beta,
            ignore_index=IGNORE_INDEX,
        )
    else:
        if args.loss_type == "asymmetric":
            print("WARNING: AsymmetricPDCELoss import failed; falling back to DiceCELoss.")
        else:
            print("[Loss] DiceCELoss")
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
            is_gold_standard = batch_data.get("is_gold_standard", False)
            # Handle case where is_gold_standard might be a tensor or batch
            if isinstance(is_gold_standard, torch.Tensor):
                # If it's a batch, we need to check if any sample in the batch is gold standard
                # For simplicity, we'll apply weighting if any sample in the batch is gold standard
                # In practice, you might want to weight each sample individually
                is_gold_any = is_gold_standard.any().item()
            else:
                is_gold_any = bool(is_gold_standard)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            # Apply 10x weight for gold standard samples
            if is_gold_any:
                loss = loss * 10.0

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
                        val_outputs = sliding_window_inference(
                            inputs=val_inputs,
                            roi_size=(96, 96, 96),
                            sw_batch_size=4,
                            predictor=model
                        )

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
