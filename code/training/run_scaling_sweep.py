#!/usr/bin/env python3
"""
run_scaling_sweep.py

Dataset scaling law analysis for GI organ segmentation in mp-factory.

Trains MedNeXt-B across sub-cohort sizes N = [4, 10, 15, 20, 30, 50] with
Asymmetric Partial-CE/Dice loss (AsymmetricPDCELoss), logging per-epoch Dice, HD95,
and radiologist true-Dice metrics to structured JSON + CSV files.

Usage:
    python code/training/run_scaling_sweep.py \
        --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
        --mask_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
        --cohort_sizes 4 10 15 20 30 50 \
        --loss asymmetric \
        --epochs 30 \
        --radiologist_gt_dir /mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected \
        --output_dir /mnt/scratch/user/chrsong/mp-factory/results/scaling_sweep

    # Smoke-test with synthetic data:
    python code/training/run_scaling_sweep.py \
        --generate_dummy --cohort_sizes 4 10 --epochs 2 --output_dir /tmp/sweep_test
"""

import os
import sys
import glob
import json
import csv
import time
import random
import shutil
import tempfile
import argparse
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader

import monai
from monai.networks.nets import UNet
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
    RandFlipd,
    EnsureTyped,
    Lambdad,
)
from monai.data import Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance
from monai.losses import DiceCELoss

# Robust local imports for MedNeXt-B and AsymmetricPDCELoss within mp-factory
try:
    from code.training.mednext import MedNeXtB
except ImportError:
    try:
        from mednext import MedNeXtB
    except ImportError:
        try:
            from src.segmentation.models.mednext import MedNeXtB
        except ImportError:
            MedNeXtB = None

try:
    from nnunet_mednext import create_mednext_v1
except ImportError:
    try:
        from mednext.create_mednext_v1 import create_mednext_v1
    except ImportError:
        create_mednext_v1 = None

try:
    from code.training.asymmetric_loss import AsymmetricPDCELoss
except ImportError:
    try:
        from asymmetric_loss import AsymmetricPDCELoss
    except ImportError:
        try:
            from src.segmentation.losses.asymmetric_loss import AsymmetricPDCELoss
        except ImportError:
            AsymmetricPDCELoss = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_COHORT_SIZES = [4, 10, 15, 20, 30, 50]
DEFAULT_EPOCHS = 30
DEFAULT_ROI_SIZE = (96, 96, 96)
NUM_CLASSES = 4  # stomach, duodenum, small bowel, colon (label map 1-4)
ORGAN_NAMES = ["stomach", "duodenum", "small_bowel", "colon"]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_cases(data_dir, mask_dir):
    """Return list of {image, label, name} dicts for paired NIfTI cases."""
    cases = []

    # Layout 1: data_dir/BDMAP_*/ct.nii.gz + mask_dir/BDMAP_*/gi_mask.nii.gz
    for ct_path in sorted(glob.glob(os.path.join(data_dir, "BDMAP_*", "ct.nii.gz"))):
        patient_id = os.path.basename(os.path.dirname(ct_path))
        mask_path = os.path.join(mask_dir, patient_id, "gi_mask.nii.gz")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, patient_id, "segmentations", "stomach.nii.gz")
        if os.path.exists(ct_path) and os.path.exists(mask_path):
            cases.append({"image": ct_path, "label": mask_path, "name": patient_id})

    # Layout 2: CancerVerse / mp-factory subject directories (sub/ct.nii.gz + sub/gi_mask.nii.gz)
    if not cases:
        for ct_path in sorted(glob.glob(os.path.join(data_dir, "*", "ct.nii.gz"))):
            patient_id = os.path.basename(os.path.dirname(ct_path))
            mask_path = os.path.join(data_dir, patient_id, "gi_mask.nii.gz")
            if os.path.exists(mask_path):
                cases.append({"image": ct_path, "label": mask_path, "name": patient_id})

    # Layout 3: Flat layout data_dir/*.nii.gz + mask_dir/*.nii.gz
    if not cases:
        for img_path in sorted(glob.glob(os.path.join(data_dir, "*.nii.gz"))):
            fname = os.path.basename(img_path)
            lbl_path = os.path.join(mask_dir, fname)
            if os.path.exists(lbl_path):
                cases.append({"image": img_path, "label": lbl_path, "name": fname.split(".")[0]})

    return cases


def create_dummy_cohort(tmp_dir, n_cases=50):
    """Generate tiny dummy NIfTI volumes for smoke-testing without real data."""
    img_dir = os.path.join(tmp_dir, "images")
    lbl_dir = os.path.join(tmp_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cases = []
    for i in range(n_cases):
        name = f"dummy_{i:04d}"
        vol = np.random.randn(32, 32, 32).astype(np.float32)
        lbl = np.zeros((32, 32, 32), dtype=np.int32)
        lbl[4:12, 4:12, 4:12] = 1   # stomach
        lbl[12:18, 4:12, 4:12] = 2  # duodenum
        lbl[18:24, 4:12, 4:12] = 3  # small bowel
        lbl[24:30, 4:12, 4:12] = 4  # colon

        img_path = os.path.join(img_dir, f"{name}.nii.gz")
        lbl_path = os.path.join(lbl_dir, f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(vol, np.eye(4)), img_path)
        nib.save(nib.Nifti1Image(lbl, np.eye(4)), lbl_path)
        cases.append({"image": img_path, "label": lbl_path, "name": name})

    return cases


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def remap_gi_labels(lbl):
    """
    Remaps BDMAP / AbdomenAtlas GI labels to 4-class scaling baseline:
      2 (stomach)      -> 1
      3 (duodenum)     -> 2
      4 (jejunum)      -> 3
      5 (ileum)        -> 3 (smush jejunum+ileum -> small_bowel)
      6 (colon)        -> 4
    If already mapped to 1..4, preserves 1..4.
    """
    if (lbl == 5).any() or (lbl == 6).any():
        out = torch.zeros_like(lbl)
        out[lbl == 2] = 1
        out[lbl == 3] = 2
        out[(lbl == 4) | (lbl == 5)] = 3
        out[lbl == 6] = 4
        return out
    return lbl


def build_transforms(roi_size=DEFAULT_ROI_SIZE, is_train=True):
    shared = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys=["label"], func=remap_gi_labels),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ]
    if is_train:
        shared.insert(5, CropForegroundd(keys=["image", "label"], source_key="image"))
        shared.insert(6, SpatialPadd(keys=["image", "label"], spatial_size=roi_size))
        shared.insert(7, RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1, neg=1, num_samples=2,
            image_key="image", image_threshold=0,
        ))
        shared.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0))
        shared.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1))
        shared.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2))
    return Compose(shared)


def build_model(num_classes, device):
    """Build MedNeXt-B; fall back to create_mednext_v1 or UNet if needed."""
    model = None
    if MedNeXtB is not None:
        try:
            model = MedNeXtB(in_channels=1, n_channels=32, n_classes=num_classes + 1)
        except Exception:
            pass

    if model is None and create_mednext_v1 is not None:
        try:
            model = create_mednext_v1(
                num_classes=num_classes + 1,
                model_id="B",
                kernel_size=3,
                deep_supervision=False,
            )
        except Exception:
            pass

    if model is None:
        print("Warning: MedNeXt import failed — falling back to MONAI UNet.")
        model = UNet(
            spatial_dims=3, in_channels=1, out_channels=num_classes + 1,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
        )

    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Enabling torch.nn.DataParallel across {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    return model



def compute_dice_hd95(pred_one_hot, label_one_hot, num_classes):
    """Returns per-class Dice and HD95 lists."""
    dice_vals, hd95_vals = [], []
    for c in range(num_classes):
        p = pred_one_hot[:, c:c+1, ...]
        g = label_one_hot[:, c:c+1, ...]
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        dice = (2.0 * inter / (union + 1e-6)).item()
        dice_vals.append(dice)

        hd95 = float("nan")
        try:
            hd_t = compute_hausdorff_distance(p, g, percentile=95)
            if not (torch.isnan(hd_t).all() or torch.isinf(hd_t).all()):
                hd95 = hd_t.item()
        except Exception:
            pass
        hd95_vals.append(hd95)

    return dice_vals, hd95_vals


def evaluate_radiologist_true_dice(model, rad_cases, roi_size, device):
    """
    Evaluates model predictions against expert radiologist GT cases.
    Returns mean radiologist true-Dice score across available radiologist GT cases.
    """
    if not rad_cases:
        return None
    try:
        model.eval()
        val_ds = Dataset(data=rad_cases, transform=build_transforms(roi_size, is_train=False))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        dices = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                lbls = batch["label"].to(device)
                out = sliding_window_inference(
                    inputs=imgs, roi_size=roi_size,
                    sw_batch_size=1, predictor=model, overlap=0.25,
                )
                if isinstance(out, (tuple, list)):
                    out = out[0]
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True)

                pred_oh = torch.zeros(imgs.shape[0], NUM_CLASSES, *imgs.shape[2:], device=device)
                lbl_oh  = torch.zeros_like(pred_oh)
                for c in range(NUM_CLASSES):
                    pred_oh[:, c, ...] = (preds[:, 0, ...] == (c + 1)).float()
                    lbl_oh[:, c, ...]  = (lbls[:, 0, ...] == (c + 1)).float()

                dice_vals, _ = compute_dice_hd95(pred_oh, lbl_oh, NUM_CLASSES)
                dices.append(np.mean(dice_vals))

        return float(np.mean(dices)) if dices else None
    except Exception as err:
        print(f"  Warning: radiologist true-Dice evaluation skipped ({err})")
        return None


# ---------------------------------------------------------------------------
# Per-N training run
# ---------------------------------------------------------------------------

def run_single_sweep(
    n_cases: int,
    all_cases: list,
    output_dir: str,
    epochs: int,
    roi_size: tuple,
    device: torch.device,
    loss_name: str = "asymmetric",
    alpha: float = 2.0,
    beta: float = 1.0,
    rad_cases: list = None,
    seed: int = 42,
):
    rng = random.Random(seed)
    if n_cases > len(all_cases):
        print(f"  [N={n_cases}] Only {len(all_cases)} cases available; using all.")
        n_cases = len(all_cases)

    cohort = rng.sample(all_cases, n_cases)

    # 80/20 train/val split (minimum 1 val case)
    n_val = max(1, int(0.2 * n_cases))
    n_train = n_cases - n_val
    train_cases = cohort[:n_train]
    val_cases = cohort[n_train:]

    print(f"\n{'='*60}")
    print(f"  Scaling Sweep: N={n_cases}  |  train={n_train}  val={n_val}  loss={loss_name}")
    print(f"{'='*60}")

    train_ds = Dataset(data=train_cases, transform=build_transforms(roi_size, is_train=True))
    val_ds   = Dataset(data=val_cases,   transform=build_transforms(roi_size, is_train=False))

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=monai.data.list_data_collate,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(NUM_CLASSES, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    if loss_name == "asymmetric" and AsymmetricPDCELoss is not None:
        loss_fn = AsymmetricPDCELoss(apply_softmax=True, alpha=alpha, beta=beta)
    else:
        if loss_name == "asymmetric":
            print("Warning: AsymmetricPDCELoss not imported; falling back to DiceCELoss.")
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    # Output paths
    n_dir = os.path.join(output_dir, f"N{n_cases:04d}")
    os.makedirs(n_dir, exist_ok=True)
    json_path = os.path.join(n_dir, "epoch_metrics.json")
    csv_path  = os.path.join(n_dir, "epoch_metrics.csv")

    epoch_records = []
    csv_header = (
        ["epoch", "train_loss"]
        + [f"val_dice_{o}" for o in ORGAN_NAMES]
        + [f"val_hd95_{o}" for o in ORGAN_NAMES]
        + ["val_mean_dice", "val_mean_hd95", "val_true_dice"]
    )

    csv_rows = []
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(imgs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # ---- Validate ----
        model.eval()
        all_dice, all_hd95 = [[] for _ in range(NUM_CLASSES)], [[] for _ in range(NUM_CLASSES)]

        with torch.no_grad():
            for vbatch in val_loader:
                v_imgs = vbatch["image"].to(device)
                v_lbls = vbatch["label"].to(device)

                v_out = sliding_window_inference(
                    inputs=v_imgs, roi_size=roi_size,
                    sw_batch_size=1, predictor=model, overlap=0.25,
                )
                if isinstance(v_out, (tuple, list)):
                    v_out = v_out[0]

                probs = torch.softmax(v_out, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True)

                pred_oh = torch.zeros(
                    v_imgs.shape[0], NUM_CLASSES, *v_imgs.shape[2:], device=device
                )
                lbl_oh  = torch.zeros_like(pred_oh)
                for c in range(NUM_CLASSES):
                    pred_oh[:, c, ...] = (preds[:, 0, ...] == (c + 1)).float()
                    lbl_oh[:, c, ...]  = (v_lbls[:, 0, ...] == (c + 1)).float()

                dice_vals, hd95_vals = compute_dice_hd95(pred_oh, lbl_oh, NUM_CLASSES)
                for c in range(NUM_CLASSES):
                    all_dice[c].append(dice_vals[c])
                    if not np.isnan(hd95_vals[c]):
                        all_hd95[c].append(hd95_vals[c])

        per_organ_dice = [float(np.mean(all_dice[c])) if all_dice[c] else 0.0
                          for c in range(NUM_CLASSES)]
        per_organ_hd95 = [float(np.mean(all_hd95[c])) if all_hd95[c] else float("nan")
                          for c in range(NUM_CLASSES)]
        mean_dice = float(np.mean(per_organ_dice))
        mean_hd95 = float(np.nanmean(per_organ_hd95)) if any(not np.isnan(v) for v in per_organ_hd95) else float("nan")

        true_dice = evaluate_radiologist_true_dice(model, rad_cases, roi_size, device)

        record = {
            "epoch": epoch,
            "n_cases": n_cases,
            "train_loss": round(avg_loss, 6),
            "val_dice": {ORGAN_NAMES[c]: round(per_organ_dice[c], 6) for c in range(NUM_CLASSES)},
            "val_hd95": {ORGAN_NAMES[c]: round(per_organ_hd95[c], 4) for c in range(NUM_CLASSES)},
            "val_mean_dice": round(mean_dice, 6),
            "val_mean_hd95": round(mean_hd95, 4) if not np.isnan(mean_hd95) else None,
            "val_true_dice": round(true_dice, 6) if true_dice is not None else None,
        }
        epoch_records.append(record)

        csv_row = (
            [epoch, round(avg_loss, 6)]
            + [round(per_organ_dice[c], 6) for c in range(NUM_CLASSES)]
            + [round(per_organ_hd95[c], 4) if not np.isnan(per_organ_hd95[c]) else ""
               for c in range(NUM_CLASSES)]
            + [round(mean_dice, 6), round(mean_hd95, 4) if not np.isnan(mean_hd95) else "",
               round(true_dice, 6) if true_dice is not None else ""]
        )
        csv_rows.append(csv_row)

        td_str = f" | true_dice={true_dice:.4f}" if true_dice is not None else ""
        print(
            f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
            f"mean_dice={mean_dice:.4f} | mean_hd95={mean_hd95:.2f}{td_str}"
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_mean_dice": mean_dice, "n_cases": n_cases},
                os.path.join(n_dir, "best_checkpoint.pt"),
            )

        with open(json_path, "w") as f:
            json.dump({"n_cases": n_cases, "epochs": epoch_records}, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)

    print(f"  Saved metrics → {json_path}")
    print(f"  Saved metrics → {csv_path}")
    print(f"  Best val Dice (N={n_cases}): {best_dice:.4f}")
    return json_path


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def write_sweep_summary(sweep_results: list, output_dir: str):
    summary_json_path = os.path.join(output_dir, "sweep_summary.json")
    summary_csv_path  = os.path.join(output_dir, "sweep_summary.csv")

    summary = []
    for res in sweep_results:
        n = res["n_cases"]
        records = res["epochs"]
        if not records:
            continue
        best = max(records, key=lambda r: r["val_mean_dice"])
        summary.append({
            "n_cases": n,
            "best_epoch": best["epoch"],
            "best_val_mean_dice": best["val_mean_dice"],
            "best_val_mean_hd95": best["val_mean_hd95"],
            "best_val_true_dice": best.get("val_true_dice"),
            **{f"best_dice_{o}": best["val_dice"][o] for o in ORGAN_NAMES},
            **{f"best_hd95_{o}": best["val_hd95"][o] for o in ORGAN_NAMES},
        })

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    if summary:
        header = list(summary[0].keys())
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(summary)

    print(f"\nSweep summary saved → {summary_json_path}")
    print(f"Sweep summary saved → {summary_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Dataset scaling law sweep for GI segmentation.")
    parser.add_argument("--data_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument("--mask_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument(
        "--cohort_sizes", type=int, nargs="+", default=DEFAULT_COHORT_SIZES,
        help="List of sub-cohort sizes N to sweep (default: 4 10 15 20 30 50)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--roi_size", type=str, default="96,96,96")
    parser.add_argument(
        "--loss", type=str, default="asymmetric", choices=["asymmetric", "dice_ce"],
        help="Loss function to use: 'asymmetric' (AsymmetricPDCELoss) or 'dice_ce' (DiceCELoss)",
    )
    parser.add_argument("--alpha", type=float, default=2.0, help="Alpha weight for false negatives in AsymmetricPDCELoss")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta weight for false positives in AsymmetricPDCELoss")
    parser.add_argument("--radiologist_gt_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected")
    parser.add_argument("--output_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/scaling_sweep")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generate_dummy", action="store_true",
        help="Generate synthetic dummy data for smoke-testing.",
    )
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ROI size
    roi_size = tuple(int(x) for x in args.roi_size.split(","))
    assert len(roi_size) == 3, "roi_size must have exactly 3 values"

    # Data
    tmp_dir = None
    rad_cases = []
    if args.generate_dummy or not (
        os.path.exists(args.data_dir) and os.path.exists(args.mask_dir)
    ):
        print("Generating dummy cohort for smoke-test …")
        tmp_dir = tempfile.mkdtemp()
        max_n = max(args.cohort_sizes)
        all_cases = create_dummy_cohort(tmp_dir, n_cases=max_n + 5)
        rad_dir = os.path.join(tmp_dir, "rad_gt")
        rad_cases = create_dummy_cohort(rad_dir, n_cases=3)
    else:
        all_cases = find_cases(args.data_dir, args.mask_dir)
        if not all_cases:
            print(f"Error: no cases found in {args.data_dir}")
            sys.exit(1)
        if os.path.exists(args.radiologist_gt_dir):
            rad_cases = find_cases(args.radiologist_gt_dir, args.radiologist_gt_dir)

    print(f"Total cases available: {len(all_cases)}")
    if rad_cases:
        print(f"Radiologist GT cases available for true-Dice validation: {len(rad_cases)}")

    os.makedirs(args.output_dir, exist_ok=True)

    sweep_results = []
    t0 = time.time()
    for n in sorted(set(args.cohort_sizes)):
        json_path = run_single_sweep(
            n_cases=n,
            all_cases=all_cases,
            output_dir=args.output_dir,
            epochs=args.epochs,
            roi_size=roi_size,
            device=device,
            loss_name=args.loss,
            alpha=args.alpha,
            beta=args.beta,
            rad_cases=rad_cases,
            seed=args.seed,
        )
        with open(json_path) as f:
            sweep_results.append(json.load(f))

    write_sweep_summary(sweep_results, args.output_dir)

    elapsed = time.time() - t0
    print(f"\nFull scaling sweep complete in {elapsed/60:.1f} min.")

    if tmp_dir:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
