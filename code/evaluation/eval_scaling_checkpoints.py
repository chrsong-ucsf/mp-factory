#!/usr/bin/env python3
"""
Post-hoc evaluation script for scaling sweep checkpoints.
Loads checkpoints saved under results/scaling_sweep/N* and calculates Dice/HD95.
"""

import os
import sys
import glob
import json
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import monai
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
    Lambdad,
)
from monai.data import Dataset

# Local imports for MedNeXt-B
try:
    from code.training.mednext import MedNeXtB
except ImportError:
    try:
        from mednext import MedNeXtB
    except ImportError:
        MedNeXtB = None

ORGAN_NAMES = ["stomach", "duodenum", "small_bowel", "colon"]
NUM_CLASSES = 4


def remap_gi_labels(lbl):
    if torch.is_tensor(lbl):
        has_bdmap = ((lbl == 2) | (lbl == 3) | (lbl == 5) | (lbl == 6)).any()
    else:
        has_bdmap = bool(np.isin(lbl, [2, 3, 5, 6]).any())

    if has_bdmap:
        out = torch.zeros_like(lbl)
        out[lbl == 2] = 1
        out[lbl == 3] = 2
        out[(lbl == 4) | (lbl == 5)] = 3
        out[lbl == 6] = 4
        return out
    return lbl


def build_val_transforms(roi_size=(96, 96, 96)):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys=["label"], func=remap_gi_labels),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])


def find_cases(data_dir, mask_dir):
    cases = []
    for ct_path in sorted(glob.glob(os.path.join(data_dir, "BDMAP_*", "ct.nii.gz"))):
        pid = os.path.basename(os.path.dirname(ct_path))
        mask_path = os.path.join(mask_dir, pid, "gi_mask.nii.gz")
        if os.path.exists(ct_path) and os.path.exists(mask_path):
            cases.append({"image": ct_path, "label": mask_path, "name": pid})
    if not cases:
        for ct_path in sorted(glob.glob(os.path.join(data_dir, "*", "ct.nii.gz"))):
            pid = os.path.basename(os.path.dirname(ct_path))
            mask_path = os.path.join(data_dir, pid, "gi_mask.nii.gz")
            if os.path.exists(mask_path):
                cases.append({"image": ct_path, "label": mask_path, "name": pid})
    return cases


def evaluate_checkpoint(ckpt_path, cases, roi_size, device):
    if MedNeXtB is not None:
        model = MedNeXtB(in_channels=1, n_channels=32, n_classes=NUM_CLASSES + 1)
    else:
        from monai.networks.nets import UNet
        model = UNet(
            spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES + 1,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    model = model.to(device).eval()

    val_ds = Dataset(data=cases, transform=build_val_transforms(roi_size))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    all_dice = [[] for _ in range(NUM_CLASSES)]
    all_hd95 = [[] for _ in range(NUM_CLASSES)]

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

            for c in range(NUM_CLASSES):
                p = (preds[:, 0, ...] == (c + 1)).float()
                g = (lbls[:, 0, ...] == (c + 1)).float()
                inter = (p * g).sum().item()
                union = p.sum().item() + g.sum().item()
                dice = (2.0 * inter / (union + 1e-6))
                all_dice[c].append(dice)

                try:
                    p_t = p.unsqueeze(1)
                    g_t = g.unsqueeze(1)
                    hd = compute_hausdorff_distance(p_t, g_t, percentile=95).item()
                    if not np.isnan(hd) and not np.isinf(hd):
                        all_hd95[c].append(hd)
                except Exception:
                    pass

    per_organ_dice = {ORGAN_NAMES[c]: float(np.mean(all_dice[c])) if all_dice[c] else 0.0
                      for c in range(NUM_CLASSES)}
    per_organ_hd95 = {ORGAN_NAMES[c]: float(np.mean(all_hd95[c])) if all_hd95[c] else float("nan")
                      for c in range(NUM_CLASSES)}
    mean_dice = float(np.mean(list(per_organ_dice.values())))

    return {
        "mean_dice": mean_dice,
        "per_organ_dice": per_organ_dice,
        "per_organ_hd95": per_organ_hd95,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate scaling sweep checkpoints post-hoc.")
    parser.add_argument("--results_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/scaling_sweep")
    parser.add_argument("--data_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument("--rad_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_val_cases", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Evaluation device: {device}")

    val_cases = find_cases(args.data_dir, args.data_dir)[:args.num_val_cases]
    print(f"Evaluating across {len(val_cases)} validation cases.")

    ckpts = sorted(glob.glob(os.path.join(args.results_dir, "N*", "best_checkpoint.pt")))
    if not ckpts:
        print(f"No best_checkpoint.pt found in {args.results_dir}/N*")
        return

    summary = []
    for ckpt in ckpts:
        n_folder = os.path.basename(os.path.dirname(ckpt))
        print(f"\nEvaluating {n_folder} -> {ckpt} ...")
        res = evaluate_checkpoint(ckpt, val_cases, (96, 96, 96), device)
        print(f"  {n_folder} Mean Dice: {res['mean_dice']:.4f}")
        for organ, d in res['per_organ_dice'].items():
            print(f"    {organ:12s}: {d:.4f}")
        summary.append({
            "cohort": n_folder,
            "mean_dice": res["mean_dice"],
            **{f"dice_{k}": v for k, v in res["per_organ_dice"].items()},
            **{f"hd95_{k}": v for k, v in res["per_organ_hd95"].items()},
        })

    out_csv = os.path.join(args.results_dir, "post_hoc_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved post-hoc summary to: {out_csv}")


if __name__ == "__main__":
    main()
