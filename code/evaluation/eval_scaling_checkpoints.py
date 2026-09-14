#!/usr/bin/env python3
"""
Post-hoc evaluation script for scaling sweep checkpoints.
Loads checkpoints saved under results/scaling_sweep/N*/seed*/best_mednext_phase2.pt
and calculates validation Dice across the 4 GI organs.
"""

import os
import sys
import glob
import csv
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import monai
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance
from monai.data import Dataset

# Add training dir to path to import model creation and transforms
SCRATCH_DEFAULT = "/mnt/scratch/user/chrsong/mp-factory"
sys.path.insert(0, os.path.join(SCRATCH_DEFAULT, "code", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

try:
    from train_mednext_phase2 import create_mednext_v1, get_transforms, FixLiarLabelAffined
except ImportError:
    create_mednext_v1 = None
    get_transforms = None

ORGAN_NAMES = ["stomach", "duodenum", "small_bowel", "colon"]
NUM_CLASSES = 5  # 0: bg, 1: stomach, 2: duodenum, 3: small bowel, 4: colon


def find_validation_cases(data_dir, ensemble_out_dir, audit_csv, max_cases=20):
    cases = []
    if audit_csv and os.path.exists(audit_csv):
        df = pd.read_csv(audit_csv)
        clean = df[df["triage_category"] == "CLEAN_HIGH_CONFIDENCE"]
        for _, row in clean.iterrows():
            sub = str(row["subject_id"])
            ct = os.path.join(data_dir, sub, "ct.nii.gz")
            lbl = os.path.join(ensemble_out_dir, f"{sub}_consensus.nii.gz")
            if os.path.exists(ct) and os.path.exists(lbl):
                cases.append({"image": ct, "label": lbl, "name": sub})
                if len(cases) >= max_cases:
                    break

    if not cases and os.path.exists(ensemble_out_dir):
        for lbl in sorted(glob.glob(os.path.join(ensemble_out_dir, "*_consensus.nii.gz"))):
            sub = os.path.basename(lbl).replace("_consensus.nii.gz", "")
            ct = os.path.join(data_dir, sub, "ct.nii.gz")
            if os.path.exists(ct):
                cases.append({"image": ct, "label": lbl, "name": sub})
                if len(cases) >= max_cases:
                    break

    return cases


def evaluate_checkpoint(model, ckpt_path, cases, val_tf, roi_size, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    model = model.to(device).eval()

    val_ds = Dataset(data=cases, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 4 organ classes (indices 1..4)
    all_dice = [[] for _ in range(4)]

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
            preds = torch.argmax(out, dim=1, keepdim=True)

            # Clean ignore index if any
            lbls_clean = lbls.clone()
            lbls_clean[lbls_clean == 255] = 0

            for c in range(4):
                organ_idx = c + 1
                p = (preds[:, 0, ...] == organ_idx).float()
                g = (lbls_clean[:, 0, ...] == organ_idx).float()
                inter = (p * g).sum().item()
                union = p.sum().item() + g.sum().item()
                if union > 0:
                    dice = (2.0 * inter) / (union + 1e-6)
                    all_dice[c].append(dice)

    per_organ_dice = {
        ORGAN_NAMES[c]: float(np.mean(all_dice[c])) if len(all_dice[c]) > 0 else 0.0
        for c in range(4)
    }
    mean_dice = float(np.mean(list(per_organ_dice.values())))

    return {
        "mean_dice": mean_dice,
        "per_organ_dice": per_organ_dice,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate scaling sweep checkpoints post-hoc.")
    parser.add_argument("--results_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/scaling_sweep")
    parser.add_argument("--data_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument("--ensemble_out_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out")
    parser.add_argument("--audit_csv", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_val_cases", type=int, default=20)
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 70)
    print("      SCALING SWEEP POST-HOC EVALUATION")
    print(f"      Device:     {device}")
    print(f"      Results:    {args.results_dir}")
    print("=" * 70)

    # 1. Find validation cases
    val_cases = find_validation_cases(
        args.data_dir, args.ensemble_out_dir, args.audit_csv, max_cases=args.num_val_cases
    )
    if not val_cases:
        print(f"ERROR: No valid (CT, label) pairs found in {args.data_dir} / {args.ensemble_out_dir}!")
        sys.exit(1)
    print(f"Loaded {len(val_cases)} validation cases for evaluation.")

    # 2. Get transforms with FixLiarLabelAffined
    _, val_tf = get_transforms(roi_size=tuple(args.roi_size))

    # 3. Discover all checkpoints
    # Pattern 1: results/scaling_sweep/N*/seed*/best_mednext_phase2.pt
    ckpts = sorted(glob.glob(os.path.join(args.results_dir, "N*", "seed*", "best_mednext_phase2.pt")))
    if not ckpts:
        # Fallback to any best_*.pt
        ckpts = sorted(glob.glob(os.path.join(args.results_dir, "**", "best_*.pt"), recursive=True))

    if not ckpts:
        print(f"No checkpoints found under {args.results_dir}")
        sys.exit(1)

    print(f"Discovered {len(ckpts)} checkpoints to evaluate.\n")

    # 4. Instantiate Model
    if create_mednext_v1 is None:
        print("ERROR: create_mednext_v1 could not be imported. Ensure MedNeXt is installed.")
        sys.exit(1)

    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=NUM_CLASSES,
        model_id="B",
        kernel_size=3,
        deep_supervision=False,
    ).to(device)

    summary = []
    for ckpt in ckpts:
        rel = os.path.relpath(ckpt, args.results_dir)
        parts = rel.split(os.sep)
        n_folder = parts[0] if len(parts) > 0 else "unknown"
        seed_folder = parts[1] if len(parts) > 1 else "unknown"

        print(f"Evaluating [{n_folder} | {seed_folder}] -> {ckpt} ...")
        res = evaluate_checkpoint(model, ckpt, val_cases, val_tf, tuple(args.roi_size), device)
        print(f"  -> Mean Dice: {res['mean_dice']:.4f}")
        for organ, d in res["per_organ_dice"].items():
            print(f"     {organ:12s}: {d:.4f}")

        row = {
            "cohort": n_folder,
            "seed": seed_folder,
            "checkpoint": ckpt,
            "mean_dice": res["mean_dice"],
            **{f"dice_{k}": v for k, v in res["per_organ_dice"].items()},
        }
        summary.append(row)

    # 5. Output CSV
    out_csv = os.path.join(args.results_dir, "scaling_sweep_posthoc_eval.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 70)
    print(f"  Evaluation Complete! Saved summary to: {out_csv}")
    print("=" * 70)

    # Grouped summary by cohort
    df_sum = pd.DataFrame(summary)
    grouped = df_sum.groupby("cohort")["mean_dice"].agg(["mean", "std", "count"])
    print("\nScaling Curve Aggregated (Mean Dice across seeds):")
    print(grouped.to_string())


if __name__ == "__main__":
    main()
