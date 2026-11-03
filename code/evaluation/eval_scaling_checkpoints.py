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
import json
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
    import importlib
    _phase2 = importlib.import_module("train_mednext_phase2")
    create_mednext_v1 = getattr(_phase2, "create_mednext_v1", None)
    get_transforms = getattr(_phase2, "get_transforms", None)
    FixLiarLabelAffined = getattr(_phase2, "FixLiarLabelAffined", None)
except Exception as _e:
    print(f"[WARNING] Could not import train_mednext_phase2: {_e}")
    create_mednext_v1 = None
    get_transforms = None
    FixLiarLabelAffined = None

ORGAN_NAMES = ["stomach", "duodenum", "small_bowel", "colon"]
NUM_CLASSES = 5  # 0: bg, 1: stomach, 2: duodenum, 3: small bowel, 4: colon


JHU_SCAN_TO_PATIENT_MAP = {
    "BDMAP_00242113": "PATIENT_BDMAP_P1", "BDMAP_00242114": "PATIENT_BDMAP_P1",
    "BDMAP_00242115": "PATIENT_BDMAP_P1", "BDMAP_00242116": "PATIENT_BDMAP_P1",
    "BDMAP_00242131": "PATIENT_BDMAP_P2", "BDMAP_00242132": "PATIENT_BDMAP_P2", "BDMAP_00242133": "PATIENT_BDMAP_P2",
    "BDMAP_00242134": "PATIENT_BDMAP_P3", "BDMAP_00242135": "PATIENT_BDMAP_P4", "BDMAP_00242136": "PATIENT_BDMAP_P4",
    "BDMAP_00394224": "PATIENT_BDMAP_P5",
}

def extract_patient_id(subject_id: str) -> str:
    s = str(subject_id).strip()
    if s in JHU_SCAN_TO_PATIENT_MAP:
        return JHU_SCAN_TO_PATIENT_MAP[s]
    parts = s.split("_")
    if len(parts) >= 2 and parts[0] in ("CV", "CancerVerse", "BDMAP"):
        return f"{parts[0]}_{parts[1]}"
    return s


def find_validation_cases(data_dir, ensemble_out_dir, audit_csv, gold_standard_dir=None, max_cases=20, exclude_subjects=None):
    cases = []
    raw_exclude = set(str(s) for s in (exclude_subjects or []))
    exclude_set = set()
    for s in raw_exclude:
        exclude_set.add(s)
        exclude_set.add(extract_patient_id(s))

    # If gold_standard_dir is provided, look for gold standard masks
    if gold_standard_dir and os.path.exists(gold_standard_dir):
        gs_files = sorted(glob.glob(os.path.join(gold_standard_dir, "*.nii.gz")))
        for gf in gs_files:
            sub = os.path.basename(gf).replace(".nii.gz", "")
            if sub in exclude_set or extract_patient_id(sub) in exclude_set:
                continue
            ct = os.path.join(data_dir, sub, "ct.nii.gz")
            if os.path.exists(ct):
                cases.append({"image": ct, "label": gf, "name": sub})
                if len(cases) >= max_cases:
                    break
        if cases:
            print(f"Found {len(cases)} cases from gold standard directory: {gold_standard_dir}")
            return cases

    if audit_csv and os.path.exists(audit_csv):
        df = pd.read_csv(audit_csv)
        clean = df[df["triage_category"] == "CLEAN_HIGH_CONFIDENCE"]
        for _, row in clean.iterrows():
            sub = str(row["subject_id"])
            if sub in exclude_set or extract_patient_id(sub) in exclude_set:
                continue
            ct = os.path.join(data_dir, sub, "ct.nii.gz")
            lbl = os.path.join(ensemble_out_dir, f"{sub}_consensus.nii.gz")
            if os.path.exists(ct) and os.path.exists(lbl):
                cases.append({"image": ct, "label": lbl, "name": sub})
                if len(cases) >= max_cases:
                    break

    if not cases and os.path.exists(ensemble_out_dir):
        for lbl in sorted(glob.glob(os.path.join(ensemble_out_dir, "*_consensus.nii.gz"))):
            sub = os.path.basename(lbl).replace("_consensus.nii.gz", "")
            if sub in exclude_set or extract_patient_id(sub) in exclude_set:
                continue
            ct = os.path.join(data_dir, sub, "ct.nii.gz")
            if os.path.exists(ct):
                cases.append({"image": ct, "label": lbl, "name": sub})
                if len(cases) >= max_cases:
                    break

    return cases


def evaluate_checkpoint(model, ckpt_path, cases, val_tf, roi_size, device):
    # train_mednext_phase2 saves raw.state_dict() directly (a flat OrderedDict)
    # It is NOT wrapped in a dict with 'model_state_dict' key.
    # We try common wrappers first, then fall back to treating the whole thing as a state dict.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "state_dict" in ckpt):
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    else:
        # Saved with torch.save(raw.state_dict(), path) — already the state dict
        sd = ckpt
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        raise RuntimeError(f"Strict loading failed — missing checkpoint keys: {missing[:5]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys in checkpoint: {unexpected[:3]}")
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

            # Mask out ignore index (255) from both predictions and targets (F15)
            valid_mask = (lbls[:, 0, ...] != 255)

            for c in range(4):
                organ_idx = c + 1
                p = ((preds[:, 0, ...] == organ_idx) & valid_mask).float()
                g = ((lbls[:, 0, ...] == organ_idx) & valid_mask).float()
                p_sum = p.sum().item()
                g_sum = g.sum().item()
                if p_sum > 0 or g_sum > 0:
                    inter = (p * g).sum().item()
                    dice = (2.0 * inter) / (p_sum + g_sum + 1e-6)
                    all_dice[c].append(dice)

    per_organ_dice = {
        ORGAN_NAMES[c]: float(np.mean(all_dice[c])) if len(all_dice[c]) > 0 else np.nan
        for c in range(4)
    }
    valid_organ_dices = [d for d in per_organ_dice.values() if not np.isnan(d)]
    mean_dice = float(np.mean(valid_organ_dices)) if valid_organ_dices else np.nan

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
    parser.add_argument("--gold_standard_dir", type=str, default=None, help="Path to JHU radiologist-corrected labels directory")
    parser.add_argument("--output_csv", type=str, default=None, help="Custom output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_val_cases", type=int, default=20)
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--exclude_manifest", type=str, default=None,
                        help="Optional CSV manifest of training cases to strictly exclude from validation (F06/F13)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 70)
    print("      SCALING SWEEP POST-HOC EVALUATION")
    print(f"      Device:     {device}")
    print(f"      Results:    {args.results_dir}")
    if args.gold_standard_dir:
        print(f"      Gold Std:   {args.gold_standard_dir}")
    print("=" * 70)

    exclude_subs = set()
    if args.exclude_manifest:
        if not os.path.exists(args.exclude_manifest):
            raise FileNotFoundError(f"Specified --exclude_manifest does not exist: {args.exclude_manifest}")
        ex_df = pd.read_csv(args.exclude_manifest)
        col = "subject_id" if "subject_id" in ex_df.columns else ("name" if "name" in ex_df.columns else None)
        if col is None:
            raise ValueError(f"--exclude_manifest '{args.exclude_manifest}' is malformed: missing 'subject_id' or 'name' column.")
        exclude_subs = set(ex_df[col].astype(str))
        print(f"Excluding {len(exclude_subs)} training subjects and their sibling phases from validation holdout.")

    # 1. Find validation cases
    val_cases = find_validation_cases(
        args.data_dir, args.ensemble_out_dir, args.audit_csv,
        gold_standard_dir=args.gold_standard_dir, max_cases=args.num_val_cases,
        exclude_subjects=exclude_subs,
    )
    if not val_cases:
        print(f"ERROR: No valid (CT, label) pairs found in {args.data_dir} / {args.ensemble_out_dir}!")
        sys.exit(1)
    print(f"Loaded {len(val_cases)} validation cases for evaluation.")

    # 2. Get transforms with FixLiarLabelAffined
    if get_transforms is None:
        print("ERROR: get_transforms could not be imported from train_mednext_phase2.")
        sys.exit(1)
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

    # 4. Instantiate Model (model_id="B", kernel_size=3 to match sweep config in submit_scaling_sweep.sh)
    if create_mednext_v1 is None:
        print("ERROR: create_mednext_v1 could not be imported. Ensure MedNeXt is installed in the conda env:")
        print("  pip install git+https://github.com/MIC-DKFZ/MedNeXt.git")
        sys.exit(1)

    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=NUM_CLASSES,  # 5: bg + 4 organs
        model_id="B",             # Must match --model_id used in training
        kernel_size=3,            # Must match --kernel_size used in training
        deep_supervision=False,
    ).to(device)

    summary = []
    for ckpt in ckpts:
        rel = os.path.relpath(ckpt, args.results_dir)
        parts = rel.split(os.sep)
        n_folder = parts[0] if len(parts) > 0 else "unknown"
        seed_folder = parts[1] if len(parts) > 1 else "unknown"

        # Ensure per-checkpoint training membership exclusion
        ckpt_dir = os.path.dirname(ckpt)
        ckpt_manifest = os.path.join(ckpt_dir, "train_manifest.json")
        ckpt_excludes = set(exclude_subs)
        if os.path.exists(ckpt_manifest):
            try:
                with open(ckpt_manifest, "r") as mf:
                    minfo = json.load(mf)
                    for c_name in minfo.get("train_cases", []) + minfo.get("train_scans", []) + minfo.get("train_patients", []):
                        ckpt_excludes.add(str(c_name))
            except Exception as me:
                print(f"Warning: Could not read {ckpt_manifest}: {me}")

        curr_val_cases = [c for c in val_cases if c.get("name", "") not in ckpt_excludes]
        if not curr_val_cases:
            curr_val_cases = find_validation_cases(
                args.data_dir, args.ensemble_out_dir, args.audit_csv,
                gold_standard_dir=args.gold_standard_dir, max_cases=args.num_val_cases,
                exclude_subjects=ckpt_excludes,
            )

        print(f"Evaluating [{n_folder} | {seed_folder}] on {len(curr_val_cases)} cases -> {ckpt} ...")
        res = evaluate_checkpoint(model, ckpt, curr_val_cases, val_tf, tuple(args.roi_size), device)
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
    if args.output_csv:
        out_csv = args.output_csv
    else:
        out_csv = os.path.join(args.results_dir, "scaling_sweep_posthoc_eval.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
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
