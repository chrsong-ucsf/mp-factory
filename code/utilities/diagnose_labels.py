#!/usr/bin/env python3
"""
diagnose_labels.py

Fast diagnostic: load 5 consensus/autolabel masks and print their
unique label values, foreground voxel counts, affines, and shapes.
Also loads the paired CT and shows shape before/after ResampleToMatchd.

Run on cluster login node (no GPU needed):
    python code/utilities/diagnose_labels.py

This will tell us definitively:
  1. Do the labels contain foreground voxels (1-4)?
  2. Are the label affines different from CT affines?
  3. Does ResampleToMatchd actually fix the affine mismatch?
  4. After resampling, how many foreground voxels survive?
"""

import os, sys, glob
import numpy as np

SCRATCH = "/mnt/scratch/user/chrsong/mp-factory"
AUDIT_CSV        = f"{SCRATCH}/results/ensemble_audit_summary.csv"
ENSEMBLE_OUT_DIR = f"{SCRATCH}/results/ensemble_out"
AUTOLABEL_DIR    = f"{SCRATCH}/results/autolabel_out"
DATA_DIR         = f"{SCRATCH}/CancerVerse_dbox"
N_INSPECT        = 5

print("=" * 70)
print("  Label Diagnostic")
print("=" * 70)

# 1. Find subjects
try:
    import pandas as pd
    df = pd.read_csv(AUDIT_CSV)
    df = df[df["status"] == "SUCCESS"]
    clean = df[df["triage_category"] == "CLEAN_HIGH_CONFIDENCE"]["subject_id"].astype(str).tolist()
    weak  = df[df["triage_category"] == "WEAK_COARSE"]["subject_id"].astype(str).tolist()
    print(f"Audit CSV: {len(clean)} CLEAN, {len(weak)} WEAK subjects")
except Exception as e:
    print(f"ERROR reading audit CSV: {e}")
    sys.exit(1)

def inspect_label(ct_path, label_path, label_kind):
    try:
        import nibabel as nib
        ct_nii  = nib.load(ct_path)
        lbl_nii = nib.load(label_path)

        ct_arr  = np.asanyarray(ct_nii.dataobj).squeeze()
        lbl_arr = np.asanyarray(lbl_nii.dataobj).squeeze()

        unique, counts = np.unique(lbl_arr, return_counts=True)
        fg_voxels = int(np.sum(lbl_arr[(lbl_arr >= 1) & (lbl_arr <= 4)].size > 0) > 0)
        fg_count  = int(np.sum((lbl_arr >= 1) & (lbl_arr <= 4)))
        ignore_count = int(np.sum(lbl_arr == 255))

        ct_shape  = ct_arr.shape
        lbl_shape = lbl_arr.shape
        same_affine = np.allclose(ct_nii.affine, lbl_nii.affine, atol=1e-3)

        print(f"\n  [{label_kind}] {os.path.basename(label_path)}")
        print(f"    CT  shape: {ct_shape}   affine[:3,3]: {ct_nii.affine[:3,3].round(1)}")
        print(f"    Lbl shape: {lbl_shape}  affine[:3,3]: {lbl_nii.affine[:3,3].round(1)}")
        print(f"    Same affine: {same_affine}")
        print(f"    Unique values: {dict(zip(unique.tolist(), counts.tolist()))}")
        print(f"    Foreground voxels (1-4): {fg_count:,}  |  Ignore voxels (255): {ignore_count:,}")

        if not same_affine:
            # Test ResampleToMatchd manually via nibabel
            try:
                from nilearn.image import resample_to_img
                ref_img = nib.Nifti1Image(np.zeros(ct_arr.shape, dtype=np.float32), ct_nii.affine, ct_nii.header)
                src_img = nib.Nifti1Image(lbl_arr.astype(np.float32), lbl_nii.affine, lbl_nii.header)
                resampled = np.asanyarray(resample_to_img(src_img, ref_img, interpolation='nearest', copy=False).dataobj).squeeze()
                fg_after = int(np.sum((resampled >= 1) & (resampled <= 4)))
                print(f"    >> After resample_to_img: shape={resampled.shape}, fg_voxels={fg_after:,}")
                if fg_after == 0:
                    print(f"    !! WARNING: Foreground DISAPPEARS after resampling to CT grid!")
                    print(f"    !! This is the root cause of Dice=0.")
            except ImportError:
                print("    (nilearn not available — skipping resample test)")
            except Exception as e:
                print(f"    (resample test failed: {e})")
    except Exception as e:
        print(f"  ERROR: {e}")

# 2. Inspect CLEAN subjects (consensus labels)
print(f"\n--- CLEAN_HIGH_CONFIDENCE (consensus masks) ---")
inspected = 0
for sub_id in clean[:N_INSPECT]:
    ct_path  = os.path.join(DATA_DIR, sub_id, "ct.nii.gz")
    lbl_path = os.path.join(ENSEMBLE_OUT_DIR, f"{sub_id}_consensus.nii.gz")
    if os.path.exists(ct_path) and os.path.exists(lbl_path):
        inspect_label(ct_path, lbl_path, "CLEAN")
        inspected += 1
if inspected == 0:
    print("  No CLEAN subjects with both CT and consensus found!")

# 3. Inspect WEAK subjects (autolabels)
print(f"\n--- WEAK_COARSE (autolabels) ---")
inspected = 0
for sub_id in weak[:N_INSPECT]:
    ct_path  = os.path.join(DATA_DIR, sub_id, "ct.nii.gz")
    lbl_path = os.path.join(AUTOLABEL_DIR, f"{sub_id}_autolabel.nii.gz")
    if not os.path.exists(lbl_path):
        lbl_path = os.path.join(ENSEMBLE_OUT_DIR, f"{sub_id}_consensus.nii.gz")
        kind = "WEAK-fallback-consensus"
    else:
        kind = "WEAK-autolabel"
    if os.path.exists(ct_path) and os.path.exists(lbl_path):
        inspect_label(ct_path, lbl_path, kind)
        inspected += 1
if inspected == 0:
    print("  No WEAK subjects with both CT and autolabel found!")

# 4. Sanity check: load one through MONAI transforms and print tensor stats
print(f"\n--- MONAI Transform Sanity Check (first CLEAN case) ---")
try:
    sys.path.insert(0, os.path.join(SCRATCH, "code", "training"))
    from train_mednext_phase2 import get_transforms
    from monai.data import Dataset, DataLoader

    sub_id = clean[0]
    ct_path  = os.path.join(DATA_DIR, sub_id, "ct.nii.gz")
    lbl_path = os.path.join(ENSEMBLE_OUT_DIR, f"{sub_id}_consensus.nii.gz")
    if os.path.exists(ct_path) and os.path.exists(lbl_path):
        _, val_tf = get_transforms()
        ds = Dataset(data=[{"image": ct_path, "label": lbl_path}], transform=val_tf)
        item = ds[0]
        img_t = item["image"]
        lbl_t = item["label"]
        print(f"  After val_transforms:")
        print(f"    image shape: {tuple(img_t.shape)}, range: [{float(img_t.min()):.3f}, {float(img_t.max()):.3f}]")
        print(f"    label shape: {tuple(lbl_t.shape)}")
        import torch
        lbl_np = lbl_t.numpy().squeeze()
        unique, counts = np.unique(lbl_np, return_counts=True)
        print(f"    label unique values: {dict(zip(unique.astype(int).tolist(), counts.tolist()))}")
        fg = int(np.sum((lbl_np >= 1) & (lbl_np <= 4)))
        print(f"    label foreground voxels (1-4): {fg:,}")
        if fg == 0:
            print("    !! CRITICAL: No foreground in transformed label -> training on all-background -> Dice=0")
    else:
        print("  Files not found for MONAI check.")
except Exception as e:
    print(f"  MONAI check failed: {e}")
    import traceback; traceback.print_exc()

print("\n" + "=" * 70)
print("  Done. Paste the output above and we'll know exactly what's wrong.")
print("=" * 70)
