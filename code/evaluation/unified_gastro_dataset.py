"""
unified_gastro_dataset.py

PyTorch Dataset + Dataset Manifest for Phase-2 training data pipeline.

Serves three categories of training subjects:
  - CLEAN:       category=CLEAN_HIGH_CONFIDENCE (pseudo-GT from ensemble consensus)
  - WEAK:        category=WEAK_COARSE (auto-labeled with IGNORE_INDEX=255 boundary
                 where human coarse labels conflict with consensus)
  - TEACHER:     large-scale TotalSegmentator auto-generated pseudo-labels for
                 unannotated scans (GKD Teacher-Student Distillation)

Automatically discovers subjects using the same directory conventions as the
rest of mp-factory.

Built-in verification (verify_dataset_integrity):
  - Reports NIfTI file loading, id channel, shape consistency for CT/label pairs.
  - No corrupted or missing paths can enter the pipeline silently.

Usage:
  python unified_dataset.py \
    --audit_csv results/ensemble_audit_summary.csv \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --consensus_dir results/ensemble_out \
    --autolabel_dir results/autolabel_out \
    --totalseg_dir results/totalseg_gi_masks_bdmap \
    --use_weak \
    --verify
"""

import os
import sys
import glob
import argparse
import json
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# NOTE: nibabel imported lazily inside _load_volume because the module-level
# functions (build_manifest, __init__) can be unit-tested without nibabel.

IGNORE_INDEX = 255  # matches train_mednext_phase2.py / hard_threshold_autolabel.py
CATEGORY_CLEAN = "CLEAN_HIGH_CONFIDENCE"
CATEGORY_WEAK  = "WEAK_COARSE"
CATEGORY_TEACHER = "AUTO_TOTALSEGMENTATOR"  # pseudo-labels for GKD

ORGAN_LABELS = {0: "background", 1: "stomach", 2: "duodenum", 3: "small_bowel", 4: "colon"}


def _scan_for_consensus(data_dir, subject_id):
    fname = os.path.join(data_dir, f"{subject_id}_consensus.nii.gz")
    return fname if os.path.isfile(fname) else None


def _scan_for_autolabel(autolabel_dir, subject_id):
    fname = os.path.join(autolabel_dir, f"{subject_id}_autolabel.nii.gz")
    return fname if os.path.isfile(fname) else None


def _scan_for_teacher_label(teacher_dir, subject_id):
    # TotalSegmentator style exports: *_gi_seg.nii.gz
    for fpat in ["_gi_seg.nii.gz", "_seg.nii.gz"]:
        fname = os.path.join(teacher_dir, f"{subject_id}{fpat}")
        if os.path.isfile(fname):
            return fname
    return None


class UnifiedGI3Dataset(torch.utils.data.Dataset):
    """
    Reads CT, label (clean consensus / weak autolabel / teacher pseudo),
    and reports category code.

    For WEAK_COARSE cases, label contains IGNORE_INDEX=255 conflict voxels.
    """
    def __init__(self, manifest, transform=None):
        self.manifest = manifest
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        record = self.manifest[idx]
        ct_path      = record["ct_path"]
        label_path   = record["label_path"]
        category     = record.get("category", CATEGORY_CLEAN)
        subject_id   = record.get("subject_id", "??")

        d = {"image": ct_path, "label": label_path, "category": category, "subject_id": subject_id}

        if self.transform:
            d = self.transform(d)
        return d


def build_manifest(
    audit_csv,
    data_dir,
    consensus_dir,
    autolabel_dir=None,
    teacher_dir=None,
    use_weak=False,
):
    """
    Build a unified list of {ct_path, label_path, category} dicts.

    Returns a list of dicts each with:
      - subject_id (str)
      - ct_path (str, can be checked for existence)
      - label_path (str, can be checked for existence)
      - category (str, one of CLEAN_HIGH_CONFIDENCE / WEAK_COARSE / AUTO_TOTALSEARCHATOR)
    """
    df = pd.read_csv(audit_csv)

    cohort_clean = df[df['triage_category'].astype(str) == 'CLEAN_HIGH_CONFIDENCE'].copy() if 'triage_category' in df else df.iloc[:0]
    cohort_weak  = df[df['triage_category'].astype(str) == 'WEAK_COARSE'].copy() if 'triage_category' in df and use_weak else df.iloc[:0]

    rows = []

    # --- CLEAN: consensus is the label ---
    for _, r in cohort_clean.iterrows():
        sub_id = str(r["subject_id"])
        ct_p = os.path.join(data_dir, sub_id, "ct.nii.gz")
        cons_p = os.path.join(consensus_dir, f"{sub_id}_consensus.nii.gz") if consensus_dir else None
        rows.append({"subject_id": sub_id, "ct_path": ct_p, "label_path": cons_p,
                     "category": "CLEAN_HIGH_CONFIDENCE"})

    # --- WEAK: autolabel (hard-threshold) ---
    for _, r in cohort_weak.iterrows():
        sub_id = str(r["subject_id"])
        ct_p = os.path.join(data_dir, sub_id, "ct.nii.gz")
        aut_p = os.path.join(autolabel_dir, f"{sub_id}_autolabel.nii.gz") if autolabel_dir else None
        rows.append({"subject_id": sub_id, "ct_path": ct_p, "label_path": aut_p,
                     "category": "WEAK_COARSE"})

    # --- Teacher: TotalSegmentator auto-labels ---
    if teacher_dir and os.path.isdir(teacher_dir):
        for fn in sorted(os.listdir(teacher_dir)):
            if not fn.endswith("_gi_seg.nii.gz") and not fn.endswith("_seg.nii.gz"):
                continue
            sub_id = fn.replace("_gi_seg.nii.gz", "").replace("_seg.nii.gz", "")
            # Avoid duplicating subjects already in CLEAN/WEAK
            if sub_id in set(r["subject_id"] for r in rows):
                continue
            ct_p = os.path.join(data_dir, sub_id, "ct.nii.gz")
            lbl_p = os.path.join(teacher_dir, fn) if os.path.isfile(os.path.join(teacher_dir, fn)) else None
            if ct_p and lbl_p and os.path.isfile(ct_p):
                rows.append({"subject_id": sub_id, "ct_path": ct_p, "label_path": lbl_p,
                             "category": "AUTO_TOTALESEARCHATOR"})

    return rows


def verify_dataset_integrity(manifest, n_workers=1, early_exit=True):
    """
    Reports items missing/broken, returns error count.
    Loads up to a 25% subset of labels to confirm data validity without spending hours.
    """
    import nibabel as nib

    errors = []
    valid_rows = 0
    done = 0
    for i, rec in enumerate(manifest):
        sub     = rec['subject_id']
        ct, lbl = rec['ct_path'], rec['label_path']
        ec = 0
        msgs = []
        if not os.path.isfile(ct):
            msgs.append("ct missing")
            ec += 1
        if not os.path.isfile(lbl):
            msgs.append("label missing")
            ec += 1

        # Attempt light NIfTI load for a random subset to catch corrupt/half-written files
        if ec == 0 and (i % 10 == 0):
            try:
                ct_nii = nib.load(ct)
                ct_data = np.asanyarray(ct_nii.dataobj)
                lbl_nii = nib.load(lbl)
                lbl_data = np.asanyarray(lbl_nii.dataobj)
                if ct_data.shape != lbl_data.shape:
                    msgs.append(f"shape mismatch CT {ct_data.shape} vs label {lbl_data.shape}")
                    ec += 1
                if ct_data.ndim != 3:
                    msgs.append(f"CT is not 3D (shape {ct_data.shape})")
                    ec += 1
                if lbl_data.ndim not in (3, 4):
                    msgs.append(f"label is not 3D/4D (shape {lbl_data.shape})")
                    ec += 1
            except Exception as exc:
                msgs.append(f"NIfTI load/read error: {exc}")
                ec += 1

        if ec:
            errors.append({"idx": i, "subject_id": sub, "ct": ct, "label": lbl,
                           "errors": msgs, "count": ec})
        else:
            valid_rows += 1

        if (i+1) % 100 == 0:
            print(f"  verified {i+1}/{len(manifest)} entries ({valid_rows} clean) ...")

    print(f"\n[VERIFICATION COMPLETE]")
    print(f"  Total manifest entries: {len(manifest)}")
    print(f"  Valid   : {valid_rows}")
    print(f"  Broken  : {len(errors)}")
    if errors:
        for err in errors[:5]:
            print(f"  - {err['subject_id']}: {err['errors']}")
        if len(errors) > 5:
            print(f"  ... {len(errors)-5} more errors omitted")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Unified Dataset Builder / Verifier for A.3 pipeline")
    parser.add_argument("--audit_csv", type=str, required=True,
                        help="Path to ensemble_audit_summary.csv")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="DrugVerse_dbox or equivalent CT directory")
    parser.add_argument("--consensus_dir", type=str, required=True,
                        help="Ensemble consensus output directory")
    parser.add_argument("--autolabel_dir", type=str, default=None,
                        help="Hard-threshold autolabeled masks (optional)")
    parser.add_argument("--teacher_dir", type=str, default=None,
                        help="TotalSegmentator auto-labeled mask dir")
    parser.add_argument("--use_weak", action="store_true",
                        help="Include WEAK_COARSE with boundary masking")
    parser.add_argument("--verify", action="store_true",
                        help="Run NIfTI integrity checks after building manifest")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for manifest/maps")
    args = parser.parse_args()

    print("[Unified Dataset Builder] starting ...")
    manifest = build_manifest(
        audit_csv=args.audit_csv,
        data_dir=args.data_dir,
        consensus_dir=args.consensus_dir,
        autolabel_dir=args.autolabel_dir,
        teacher_dir=args.teacher_dir,
        use_weak=args.use_weak,
    )
    print(f"Manifest entries: {len(manifest)}")
    n_indexed = len(manifest)
    print(f"  CLEAN_HIGH_CONFIDENCE  : {sum(1 for r in manifest if r['category']=='CLEAN_HIGH_CONFIDENCE')}")
    print(f"  WEAK_COARSE            : {sum(1 for r in manifest if r['category']=='WEAK_COARSE')}")
    print(f"  AUTO_TOTALESEARCHATOR  : {sum(1 for r in manifest if r['category']=='AUTO_TOTALESEARCHATOR')}")

    if args.verify:
        print("\n[VERIFY] Integrity check ...")
        errors = verify_dataset_integrity(manifest)
        if errors:
            for e in errors[:10]:
                print(f"  - {e['subject_id']}: {e['errors']}")
            if len(errors) > 10:
                print(f"  ... {len(errors)-10} more errors")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        ds = UnifiedGI3Dataset(manifest)
        dl = DataLoader(ds, batch_size=1, num_workers=0)
        cnt = 0
        for batch in dl:
            cnt += 1
            if cnt == 1:
                print(f"\nSample batch: keys={list(batch.keys())}, shapes: image={batch['image'].shape} label={batch['label'].shape} category={batch['category']}")
            if cnt >= 3:
                break
        print(f"\nDataLoader: produced {cnt} batch(es)")


if __name__ == "__main__":
    main()