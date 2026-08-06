"""
hard_threshold_autolabel.py

Hard-Threshold Auto-Labeling: compares human coarse annotations with ensemble
consensus pseudo-ground truth, and assigns an ignore class (255) to conflicting
voxels (where human label != consensus) so downstream training (e.g.
train_mednext_phase2.py) can skip those pixels via ignore_index=255.

Process:
  1. Read WEAK_COARSE subjects from the ensemble audit CSV (or a split txt file).
  2. For each subject:
       a. Load the human coarse label (CancerVerse_dbox/<subid>/gi_mask.nii.gz
          or folder segmentations/).
       b. Load the ensemble consensus pseudo-GT from ensemble_out/<subid>_consensus.nii.gz.
       c. Resample the consensus to the human label's voxel grid using nearest-neighbor.
       d. Identify agreeing foreground voxels (human label > 0, consensus agrees).
       e. For conflicting voxels (human label > 0, consensus disagrees):
          set to IGNORE_INDEX = 255.
       f. Save the autolabel result as <subject_id>_autolabel.nii.gz in out_dir.
  3. Write augmented CSV recording the percentage of output voxels marked as 'ignore'.

Supports a --splits_file txt alternative to reading the audit CSV directly.
For scans lacking a human coarse label, the consensus mask is used as-is.

Organ label convention (same across all mp-factory pipelines):
    0 = Background, 1 = Stomach, 2 = Duodenum, 3 = Small Bowel, 4 = Colon.
IGNORE_INDEX = 255 matches the IGNORE_INDEX in train_mednext_phase2.py.

Note on ignore_index: The task spec mentions -1, but the training code
(train_mednext_phase2.py DiceCEWithIgnoreLoss) and the NIfTI uint8 label
storage both require 255. PyTorch's F.cross_entropy(ignore_index=255) skips
those voxels identically to ignore_index=-1, but 255 fits in uint8 while -1
does not. We use 255 everywhere for end-to-end consistency.

Usage:
  python hard_threshold_autolabel.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_v2 \
    --consensus_dir /mnt/scratch/user/chrsong/mp-factory/results/ensemble_out \
    --audit_csv /mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv \
    --out_dir /mnt/scratch/user/chrsong/mp-factory/results/autolabel_out \
    --num_workers 16
"""

import os
import sys
import glob
import argparse
import gc
import numpy as np
import pandas as pd

# NOTE: nibabel (and optionally nilearn) imported lazily inside
# autolabel_subject so this module is importable (for unit tests) even where
# those deps are absent.

# Matches IGNORE_INDEX in train_mednext_phase2.py (DiceCEWithIgnoreLoss).
IGNORE_INDEX = 255

ORGAN_MAP = {
    1: 'stomach',
    2: 'duodenum',
    3: 'small_bowel',
    4: 'colon',
}


def resolve_human_label(data_dir, subject_id):
    """Return the primary human label file for a subject, or None.

    Checks, in order:
      1. ``{data_dir}/{subject_dir}/gi_mask.nii.gz``  (single multi-organ mask)
      2. ``{data_dir}/{subject_dir}/segmentations/``   (individual organ files)
    """
    combined = os.path.join(data_dir, subject_id, "gi_mask.nii.gz")
    if os.path.isfile(combined):
        return ("combined", combined)

    seg_dir = os.path.join(data_dir, subject_id, "segmentations")
    if os.path.isdir(seg_dir) and glob.glob(os.path.join(seg_dir, "*.nii.gz")):
        return ("segmentations", seg_dir)

    return (None, None)


def autolabel_subject(subject_id, data_dir, consensus_dir, out_dir):
    """Process a single WEAK_COARSE subject -> *_autolabel.nii.gz + stats dict."""
    import nibabel as nib

    result = {'subject_id': subject_id, 'status': 'PROCESSING'}
    try:
        label_kind, label_path = resolve_human_label(data_dir, subject_id)
        if label_kind is None:
            result['status'] = 'SKIP: no human coarse label found'
            return result

        consensus_path = os.path.join(consensus_dir, f"{subject_id}_consensus.nii.gz")
        if not os.path.isfile(consensus_path):
            result['status'] = 'SKIP: no consensus mask'
            return result

        # ---- load human label(s) ----
        if label_kind == "combined":
            human_nii = nib.load(label_path)
            human_arr = np.asanyarray(human_nii.dataobj).astype(np.uint8)
        elif label_kind == "segmentations":
            # Merge individual organ NIfTIs, guided by ct.nii.gz for shape/affine.
            ct_path = os.path.join(data_dir, subject_id, "ct.nii.gz")
            ct_nii = nib.load(ct_path)
            ct_arr = np.asanyarray(ct_nii.dataobj).squeeze()
            human_arr = np.zeros_like(ct_arr, dtype=np.uint8)
            # Affine-aware resampler for cases where seg files are not in CT voxel space.
            try:
                from nilearn.image import resample_to_img
                _have_nilearn = True
            except ImportError:
                _have_nilearn = False

            seg_dir = label_path
            matched_ids = set()
            for fname in sorted(os.listdir(seg_dir)):
                if not fname.endswith('.nii.gz'):
                    continue
                base = fname.replace('.nii.gz', '').lower()
                seg_nii = nib.load(os.path.join(seg_dir, fname))
                seg_arr = np.asanyarray(seg_nii.dataobj).squeeze().astype(np.uint8)
                # Resample to CT voxel grid if shapes differ.
                if seg_arr.shape != human_arr.shape and _have_nilearn:
                    ref_img = nib.Nifti1Image(human_arr.astype(np.float32),
                                              ct_nii.affine, ct_nii.header)
                    src_img = nib.Nifti1Image(seg_arr.astype(np.float32),
                                              seg_nii.affine, seg_nii.header)
                    seg_arr = np.asanyarray(
                        resample_to_img(src_img, ref_img, interpolation='nearest',
                                        copy=False).dataobj).astype(np.uint8)
                # Guess organ id from filename.
                for oid, oname in ORGAN_MAP.items():
                    if oname in base and oid not in matched_ids:
                        human_arr[seg_arr > 0] = oid
                        matched_ids.add(oid)
                        break
            human_nii = nib.Nifti1Image(human_arr, ct_nii.affine, ct_nii.header)
        else:
            result['status'] = 'SKIP: unknown label kind'
            return result

        # Clamp to 0..4 range.
        human_arr = np.where((human_arr >= 1) & (human_arr <= 4), human_arr, 0).astype(np.uint8)
        human_nii = nib.Nifti1Image(human_arr, human_nii.affine, human_nii.header)

        # ---- load consensus, resample to human grid ----
        cons_nii = nib.load(consensus_path)
        cons_arr = np.asanyarray(cons_nii.dataobj).astype(np.uint8)

        if cons_arr.shape != human_arr.shape:
            try:
                from nilearn.image import resample_to_img as _r
            except ImportError:
                result['status'] = 'SKIP: nilearn not installed for resampling'
                return result
            ref_img = human_nii
            src_img = nib.Nifti1Image(cons_arr.astype(np.float32), cons_nii.affine, cons_nii.header)
            cons_arr = np.asanyarray(
                _r(src_img, ref_img, interpolation='nearest', copy=False).dataobj
            ).astype(np.uint8)

        cons_arr = np.where((cons_arr >= 1) & (cons_arr <= 4), cons_arr, 0).astype(np.uint8)

        # ---- hard threshold: conflicting voxels -> IGNORE_INDEX ----
        autolabel = human_arr.copy()
        fg_mask = human_arr > 0
        n_fg = int(fg_mask.sum())

        # conflicting foreground voxels: human has an organ label, consensus does NOT match
        conflict = fg_mask & (human_arr != cons_arr)
        n_ignore = int(conflict.sum())
        pct_ignore = round((n_ignore / n_fg * 100.0) if n_fg > 0 else 0.0, 2)

        autolabel[conflict] = IGNORE_INDEX

        out_path = os.path.join(out_dir, f"{subject_id}_autolabel.nii.gz")
        out_nii = nib.Nifti1Image(autolabel, human_nii.affine, human_nii.header)
        nib.save(out_nii, out_path)

        result.update({
            'status': 'SUCCESS',
            'n_foreground_voxels': n_fg,
            'n_ignore_voxels': n_ignore,
            'pct_ignore': pct_ignore,
            'output_path': out_path,
        })

    except Exception as exc:
        result['status'] = f'ERROR: {str(exc)}'
    finally:
        gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Hard-Threshold Autolabel: mark conflicting human/consensus voxels as IGNORE_INDEX=255")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="CancerVerse_dbox / human coarse label root")
    parser.add_argument("--consensus_dir", type=str, required=True,
                        help="Ensemble consensus mask directory (ensemble_out)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Autolabel NIfTI output directory")
    parser.add_argument("--audit_csv", type=str, default=None,
                        help="Audit CSV (ensemble_audit_summary.csv) — reads WEAK_COARSE subjects")
    parser.add_argument("--splits_file", type=str, default=None,
                        help="Plain text file with one subject_id per line")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Output stats CSV path (defaults to <out_dir>/autolabel_summary.csv)")
    args = parser.parse_args()

    # Resolve subject list
    if args.splits_file and os.path.isfile(args.splits_file):
        with open(args.splits_file) as fh:
            subjects = [ln.strip() for ln in fh if ln.strip()]
        print(f"Loaded {len(subjects)} subjects from {args.splits_file}")
    elif args.audit_csv and os.path.isfile(args.audit_csv):
        df = pd.read_csv(args.audit_csv)
        df_weak = df[df['triage_category'] == 'WEAK_COARSE']
        subjects = list(df_weak['subject_id'].astype(str))
        print(f"Loaded {len(subjects)} WEAK_COARSE subjects from {args.audit_csv}")
    else:
        print("ERROR: Provide --audit_csv or --splits_file.")
        sys.exit(1)

    if not subjects:
        print("No subjects to process.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = args.out_csv or os.path.join(args.out_dir, "autolabel_summary.csv")

    print(f"Processing {len(subjects)} subjects ...")
    results = []
    for completed, sub_id in enumerate(subjects, 1):
        res = autolabel_subject(sub_id, args.data_dir, args.consensus_dir, args.out_dir)
        results.append(res)
        if completed % 100 == 0 or completed == len(subjects):
            print(f"  [{completed}/{len(subjects)}] ({completed/len(subjects)*100:.1f}%)", flush=True)

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Stats CSV saved: {out_csv}")

    ok = df_out['status'] == 'SUCCESS'
    n_success = ok.sum()
    print("\n[AUTOLABEL SUMMARY]")
    print(f"Total subjects        : {len(subjects)}")
    print(f"  Success             : {n_success}")
    print(f"  Failed / Skipped    : {len(subjects) - n_success}")
    if n_success > 0:
        print(f"  Mean pct ignore     : {df_out.loc[ok, 'pct_ignore'].mean():.2f} %")
    print(f"Autolabel output dir  : {args.out_dir}")


if __name__ == "__main__":
    main()