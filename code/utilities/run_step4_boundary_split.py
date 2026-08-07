#!/usr/bin/env python3
"""
run_step4_boundary_split.py

Step 4: Centerline-skeleton 5-organ small-bowel boundary splitting (stomach, duodenum, jejunum, ileum, colon).

Applies 3D centerline-skeleton graph splitting (Treitz 40% / ileocecal 60% ratio)
to split 4-class small_bowel masks into distinct jejunum and ileum classes.

Outputs 5-organ NIfTI masks:
  1: stomach, 2: duodenum, 3: jejunum, 4: ileum, 5: colon

Usage:
  python code/utilities/run_step4_boundary_split.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --out_suffix gi_mask_5organ.nii.gz
"""

import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib

# Attempt vault-root path resolution for splitting module
_here = os.path.abspath(__file__)
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
_vault_root = os.path.dirname(os.path.dirname(_repo_root))
for path in [_repo_root, _vault_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.segmentation.utils.splitting import split_small_intestine

def process_subject(mask_path, out_path, ratio=0.40):
    """Loads 4-class gi_mask.nii.gz and splits small_bowel into jejunum (3) and ileum (4)."""
    nii = nib.load(mask_path)
    arr = np.asanyarray(nii.dataobj).astype(np.uint8)

    stomach  = (arr == 1)
    duodenum = (arr == 2)
    small_bowel = (arr == 3)
    colon    = (arr == 4)

    if small_bowel.sum() == 0:
        # If no small bowel, just shift colon to class 5
        out_arr = np.zeros_like(arr, dtype=np.uint8)
        out_arr[stomach]  = 1
        out_arr[duodenum] = 2
        out_arr[colon]    = 5
    else:
        j_mask, i_mask = split_small_intestine(
            intestine_mask=small_bowel.astype(np.uint8),
            duodenum_mask=duodenum.astype(np.uint8),
            colon_mask=colon.astype(np.uint8),
            ratio=ratio,
        )
        out_arr = np.zeros_like(arr, dtype=np.uint8)
        out_arr[stomach]  = 1
        out_arr[duodenum] = 2
        out_arr[j_mask > 0] = 3
        out_arr[i_mask > 0] = 4
        out_arr[colon]    = 5

    out_nii = nib.Nifti1Image(out_arr, nii.affine, nii.header)
    nib.save(out_nii, out_path)
    print(f"Processed 5-organ split -> {out_path}")

import concurrent.futures

def _worker_process(args_tuple):
    m_path, out_path, ratio = args_tuple
    try:
        process_subject(m_path, out_path, ratio=ratio)
        return True
    except Exception as e:
        print(f"Error processing {m_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Step 4 5-organ small-bowel boundary splitting.")
    parser.add_argument("--data_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument("--out_suffix", type=str, default="gi_mask_5organ.nii.gz")
    parser.add_argument("--ratio", type=float, default=0.40)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    mask_files = sorted(glob.glob(os.path.join(args.data_dir, "*", "gi_mask.nii.gz")))
    if not mask_files:
        mask_files = sorted(glob.glob(os.path.join(args.data_dir, "BDMAP_*", "gi_mask.nii.gz")))

    print(f"Found {len(mask_files)} mask files to process for 5-organ splitting across {args.num_workers} parallel CPU workers.")
    tasks = [(m_path, os.path.join(os.path.dirname(m_path), args.out_suffix), args.ratio) for m_path in mask_files]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(executor.map(_worker_process, tasks))

    print(f"Completed 5-organ boundary splitting: {sum(results)}/{len(tasks)} succeeded.")

if __name__ == "__main__":
    main()

