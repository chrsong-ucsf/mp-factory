"""
compare_models.py

Compares MedNeXt vs Swin-UNETR GI segmentation predictions against ground truth
organ labels. Computes per-organ and mean Dice scores for both models on a held-out
subset of subjects that have:
  - Ground truth in BDMAP_*/segmentations/ (stomach, duodenum, small_bowel/intestine, colon)
  - Swin-UNETR prediction: results/swin_unetr_predictions/{subject}_gi_seg.nii.gz
  - MedNeXt prediction: results/mednext_predictions/{subject}_gi_seg.nii.gz

Usage:
  python compare_models.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --swin_pred_dir /mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions \
    --mednext_pred_dir /mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions \
    --out_csv /mnt/scratch/user/chrsong/mp-factory/results/model_comparison.csv \
    --max_subjects 200
"""

import os
import sys
import glob
import argparse
import csv
import numpy as np
import nibabel as nib

ORGAN_IDS   = [1, 2, 3, 4]
ORGAN_NAMES = {1: 'stomach', 2: 'duodenum', 3: 'small_bowel', 4: 'colon'}
ORGAN_ALIASES = {
    1: ['stomach'],
    2: ['duodenum'],
    3: ['small_bowel', 'intestine', 'small_intestine'],
    4: ['colon'],
}


def dice_score(pred, gt, organ_id):
    """Compute binary Dice score for a single organ class."""
    p = (pred == organ_id).astype(np.uint8)
    g = (gt   == organ_id).astype(np.uint8)
    intersection = (p * g).sum()
    denom = p.sum() + g.sum()
    if denom == 0:
        return float('nan')   # organ absent in both pred and GT → skip
    return 2.0 * intersection / denom


def load_ground_truth(subject_dir):
    """Merge individual organ segmentation NIfTIs into a single label map."""
    seg_dir = os.path.join(subject_dir, "segmentations")
    search_dir = seg_dir if os.path.exists(seg_dir) else subject_dir

    # Check for combined mask first
    combined_mask = os.path.join(subject_dir, "gi_mask.nii.gz")
    if os.path.exists(combined_mask):
        return nib.load(combined_mask).get_fdata().astype(np.uint8)

    ct_path = os.path.join(subject_dir, "ct.nii.gz")
    ct_nii  = nib.load(ct_path)
    gt_arr  = np.zeros(ct_nii.shape, dtype=np.uint8)

    for organ_id, aliases in ORGAN_ALIASES.items():
        for alias in aliases:
            fpath = os.path.join(search_dir, f"{alias}.nii.gz")
            if os.path.exists(fpath):
                o_arr = nib.load(fpath).get_fdata() > 0
                gt_arr[o_arr] = organ_id
                break

    return gt_arr


def main():
    parser = argparse.ArgumentParser(description="Compare MedNeXt vs Swin-UNETR GI segmentation quality")
    parser.add_argument("--data_dir",       type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument("--swin_pred_dir",  type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions")
    parser.add_argument("--mednext_pred_dir", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions")
    parser.add_argument("--out_csv", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/model_comparison.csv")
    parser.add_argument("--max_subjects", type=int, default=200,
                        help="Max number of subjects to evaluate (default: 200)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Collect subjects that have BOTH prediction files
    subject_dirs = sorted([
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ])

    evaluated = []
    skipped   = 0

    print(f"Scanning {len(subject_dirs)} subjects for evaluation...", flush=True)

    for sub_dir in subject_dirs:
        if len(evaluated) >= args.max_subjects:
            break

        subject_id = os.path.basename(sub_dir)
        ct_path    = os.path.join(sub_dir, "ct.nii.gz")
        swin_path  = os.path.join(args.swin_pred_dir,    f"{subject_id}_gi_seg.nii.gz")
        mednext_path = os.path.join(args.mednext_pred_dir, f"{subject_id}_gi_seg.nii.gz")

        if not all(os.path.exists(p) for p in [ct_path, swin_path, mednext_path]):
            skipped += 1
            continue

        # Check ground truth has at least one organ label
        seg_dir    = os.path.join(sub_dir, "segmentations")
        search_dir = seg_dir if os.path.exists(seg_dir) else sub_dir
        has_gt = any(
            any(os.path.exists(os.path.join(search_dir, f"{alias}.nii.gz")) for alias in aliases)
            for aliases in ORGAN_ALIASES.values()
        )
        if not has_gt:
            skipped += 1
            continue

        evaluated.append(sub_dir)

    print(f"  Evaluating {len(evaluated)} subjects | Skipped (missing files): {skipped}", flush=True)

    # CSV header
    fieldnames = ['subject_id']
    for name in ORGAN_NAMES.values():
        fieldnames += [f'swin_{name}_dice', f'mednext_{name}_dice']
    fieldnames += ['swin_mean_dice', 'mednext_mean_dice', 'winner']

    swin_organ_scores    = {oid: [] for oid in ORGAN_IDS}
    mednext_organ_scores = {oid: [] for oid in ORGAN_IDS}
    swin_wins = mednext_wins = ties = 0

    with open(args.out_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, sub_dir in enumerate(evaluated, 1):
            subject_id = os.path.basename(sub_dir)
            print(f"  [{idx}/{len(evaluated)}] {subject_id}", flush=True)

            try:
                gt_arr      = load_ground_truth(sub_dir)
                swin_arr    = nib.load(os.path.join(args.swin_pred_dir,    f"{subject_id}_gi_seg.nii.gz")).get_fdata().astype(np.uint8)
                mednext_arr = nib.load(os.path.join(args.mednext_pred_dir, f"{subject_id}_gi_seg.nii.gz")).get_fdata().astype(np.uint8)

                # Squeeze channel dim if present
                if swin_arr.ndim == 4:    swin_arr    = swin_arr[0]
                if mednext_arr.ndim == 4: mednext_arr = mednext_arr[0]

                row = {'subject_id': subject_id}
                swin_dices    = []
                mednext_dices = []

                for oid in ORGAN_IDS:
                    name = ORGAN_NAMES[oid]
                    sd   = dice_score(swin_arr,    gt_arr, oid)
                    md   = dice_score(mednext_arr, gt_arr, oid)
                    row[f'swin_{name}_dice']    = f"{sd:.4f}" if not np.isnan(sd) else 'N/A'
                    row[f'mednext_{name}_dice'] = f"{md:.4f}" if not np.isnan(md) else 'N/A'

                    if not np.isnan(sd):
                        swin_dices.append(sd)
                        swin_organ_scores[oid].append(sd)
                    if not np.isnan(md):
                        mednext_dices.append(md)
                        mednext_organ_scores[oid].append(md)

                swin_mean    = np.mean(swin_dices)    if swin_dices    else float('nan')
                mednext_mean = np.mean(mednext_dices) if mednext_dices else float('nan')

                row['swin_mean_dice']    = f"{swin_mean:.4f}"    if not np.isnan(swin_mean)    else 'N/A'
                row['mednext_mean_dice'] = f"{mednext_mean:.4f}" if not np.isnan(mednext_mean) else 'N/A'

                if np.isnan(swin_mean) and np.isnan(mednext_mean):
                    row['winner'] = 'N/A'
                elif np.isnan(swin_mean):
                    row['winner'] = 'MedNeXt'; mednext_wins += 1
                elif np.isnan(mednext_mean):
                    row['winner'] = 'Swin'; swin_wins += 1
                elif mednext_mean > swin_mean + 0.005:
                    row['winner'] = 'MedNeXt'; mednext_wins += 1
                elif swin_mean > mednext_mean + 0.005:
                    row['winner'] = 'Swin'; swin_wins += 1
                else:
                    row['winner'] = 'Tie'; ties += 1

                writer.writerow(row)

            except Exception as e:
                print(f"    [ERROR] {subject_id}: {e}", flush=True)

    # Final summary
    print("\n" + "="*60, flush=True)
    print("          MODEL COMPARISON SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"  Subjects evaluated: {len(evaluated)}", flush=True)
    print(f"\n  Per-organ Mean Dice across all subjects:", flush=True)
    print(f"  {'Organ':<20} {'Swin-UNETR':>12} {'MedNeXt':>12}", flush=True)
    print(f"  {'-'*44}", flush=True)
    for oid in ORGAN_IDS:
        name = ORGAN_NAMES[oid]
        s_mean = np.mean(swin_organ_scores[oid])    if swin_organ_scores[oid]    else float('nan')
        m_mean = np.mean(mednext_organ_scores[oid]) if mednext_organ_scores[oid] else float('nan')
        winner_tag = " ✓" if not np.isnan(m_mean) and not np.isnan(s_mean) and m_mean > s_mean else ""
        print(f"  {name:<20} {s_mean:>12.4f} {m_mean:>12.4f}{winner_tag}", flush=True)

    all_swin    = [d for oid in ORGAN_IDS for d in swin_organ_scores[oid]]
    all_mednext = [d for oid in ORGAN_IDS for d in mednext_organ_scores[oid]]
    print(f"\n  Overall Mean Dice:   Swin-UNETR={np.mean(all_swin):.4f}  MedNeXt={np.mean(all_mednext):.4f}", flush=True)
    print(f"\n  Per-subject wins:    Swin={swin_wins}  MedNeXt={mednext_wins}  Ties={ties}", flush=True)
    overall_winner = "MedNeXt" if np.mean(all_mednext) > np.mean(all_swin) else "Swin-UNETR"
    print(f"\n  *** RECOMMENDED MODEL FOR INFERENCE: {overall_winner} ***", flush=True)
    print(f"\n  Results saved to: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
