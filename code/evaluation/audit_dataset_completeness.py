"""
code/evaluation/audit_dataset_completeness.py
=============================================
Audit dataset annotation completeness across CancerVerse, BDMAP, and CV cohorts.
Addresses Audit Finding 3:
Quantifies missing-organ proportions per case (Stomach, Duodenum, Small Bowel, Colon)
to separate true background from unannotated organs and prevent false-negative training penalties.

Outputs:
  results/dataset_annotation_completeness_audit.csv
  results/dataset_annotation_completeness_summary.json
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

ORGAN_MAP = {
    1: 'stomach',
    2: 'duodenum',
    3: 'small_bowel',
    4: 'colon'
}

ORGAN_ALIASES = {
    1: ['stomach'],
    2: ['duodenum'],
    3: ['small_bowel', 'intestine', 'small_intestine'],
    4: ['colon']
}


def audit_case(case_dir: str) -> dict:
    subject_id = os.path.basename(case_dir)
    ct_file = os.path.join(case_dir, "ct.nii.gz")
    if not os.path.exists(ct_file):
        return None

    record = {
        'subject_id': subject_id,
        'case_dir': case_dir,
        'has_ct': True,
        'stomach_present': False,
        'duodenum_present': False,
        'small_bowel_present': False,
        'colon_present': False,
        'rectum_present': False,
        'num_organs_annotated': 0,
        'annotation_type': 'none',
        'is_complete_gi': False,
    }

    # Check for combined mask
    for comb_name in ["gi_mask.nii.gz", f"{subject_id}_gi_mask.nii.gz", "combined_mask.nii.gz"]:
        comb_path = os.path.join(case_dir, comb_name)
        if os.path.exists(comb_path):
            try:
                arr = np.asanyarray(nib.load(comb_path).dataobj)
                uniques = set(np.unique(arr))
                record['stomach_present'] = 1 in uniques or 2 in uniques  # depending on schema
                record['duodenum_present'] = 2 in uniques or 3 in uniques
                record['small_bowel_present'] = 3 in uniques or 4 in uniques
                record['colon_present'] = 4 in uniques or 5 in uniques
                record['annotation_type'] = 'combined_mask'
                record['num_organs_annotated'] = sum([
                    record['stomach_present'], record['duodenum_present'],
                    record['small_bowel_present'], record['colon_present']
                ])
                record['is_complete_gi'] = (record['num_organs_annotated'] == 4)
                return record
            except Exception:
                pass

    # Check segmentations directory or root
    seg_dir = os.path.join(case_dir, "segmentations")
    search_dir = seg_dir if os.path.exists(seg_dir) else case_dir

    for org_id, aliases in ORGAN_ALIASES.items():
        found = False
        for alias in aliases:
            p = os.path.join(search_dir, f"{alias}.nii.gz")
            if os.path.exists(p):
                found = True
                break
        if org_id == 1:
            record['stomach_present'] = found
        elif org_id == 2:
            record['duodenum_present'] = found
        elif org_id == 3:
            record['small_bowel_present'] = found
        elif org_id == 4:
            record['colon_present'] = found

    # Check rectum
    record['rectum_present'] = os.path.exists(os.path.join(search_dir, "rectum.nii.gz"))

    record['num_organs_annotated'] = sum([
        record['stomach_present'], record['duodenum_present'],
        record['small_bowel_present'], record['colon_present']
    ])
    record['annotation_type'] = 'individual_masks' if record['num_organs_annotated'] > 0 else 'none'
    record['is_complete_gi'] = (record['num_organs_annotated'] == 4)

    return record


def main():
    parser = argparse.ArgumentParser(description="Audit GI Annotation Completeness")
    parser.add_argument("--data_dirs", nargs="+", default=[
        "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox",
        "/mnt/scratch/user/chrsong/CancerVerse_data",
        "/mnt/scratch/user/chrsong/mp-factory/CancerVerse/CancerVerse"
    ])
    parser.add_argument("--out_csv", default="results/dataset_annotation_completeness_audit.csv")
    parser.add_argument("--out_json", default="results/dataset_annotation_completeness_summary.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    records = []

    for d in args.data_dirs:
        if not os.path.exists(d):
            print(f"[SKIP] Directory does not exist: {d}")
            continue
        print(f"Scanning directory: {d}...")
        subdirs = sorted([os.path.join(d, sub) for sub in os.listdir(d) if os.path.isdir(os.path.join(d, sub))])
        for s in subdirs:
            rec = audit_case(s)
            if rec:
                rec['parent_dir'] = os.path.basename(d)
                records.append(rec)

    if not records:
        print("No cases found to audit!")
        return

    df = pd.DataFrame(records)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved audit of {len(df)} cases to {args.out_csv}")

    summary = {
        'total_cases_audited': len(df),
        'complete_4_organ_cases': int(df['is_complete_gi'].sum()),
        'pct_complete': round(float(df['is_complete_gi'].mean() * 100), 2),
        'stomach_present_pct': round(float(df['stomach_present'].mean() * 100), 2),
        'duodenum_present_pct': round(float(df['duodenum_present'].mean() * 100), 2),
        'small_bowel_present_pct': round(float(df['small_bowel_present'].mean() * 100), 2),
        'colon_present_pct': round(float(df['colon_present'].mean() * 100), 2),
        'by_cohort': {}
    }

    for cohort, grp in df.groupby('parent_dir'):
        summary['by_cohort'][cohort] = {
            'total': len(grp),
            'complete': int(grp['is_complete_gi'].sum()),
            'pct_complete': round(float(grp['is_complete_gi'].mean() * 100), 2),
            'small_bowel_present_pct': round(float(grp['small_bowel_present'].mean() * 100), 2),
        }

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary metrics to {args.out_json}:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
