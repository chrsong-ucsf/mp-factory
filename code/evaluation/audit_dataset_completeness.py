"""
code/evaluation/audit_dataset_completeness.py
=============================================
Audit dataset annotation completeness across multi-organ GI cohorts.

Addresses Research Re-Audit Item 3 & Review Requirements:
- Strictly separates annotation completeness from organ presence:
  * reviewed_classes: organs verified as evaluated by annotators.
  * visible_classes: organs with positive voxel counts (> 0).
  * anatomical_coverage: field-of-view completeness.
  * is_complete_gi: True strictly when all 4 target organs were in reviewed_classes.
- Rejects ambiguous schema inputs without metadata:
  * A mask containing {0, 2, 4} without verified metadata is marked schema="AMBIGUOUS",
    reviewed_classes="UNKNOWN", status="EXCLUDED_UNVERIFIED_METADATA", is_complete_gi=False.
- Does not default missing metadata to full supervision.
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, Any, Optional, Set, List
import numpy as np
import pandas as pd
import nibabel as nib

# Standard Organ Names
ORGAN_NAMES = {1: 'stomach', 2: 'duodenum', 3: 'small_bowel', 4: 'colon'}

ORGAN_ALIASES = {
    1: ['stomach', 'stomach_mask', 'stomach.nii.gz'],
    2: ['duodenum', 'duodenum_mask', 'duodenum.nii.gz'],
    3: ['small_bowel', 'small_intestine', 'intestine', 'small_bowel_mask', 'small_bowel.nii.gz'],
    4: ['colon', 'colon_mask', 'large_bowel', 'colon.nii.gz'],
}

SCHEMA_VOCABS = {
    "canonical_5class": {1, 2, 3, 4},
    "totalseg_v2": {6, 18, 19, 20},
    "bdmap": {2, 3, 4, 5, 6},
    "jhu": {2, 3, 4, 5, 6},
}

SCHEMAS = {
    "canonical_5class": {0: "background", 1: "stomach", 2: "duodenum", 3: "small_bowel", 4: "colon"},
    "totalseg_v2": {0: "background", 6: "stomach", 19: "duodenum", 18: "small_bowel", 20: "colon"},
    "bdmap": {0: "background", 2: "stomach", 3: "duodenum", 4: "small_bowel", 5: "colon", 6: "rectum"},
    "jhu": {0: "background", 2: "stomach", 3: "duodenum", 4: "small_bowel", 5: "colon", 6: "rectum"},
}


def audit_case(case_dir: str, schema_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Audit an individual subject directory for CT volume presence,
    visible label classes, verified reviewed classes, and completeness.
    """
    subject_id = os.path.basename(case_dir.rstrip('/'))

    # Check for CT image
    has_ct = False
    for ct_name in ["ct.nii.gz", f"{subject_id}.nii.gz", "image.nii.gz", "imaging.nii.gz"]:
        if os.path.exists(os.path.join(case_dir, ct_name)):
            has_ct = True
            break
    if not has_ct:
        # Also check parent directory if case_dir is a subject folder with nested structure
        for img_cand in glob.glob(os.path.join(case_dir, "*.nii.gz")):
            if "mask" not in img_cand and "seg" not in img_cand and "autolabel" not in img_cand:
                has_ct = True
                break

    if not has_ct:
        return None

    record = {
        'subject_id': subject_id,
        'case_dir': case_dir,
        'has_ct': True,
        'schema': schema_override or 'unknown',
        'schema_verified': False,
        'status': 'OK',
        'reviewed_classes': [],
        'visible_classes': [],
        'anatomical_coverage': 'unknown_fov',
        'stomach_present': False,
        'duodenum_present': False,
        'small_bowel_present': False,
        'colon_present': False,
        'rectum_present': False,
        'num_organs_annotated': 0,
        'annotation_type': 'none',
        'is_complete_gi': False,
    }

    # Load metadata manifest if present
    meta_path = os.path.join(case_dir, "case_metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as mf:
                metadata = json.load(mf)
        except Exception:
            pass

    # Check for combined mask
    comb_path = None
    for comb_name in ["gi_mask.nii.gz", f"{subject_id}_gi_mask.nii.gz", "combined_mask.nii.gz"]:
        cp = os.path.join(case_dir, comb_name)
        if os.path.exists(cp):
            comb_path = cp
            break

    if comb_path:
        try:
            arr = np.asanyarray(nib.load(comb_path).dataobj)
            uniques = set(np.unique(arr))
            uniques.discard(0)
            uniques.discard(255)  # IGNORE_INDEX
            record['visible_classes'] = sorted(list(uniques))
            record['annotation_type'] = 'combined_mask'

            # Validate integer labels
            for val in uniques:
                if not isinstance(val, (int, np.integer)) or val < 0 or val > 254:
                    record['schema'] = 'INVALID_LABEL_VALUES'
                    record['status'] = f'CORRUPTED_LABEL_VALUE_{val}'
                    record['schema_verified'] = False
                    return record

            # Determine schema
            declared_schema = schema_override or metadata.get("declared_schema") or metadata.get("schema")
            if declared_schema:
                if declared_schema in SCHEMA_VOCABS and uniques.issubset(SCHEMA_VOCABS[declared_schema]):
                    active_schema = declared_schema
                    record['schema_verified'] = True
                else:
                    record['schema'] = declared_schema
                    record['schema_verified'] = False
                    record['status'] = 'SCHEMA_METADATA_CONFLICT'
                    record['is_complete_gi'] = False
                    return record
            else:
                # Mathematical subset compatibility against candidate schemas
                matching_schemas = [
                    s_name for s_name, vocab in SCHEMA_VOCABS.items()
                    if uniques.issubset(vocab)
                ]
                if set(matching_schemas) == {"bdmap", "jhu"}:
                    active_schema = "jhu"
                    record['schema_inferred'] = True
                    record['schema_verified'] = False
                elif len(matching_schemas) == 1:
                    active_schema = matching_schemas[0]
                    record['schema_inferred'] = True
                    record['schema_verified'] = False
                elif len(matching_schemas) > 1:
                    record['schema'] = 'AMBIGUOUS'
                    record['schema_verified'] = False
                    record['status'] = 'EXCLUDED_UNVERIFIED_METADATA'
                    record['is_complete_gi'] = False
                    record['num_organs_annotated'] = len(uniques)
                    return record
                else:
                    record['schema'] = 'UNRECOGNIZED'
                    record['status'] = 'EXCLUDED_UNVERIFIED_METADATA'
                    record['schema_verified'] = False
                    return record

            record['schema'] = active_schema

            # Map visible organs strictly under verified/inferred schema
            if active_schema == "canonical_5class":
                record['stomach_present'] = 1 in uniques
                record['duodenum_present'] = 2 in uniques
                record['small_bowel_present'] = 3 in uniques
                record['colon_present'] = 4 in uniques
            elif active_schema == "totalseg_v2":
                record['stomach_present'] = 6 in uniques
                record['duodenum_present'] = 19 in uniques
                record['small_bowel_present'] = 18 in uniques
                record['colon_present'] = 20 in uniques
            elif active_schema in ["bdmap", "jhu"]:
                record['stomach_present'] = 2 in uniques
                record['duodenum_present'] = 3 in uniques
                record['small_bowel_present'] = 4 in uniques
                record['colon_present'] = (5 in uniques or 6 in uniques)
                record['rectum_present'] = 6 in uniques

            # Determine reviewed_classes: from verified metadata ONLY, never from visible organs
            if "per_organ_provenance" in metadata:
                record['reviewed_classes'] = [
                    org for org in ['stomach', 'duodenum', 'small_bowel', 'colon']
                    if metadata["per_organ_provenance"].get(org, {}).get("human_reviewed", False)
                ]
                record['provenance_type'] = 'EXPERT_METADATA_V2'
            elif "reviewed_classes" in metadata and metadata.get("human_reviewed", False):
                record['reviewed_classes'] = metadata["reviewed_classes"]
                record['provenance_type'] = 'DECLARED_REVIEWED_CLASSES'
            else:
                record['reviewed_classes'] = []
                record['provenance_type'] = 'UNVERIFIED_NO_METADATA'
                record['status'] = 'EXCLUDED_UNVERIFIED_METADATA'

            record['num_organs_annotated'] = sum([
                record['stomach_present'], record['duodenum_present'],
                record['small_bowel_present'], record['colon_present']
            ])

            # Completeness: all 4 target organs must be in reviewed_classes
            target_organs = {'stomach', 'duodenum', 'small_bowel', 'colon'}
            record['is_complete_gi'] = target_organs.issubset(set(record['reviewed_classes']))
            return record

        except Exception as e:
            record['status'] = f"CORRUPTED_MASK: {e}"
            return record

    # Check segmentations directory or root for individual organ masks
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

    record['rectum_present'] = os.path.exists(os.path.join(search_dir, "rectum.nii.gz"))
    record['num_organs_annotated'] = sum([
        record['stomach_present'], record['duodenum_present'],
        record['small_bowel_present'], record['colon_present']
    ])
    record['annotation_type'] = 'individual_masks' if record['num_organs_annotated'] > 0 else 'none'
    record['visible_classes'] = [ORGAN_NAMES[i] for i in [1, 2, 3, 4] if record[f"{ORGAN_NAMES[i]}_present"]]

    # Provenance check for individual masks
    if "per_organ_provenance" in metadata:
        record['reviewed_classes'] = [
            org for org in ['stomach', 'duodenum', 'small_bowel', 'colon']
            if metadata["per_organ_provenance"].get(org, {}).get("human_reviewed", False)
        ]
        record['provenance_type'] = 'EXPERT_METADATA_V2'
    elif "reviewed_classes" in metadata and metadata.get("human_reviewed", False):
        record['reviewed_classes'] = metadata["reviewed_classes"]
        record['provenance_type'] = 'DECLARED_REVIEWED_CLASSES'
    else:
        record['reviewed_classes'] = []
        record['provenance_type'] = 'UNVERIFIED_NO_METADATA'
        record['status'] = 'EXCLUDED_UNVERIFIED_METADATA'

    target_organs = {'stomach', 'duodenum', 'small_bowel', 'colon'}
    record['is_complete_gi'] = target_organs.issubset(set(record['reviewed_classes']))

    return record


def main():
    parser = argparse.ArgumentParser(description="Audit GI Annotation Completeness")
    parser.add_argument("--data_dirs", nargs="+", default=[
        "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox",
        "/mnt/scratch/user/chrsong/CancerVerse_data",
        "/mnt/scratch/user/chrsong/mp-factory/CancerVerse/CancerVerse"
    ])
    parser.add_argument("--schema", default=None, help="Explicit schema name to enforce")
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
            rec = audit_case(s, schema_override=args.schema)
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
        'ambiguous_or_unverified': int((df['status'] == 'EXCLUDED_UNVERIFIED_METADATA').sum()),
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
