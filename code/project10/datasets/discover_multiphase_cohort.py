#!/usr/bin/env python3
"""
code/project10/datasets/discover_multiphase_cohort.py
======================================================
Phase 1 — Step 1.1: Multi-Phase Cohort Discovery & Pairing Manifest Builder

Scans the CancerVerse metadata CSV to identify all patients with at least one
Non-Contrast CT (P0) and at least one post-contrast phase (P1, P2, or P3).
For each paired patient, it builds structured training tuples:

    (subject_p0, subject_pN, phase_bucket, patient_id, same_study_day, ct_path_p0, ct_path_pN)

Outputs:
  results/project10/manifests/multiphase_pairs.csv   — one row per phase pair
  results/project10/manifests/multiphase_triplets.csv — rows where P0+P1+P3 all exist
  results/project10/manifests/discovery_stats.json    — summary statistics

Usage:
    python code/project10/datasets/discover_multiphase_cohort.py \
        --metadata CancerVerse/CancerVerse_dataset_metadata.csv \
        --data_root CancerVerse/CancerVerse \
        --out_dir results/project10/manifests \
        [--min_phases 2]           # default 2 (NCCT + at least 1 post-contrast)
        [--same_day_only]          # restrict to same exam_date groupings
        [--verify_disk]            # check ct.nii.gz exists for every subject (default: True)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from itertools import combinations
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase Normalisation Map
# ---------------------------------------------------------------------------
PHASE_BUCKET_MAP = {
    "non-contrast":   "P0_ncct",
    "arterial":       "P1_arterial_early",
    "arterial_early": "P1_arterial_early",
    "arterial_late":  "P2_arterial_late",
    "portal_venous":  "P3_venous",
    "venous":         "P3_venous",
    # "u" and unknown — excluded from pairing
}

# Priority ordering for source selection (lower = preferred source phase)
PHASE_PRIORITY = {
    "P0_ncct":            0,
    "P1_arterial_early":  1,
    "P2_arterial_late":   2,
    "P3_venous":          3,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Project 10 Step 1.1: Multi-Phase Cohort Discovery & Manifest Builder"
    )
    p.add_argument(
        "--metadata",
        type=str,
        default="CancerVerse/CancerVerse_dataset_metadata.csv",
        help="Path to CancerVerse_dataset_metadata.csv",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="CancerVerse/CancerVerse",
        help="Root directory containing CV_XXXXXXXX subject folders",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="results/project10/manifests",
        help="Directory to write output CSV/JSON manifests",
    )
    p.add_argument(
        "--min_phases",
        type=int,
        default=2,
        help="Minimum number of distinct phase buckets per patient to include (default: 2)",
    )
    p.add_argument(
        "--same_day_only",
        action="store_true",
        help="If set, only include pairs acquired on the same exam_date",
    )
    p.add_argument(
        "--no_verify_disk",
        action="store_true",
        help="Skip on-disk ct.nii.gz existence check (faster but less safe)",
    )
    return p.parse_args()


def normalise_date(date_str: str) -> Optional[str]:
    """Normalise exam_date to YYYY-MM-DD; return None if unparseable."""
    try:
        return pd.to_datetime(date_str, infer_datetime_format=True).strftime("%Y-%m-%d")
    except Exception:
        return None


def build_manifest(
    metadata_path: str,
    data_root: str,
    out_dir: str,
    min_phases: int = 2,
    same_day_only: bool = False,
    verify_disk: bool = True,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    log.info(f"Loading metadata from: {metadata_path}")
    # Check header to conditionally read Encrypted Accession Number if present
    sample_cols = pd.read_csv(metadata_path, nrows=1).columns.tolist()
    use_cols = ["CancerVerse ID", "Patient ID", "phase", "exam_date"]
    if "Encrypted Accession Number" in sample_cols:
        use_cols.append("Encrypted Accession Number")

    df = pd.read_csv(
        metadata_path,
        usecols=use_cols,
        dtype=str,
    )
    if "Encrypted Accession Number" not in df.columns:
        df["Encrypted Accession Number"] = ""
    log.info(f"  Total rows loaded: {len(df):,}")

    # ------------------------------------------------------------------
    # Normalise phase buckets
    df["phase_bucket"] = df["phase"].str.strip().map(PHASE_BUCKET_MAP)
    excluded = df["phase_bucket"].isna().sum()
    df = df.dropna(subset=["phase_bucket"]).copy()
    log.info(f"  Rows after phase normalisation (excluded {excluded} unknown): {len(df):,}")

    # ------------------------------------------------------------------
    # Normalise exam dates
    df["exam_date_norm"] = df["exam_date"].apply(normalise_date)

    # ------------------------------------------------------------------
    # On-disk verification
    df["ct_path"] = df["CancerVerse ID"].apply(
        lambda s: os.path.join(data_root, str(s).strip(), "ct.nii.gz")
    )
    if verify_disk:
        log.info("  Verifying ct.nii.gz on disk (may take a moment)...")
        df["on_disk"] = df["ct_path"].apply(os.path.exists)
        n_missing = (~df["on_disk"]).sum()
        if n_missing > 0:
            log.warning(f"    {n_missing} subjects missing ct.nii.gz — excluded from manifest.")
        df = df[df["on_disk"]].copy()
        log.info(f"  Rows after on-disk filter: {len(df):,}")

    # ------------------------------------------------------------------
    # Identify patients with multi-phase coverage
    log.info("Building multi-phase patient cohort...")
    patient_phase_counts = df.groupby("Patient ID")["phase_bucket"].nunique()
    multi_phase_patients = patient_phase_counts[patient_phase_counts >= min_phases].index
    df_mp = df[df["Patient ID"].isin(multi_phase_patients)].copy()
    log.info(f"  Multi-phase patients (≥{min_phases} buckets): {len(multi_phase_patients):,}")

    # ------------------------------------------------------------------
    # Require NCCT (P0) as anchor source phase
    patients_with_ncct = set(df_mp[df_mp["phase_bucket"] == "P0_ncct"]["Patient ID"])
    df_mp = df_mp[df_mp["Patient ID"].isin(patients_with_ncct)].copy()
    log.info(f"  Patients with NCCT (P0) + post-contrast: {df_mp['Patient ID'].nunique():,}")

    # ------------------------------------------------------------------
    # Build pairwise rows: (P0, P_i) for each patient
    pair_rows = []
    triplet_rows = []

    for patient_id, grp in df_mp.groupby("Patient ID"):
        phases_present = grp["phase_bucket"].unique().tolist()

        # All NCCT candidates for this patient
        src_rows = grp[grp["phase_bucket"] == "P0_ncct"].sort_values(
            "exam_date_norm", ascending=False, na_position="last"
        )
        if src_rows.empty:
            continue

        # Target: all non-NCCT rows for this patient
        tgt_rows = grp[grp["phase_bucket"] != "P0_ncct"]

        for _, tgt in tgt_rows.iterrows():
            tgt_acc = str(tgt.get("Encrypted Accession Number", "")).strip()
            tgt_date = tgt["exam_date_norm"]
            has_tgt_date = pd.notna(tgt_date) and str(tgt_date).strip() != ""

            # Prioritize matching NCCT: 1. same accession, 2. same exam_date, 3. closest date
            matched_src = None
            temporal_status = "unverified_date_missing"
            same_day = False

            if tgt_acc and tgt_acc != "nan":
                acc_matches = src_rows[src_rows["Encrypted Accession Number"].str.strip() == tgt_acc]
                if not acc_matches.empty:
                    matched_src = acc_matches.iloc[0]
                    temporal_status = "verified_same_session"
                    same_day = True

            if matched_src is None and has_tgt_date:
                date_matches = src_rows[src_rows["exam_date_norm"] == tgt_date]
                if not date_matches.empty:
                    matched_src = date_matches.iloc[0]
                    src_acc_val = str(matched_src.get("Encrypted Accession Number", "")).strip()
                    if tgt_acc and tgt_acc != "nan" and src_acc_val and src_acc_val != "nan" and tgt_acc != src_acc_val:
                        # Conflicting accessions on same calendar date: separate exam procedures
                        temporal_status = "same_day_different_accession"
                    else:
                        temporal_status = "same_day_unverified_accession"
                    same_day = True

            if matched_src is None:
                # Find closest date if dates are available
                valid_date_srcs = src_rows[src_rows["exam_date_norm"].notna()].copy()
                if has_tgt_date and not valid_date_srcs.empty:
                    try:
                        tgt_dt = pd.to_datetime(tgt_date)
                        valid_date_srcs["dt_diff"] = (pd.to_datetime(valid_date_srcs["exam_date_norm"]) - tgt_dt).abs()
                        matched_src = valid_date_srcs.sort_values("dt_diff").iloc[0]
                        temporal_status = "verified_longitudinal"
                        same_day = False
                    except Exception:
                        matched_src = src_rows.iloc[0]
                else:
                    matched_src = src_rows.iloc[0]
                    temporal_status = "unverified_date_missing"
                    same_day = False

            src = matched_src
            has_p0_date = pd.notna(src["exam_date_norm"]) and str(src["exam_date_norm"]).strip() != ""

            if temporal_status == "unverified_date_missing" and has_p0_date and has_tgt_date:
                if src["exam_date_norm"] == tgt_date:
                    temporal_status = "same_day_unverified_accession"
                    same_day = True
                else:
                    temporal_status = "verified_longitudinal"
                    same_day = False

            if same_day_only and not same_day:
                continue

            pair_rows.append(
                {
                    "patient_id":             patient_id,
                    "subject_p0":             src["CancerVerse ID"],
                    "subject_target":         tgt["CancerVerse ID"],
                    "phase_source":           src["phase_bucket"],
                    "phase_target":           tgt["phase_bucket"],
                    "phase_target_raw":       tgt["phase"],
                    "exam_date_p0":           src["exam_date_norm"],
                    "exam_date_target":       tgt["exam_date_norm"],
                    "accession_p0":           src.get("Encrypted Accession Number", ""),
                    "accession_target":       tgt.get("Encrypted Accession Number", ""),
                    "same_day":               same_day,
                    "temporal_match_status":  temporal_status,
                    "ct_path_p0":             src["ct_path"],
                    "ct_path_target":         tgt["ct_path"],
                }
            )

        # Triplets: P0 + P1 + P3
        has_p1 = any("P1" in p for p in phases_present)
        has_p3 = any("P3" in p for p in phases_present)
        if has_p1 and has_p3:
            p1_candidates = grp[grp["phase_bucket"] == "P1_arterial_early"]
            p3_candidates = grp[grp["phase_bucket"] == "P3_venous"]
            # Pick the closest-date p1 and p3 to the NCCT source
            for _, p1 in p1_candidates.iterrows():
                for _, p3 in p3_candidates.iterrows():
                    triplet_rows.append(
                        {
                            "patient_id":       patient_id,
                            "subject_p0":       src["CancerVerse ID"],
                            "subject_p1":       p1["CancerVerse ID"],
                            "subject_p3":       p3["CancerVerse ID"],
                            "ct_path_p0":       src["ct_path"],
                            "ct_path_p1":       p1["ct_path"],
                            "ct_path_p3":       p3["ct_path"],
                            "exam_date_p0":     src["exam_date_norm"],
                            "exam_date_p1":     p1["exam_date_norm"],
                            "exam_date_p3":     p3["exam_date_norm"],
                            "same_day_p0_p1":   src["exam_date_norm"] == p1["exam_date_norm"],
                            "same_day_p0_p3":   src["exam_date_norm"] == p3["exam_date_norm"],
                        }
                    )

    pairs_df = pd.DataFrame(pair_rows)
    triplets_df = pd.DataFrame(triplet_rows)

    # ------------------------------------------------------------------
    # Save
    pairs_out = os.path.join(out_dir, "multiphase_pairs.csv")
    triplets_out = os.path.join(out_dir, "multiphase_triplets.csv")
    pairs_df.to_csv(pairs_out, index=False)
    triplets_df.to_csv(triplets_out, index=False)
    log.info(f"  Saved {len(pairs_df):,} pair rows -> {pairs_out}")
    log.info(f"  Saved {len(triplets_df):,} triplet rows -> {triplets_out}")

    # Audit Item 4: Explicitly separate verified same-session, longitudinal, and missing-date cohorts
    if not pairs_df.empty and "temporal_match_status" in pairs_df.columns:
        for status_val in ["verified_same_session", "verified_longitudinal", "unverified_date_missing"]:
            sub_df = pairs_df[pairs_df["temporal_match_status"] == status_val]
            sub_out = os.path.join(out_dir, f"multiphase_pairs_{status_val}.csv")
            sub_df.to_csv(sub_out, index=False)
            log.info(f"  Saved {len(sub_df):,} {status_val} rows -> {sub_out}")

    # ------------------------------------------------------------------
    # Summary stats
    stats = {
        "total_cv_rows_in_metadata": int(len(df) + excluded),
        "rows_after_phase_normalisation": int(len(df)),
        "patients_with_multi_phase": int(len(multi_phase_patients)),
        "patients_with_ncct_and_post_contrast": int(df_mp["Patient ID"].nunique()),
        "total_pair_rows": int(len(pairs_df)),
        "total_triplet_rows": int(len(triplets_df)),
        "same_day_pairs": int(pairs_df["same_day"].sum()) if "same_day" in pairs_df.columns else 0,
        "phase_target_distribution": pairs_df["phase_target"].value_counts().to_dict() if not pairs_df.empty else {},
        "same_day_only_mode": same_day_only,
        "verify_disk": verify_disk,
    }
    stats_out = os.path.join(out_dir, "discovery_stats.json")
    with open(stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"  Saved discovery stats -> {stats_out}")

    # ------------------------------------------------------------------
    # Human-readable summary
    log.info("\n" + "=" * 60)
    log.info("  PROJECT 10 PHASE 1 COHORT DISCOVERY SUMMARY")
    log.info("=" * 60)
    log.info(f"  CancerVerse rows scanned:              {stats['total_cv_rows_in_metadata']:>8,}")
    log.info(f"  Unique patients with multi-phase:      {stats['patients_with_multi_phase']:>8,}")
    log.info(f"  Paired patients (NCCT + post-contrast):{stats['patients_with_ncct_and_post_contrast']:>8,}")
    log.info(f"  Total pairwise training tuples:        {stats['total_pair_rows']:>8,}")
    log.info(f"  Same-day pairs (minimal motion):       {stats['same_day_pairs']:>8,}")
    log.info(f"  Full triplets (NCCT + P1 + P3):        {stats['total_triplet_rows']:>8,}")
    log.info(f"\n  Phase target distribution:")
    for k, v in stats["phase_target_distribution"].items():
        log.info(f"    {k:<30} {v:>5}")
    log.info("=" * 60)

    return pairs_df


if __name__ == "__main__":
    args = parse_args()
    pairs = build_manifest(
        metadata_path=args.metadata,
        data_root=args.data_root,
        out_dir=args.out_dir,
        min_phases=args.min_phases,
        same_day_only=args.same_day_only,
        verify_disk=not args.no_verify_disk,
    )
    log.info(f"\n[Done] Manifest ready. Use multiphase_pairs.csv as input to register_multiphase_pairs.py")
