"""
merge_ensemble_chunks.py

Merges per-chunk ensemble audit CSVs produced by run_ensemble_eval.sbatch SLURM array jobs
(ensemble_audit_summary_0.csv, _1.csv, ...) into a single final summary file.

Usage:
    python merge_ensemble_chunks.py \
        --pattern "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary_*.csv" \
        --out_csv "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv"
"""

import argparse
import glob
import os
import sys
import pandas as pd

# Reuse the single source of truth for dataset-split export from the evaluation
# module so merged (sharded) runs produce identical splits to a single run.
_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
try:
    from evaluate_multi_model_ensemble import export_dataset_splits
except Exception:  # pragma: no cover - splits are optional if import fails
    export_dataset_splits = None


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-chunk ensemble audit CSVs into a single summary file."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Glob pattern matching all chunk CSVs (e.g. '/path/to/ensemble_audit_summary_*.csv')"
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output path for the merged CSV"
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="subject_id",
        help="Column to sort the merged DataFrame by (default: subject_id)"
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=None,
        help="Directory for exported dataset splits (default: <out_csv dir>/dataset_splits). "
             "Pass 'none' to skip split export."
    )
    parser.add_argument(
        "--ensemble_out_dir",
        type=str,
        default=None,
        help="Consensus mask directory, embedded as consensus_path in split CSVs."
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"ERROR: No files found matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(files)} chunk file(s):")
    dfs = []
    for f in files:
        df_chunk = pd.read_csv(f)
        print(f"  {f}: {len(df_chunk):,} rows")
        dfs.append(df_chunk)

    merged = pd.concat(dfs, ignore_index=True)

    # Validate duplicate subjects and reject conflicting entries (F07)
    if 'subject_id' in merged.columns:
        dup_mask = merged.duplicated(subset=['subject_id'], keep=False)
        if dup_mask.any():
            dups = merged[dup_mask]
            conflicts = []
            for sid, grp in dups.groupby('subject_id'):
                for col in ['triage_category', 'consensus_sha256', 'mean_consensus_dice', 'status']:
                    if col in grp.columns and grp[col].nunique() > 1:
                        conflicts.append((sid, col, grp[col].tolist()))
                        break
            if conflicts:
                raise ValueError(
                    f"Ensemble shards contain {len(conflicts)} conflicting duplicate subjects (e.g. {conflicts[:3]}). "
                    "Cannot merge inconsistent or corrupt worker shards."
                )
            print(f"Deduplicating {len(dups) - dups['subject_id'].nunique()} exact duplicate rows across shards.")
            merged = merged.drop_duplicates(subset=['subject_id']).reset_index(drop=True)

    if args.sort_by in merged.columns:
        merged = merged.sort_values(args.sort_by).reset_index(drop=True)

    merged.to_csv(args.out_csv, index=False)

    # Summary stats
    print(f"\nMerged {len(files)} chunks -> {len(merged):,} total rows -> {args.out_csv}")
    if 'status' in merged.columns:
        status_counts = merged['status'].value_counts().to_dict()
        for status, count in status_counts.items():
            print(f"  {status}: {count:,} ({count/len(merged)*100:.1f}%)")
    if 'triage_category' in merged.columns:
        print("\n[AUTOMATED DATA CLEANSING BREAKDOWN]")
        triage_counts = merged['triage_category'].value_counts().to_dict()
        for cat, count in triage_counts.items():
            pct = count / len(merged) * 100.0
            print(f"  {cat:<45}: {count:,} cases ({pct:.1f}%)")
    if 'mean_consensus_dice' in merged.columns:
        valid = merged[merged['status'] == 'SUCCESS']
        print(f"\nMean Consensus Dice   : {valid['mean_consensus_dice'].mean():.4f}")
        print(f"Mean Inter-Model Dice : {valid['mean_inter_model_dice'].mean():.4f}")
        print(f"Mean Uncertainty      : {valid['mean_uncertainty'].mean():.4f}")

    # Export dataset splits from the merged CSV (single source of truth in
    # evaluate_multi_model_ensemble.export_dataset_splits) unless disabled.
    splits_dir = args.splits_dir
    if isinstance(splits_dir, str) and splits_dir.lower() == "none":
        print("\n[DATASET SPLITS] Skipped (--splits_dir none).")
    elif export_dataset_splits is None:
        print("\n[DATASET SPLITS] Skipped (could not import export_dataset_splits).")
    elif 'triage_category' in merged.columns:
        if not splits_dir:
            base = os.path.dirname(args.out_csv) or "."
            splits_dir = os.path.join(base, "dataset_splits")
        export_dataset_splits(merged, splits_dir, ensemble_out_dir=args.ensemble_out_dir)


if __name__ == "__main__":
    main()
