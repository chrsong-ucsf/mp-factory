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
import sys
import pandas as pd


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


if __name__ == "__main__":
    main()
