"""
run_active_learning.py

Main script that configures MONAI pipelines, uncertainty estimation,
and executes the Active Learning Triage & Querying loop.

Usage:
  python run_active_learning.py \
    --audit_csv /mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv \
    --top_k 10 \
    --out_json /mnt/scratch/user/chrsong/mp-factory/results/radiologist_query_queue.json
"""

import os
import sys
import argparse
import json
import pandas as pd

from active_learning import (
    UncertaintyEstimator,
    ActiveLearningPool,
    ActiveLearningOrchestrator
)


def main():
    parser = argparse.ArgumentParser(description="Run Active Learning Triage & Radiologist Querying")
    parser.add_argument("--audit_csv", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv",
                        help="Path to ensemble audit summary CSV")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top most valuable edge cases to query for radiologist review")
    parser.add_argument("--out_json", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/radiologist_query_queue.json",
                        help="Output JSON file for targeted radiologist review queue")
    args = parser.parse_args()

    if not os.path.exists(args.audit_csv):
        print(f"ERROR: Audit CSV file not found: {args.audit_csv}")
        sys.exit(1)

    print("=" * 65)
    print("      ACTIVE LEARNING RADIOLOGIST TRIAGE & QUERY PIPELINE")
    print("=" * 65)

    df = pd.read_csv(args.audit_csv)
    all_subjects = df['subject_id'].astype(str).tolist()

    pool = ActiveLearningPool(all_subjects)
    estimator = UncertaintyEstimator(mode="entropy")
    orchestrator = ActiveLearningOrchestrator(pool, estimator)

    # Run active learning query cycle
    cycle_info = orchestrator.run_cycle(args.audit_csv, top_k=args.top_k)

    # Build detailed metadata for requested radiologist review scans
    query_details = []
    for sid in cycle_info['queried_cases']:
        row = df[df['subject_id'].astype(str) == sid].iloc[0].to_dict()
        query_details.append({
            'subject_id': sid,
            'mean_uncertainty': float(row.get('mean_uncertainty', 0.0)),
            'max_uncertainty': float(row.get('max_uncertainty', 0.0)),
            'max_betti_diff': int(row.get('max_betti_diff', 0)),
            'triage_category': str(row.get('triage_category', 'REJECT_OR_TRIAGE')),
            'action': str(row.get('action', 'Route to Active Learning Radiologist Queue'))
        })

    output_data = {
        'total_scans_audited': len(df),
        'top_k_queried': args.top_k,
        'radiologist_review_queue': query_details
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSuccessfully generated targeted radiologist review queue ({len(query_details)} cases)!")
    print(f"Queue exported to: {args.out_json}")
    print("=" * 65)


if __name__ == "__main__":
    main()
