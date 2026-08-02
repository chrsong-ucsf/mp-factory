"""
verify_pipeline_run.py

Verifies all output deliverables from the MedNeXt inference, Model Comparison,
and Multi-Model Active Learning Ensemble SLURM pipeline run.
"""

import os
import glob
import pandas as pd

MEDNEXT_PRED_DIR = "/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions"
SWIN_PRED_DIR    = "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions"
COMPARE_CSV      = "/mnt/scratch/user/chrsong/mp-factory/results/model_comparison.csv"
ENSEMBLE_OUT_DIR = "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out"
ENSEMBLE_CSV     = "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv"


def main():
    print("=" * 65)
    print("      PROJECT 9 PIPELINE RUN VERIFICATION REPORT")
    print("=" * 65)

    # 1. MedNeXt Predictions
    mednext_files = glob.glob(os.path.join(MEDNEXT_PRED_DIR, "*.nii.gz"))
    swin_files    = glob.glob(os.path.join(SWIN_PRED_DIR, "*.nii.gz"))
    print(f"\n[1] PREDICTION MASKS GENERATED:")
    print(f"  - Swin-UNETR Masks  : {len(swin_files):,} files")
    print(f"  - MedNeXt Masks     : {len(mednext_files):,} files")

    # 2. Model Comparison CSV
    print(f"\n[2] MODEL COMPARISON SUMMARY ({COMPARE_CSV}):")
    if os.path.exists(COMPARE_CSV):
        try:
            df_comp = pd.read_csv(COMPARE_CSV)
            if len(df_comp) > 0:
                print(f"  - Evaluated Subjects : {len(df_comp)}")
                if 'winner' in df_comp.columns:
                    win_counts = df_comp['winner'].value_counts().to_dict()
                    print(f"  - Head-to-Head Wins  : {win_counts}")
                if 'swin_mean_dice' in df_comp.columns and 'mednext_mean_dice' in df_comp.columns:
                    swin_avg = pd.to_numeric(df_comp['swin_mean_dice'], errors='coerce').mean()
                    mednext_avg = pd.to_numeric(df_comp['mednext_mean_dice'], errors='coerce').mean()
                    print(f"  - Swin-UNETR Avg Dice: {swin_avg:.4f}")
                    print(f"  - MedNeXt Avg Dice   : {mednext_avg:.4f}")
            else:
                print("  - [IN PROGRESS] Comparison job is actively writing results...")
        except Exception:
            print("  - [IN PROGRESS] Comparison job is actively writing results...")
    else:
        print(f"  - [PENDING] {COMPARE_CSV} not generated yet.")

    # 3. Active Learning Ensemble Outputs
    print(f"\n[3] ACTIVE LEARNING ENSEMBLE AUDIT ({ENSEMBLE_CSV}):")
    consensus_files   = glob.glob(os.path.join(ENSEMBLE_OUT_DIR, "*_consensus.nii.gz"))
    uncertainty_files = glob.glob(os.path.join(ENSEMBLE_OUT_DIR, "*_uncertainty.nii.gz"))
    print(f"  - Consensus Masks Saved   : {len(consensus_files):,} files")
    print(f"  - Uncertainty Heatmaps    : {len(uncertainty_files):,} files")

    if os.path.exists(ENSEMBLE_CSV):
        df_ens = pd.read_csv(ENSEMBLE_CSV)
        print(f"  - Total Scans Audited     : {len(df_ens)}")
        if 'triage_category' in df_ens.columns:
            triage_counts = df_ens['triage_category'].value_counts().to_dict()
            print(f"\n  [ACTIVE LEARNING TRIAGE BREAKDOWN]:")
            for cat, count in triage_counts.items():
                pct = (count / len(df_ens)) * 100
                print(f"    - {cat:<25}: {count:,} scans ({pct:.1f}%)")
    else:
        print(f"  - [PENDING] {ENSEMBLE_CSV} not generated yet.")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
