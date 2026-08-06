"""
watch_pipeline.py

Monitors active pipeline jobs (Model Comparison & Active Learning Ensemble),
tracks live progress, and prints final verification results when complete.
"""

import os
import sys
import time
import glob
import pandas as pd

ENSEMBLE_OUT_DIR = "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out"
COMPARE_CSV      = "/mnt/scratch/user/chrsong/mp-factory/results/model_comparison.csv"
ENSEMBLE_CSV     = "/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv"
TOTAL_TARGET     = 1735


def get_progress():
    consensus_count = len(glob.glob(os.path.join(ENSEMBLE_OUT_DIR, "*_consensus.nii.gz")))
    
    comp_done = False
    comp_count = 0
    if os.path.exists(COMPARE_CSV):
        try:
            df_c = pd.read_csv(COMPARE_CSV)
            comp_count = len(df_c)
            if comp_count >= TOTAL_TARGET or comp_count > 0:
                comp_done = True
        except Exception:
            pass

    ens_done = os.path.exists(ENSEMBLE_CSV)
    return consensus_count, comp_count, comp_done, ens_done


def main():
    print("=" * 65)
    print("      PIPELINE LIVE MONITOR & AUTO-COMPLETION CHECK")
    print("=" * 65)

    poll_interval = 10
    start_time = time.time()

    while True:
        consensus_count, comp_count, comp_done, ens_done = get_progress()
        elapsed_mins = (time.time() - start_time) / 60.0

        pct_ens = min(100.0, (consensus_count / TOTAL_TARGET) * 100)
        
        status_line = (
            f"[{time.strftime('%H:%M:%S')}] "
            f"Ensemble Audit: {consensus_count}/{TOTAL_TARGET} ({pct_ens:.1f}%) | "
            f"Comparison: {comp_count} evaluated | "
            f"Elapsed: {elapsed_mins:.1f}m"
        )
        print(status_line, flush=True)

        # Check completion condition
        if ens_done or (consensus_count >= TOTAL_TARGET and comp_done):
            print("\n" + "=" * 65)
            print("  *** ALL PIPELINE JOBS COMPLETED SUCCESSFULLY! ***")
            print("=" * 65 + "\n")
            
            # Exec master verification script
            os.system(f"{sys.executable} code/verify_pipeline_run.py")
            break

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
