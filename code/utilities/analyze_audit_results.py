import os
import pandas as pd
import numpy as np

def generate_audit_report(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df_valid = df[df['status'].astype(str).str.upper() == 'SUCCESS'].copy()
    
    total_cases = len(df)
    valid_cases = len(df_valid)

    print("=" * 70)
    print("           PHASE 1 CONSENSUS AUDIT ANALYSIS REPORT          ")
    print("=" * 70)
    print(f"Total Evaluated Scan Pairs : {total_cases}")
    print(f"Successfully Processed      : {valid_cases}")
    print("-" * 70)

    if 'overall_ari' in df_valid and 'overall_voi' in df_valid:
        print(f"Global Adjusted Rand Index (ARI) : {df_valid['overall_ari'].mean():.4f}")
        print(f"Global Variation of Info (VOI)   : {df_valid['overall_voi'].mean():.4f}")
        print("-" * 70)

    organs = ['stomach', 'duodenum', 'small_bowel', 'colon']
    print(f"{'Organ':<15} | {'Mean Dice':<10} | {'Mean IoU':<10} | {'Mean HD95 (mm)':<15}")
    print("-" * 70)

    for o in organs:
        dice_col = f'{o}_dice'
        iou_col = f'{o}_iou'
        hd_col = f'{o}_hd95'
        
        m_dice = df_valid[dice_col].mean() if dice_col in df_valid else np.nan
        m_iou = df_valid[iou_col].mean() if iou_col in df_valid else np.nan
        m_hd = df_valid[hd_col].mean() if hd_col in df_valid else np.nan

        print(f"{o:<15} | {m_dice:<10.4f} | {m_iou:<10.4f} | {m_hd:<15.2f}")

    overall_mean_dice = df_valid['mean_gi_dice'].mean()
    print("-" * 70)
    print(f"OVERALL MEAN GI DICE: {overall_mean_dice:.4f}")
    print("-" * 70)

    # Compute max Betti difference across GI organs for topological filter
    betti_diff_cols = [c for c in df_valid.columns if c.endswith('_betti_diff')]
    df_valid['max_betti_diff'] = df_valid[betti_diff_cols].max(axis=1) if betti_diff_cols else 0

    # Categorization Buckets
    reject_mask = (df_valid['mean_gi_dice'] < 0.50) | (df_valid['max_betti_diff'] > 5)
    clean_mask = (df_valid['mean_gi_dice'] >= 0.82) & (~reject_mask)
    weak_mask = ~clean_mask & ~reject_mask

    n_clean = clean_mask.sum()
    n_weak = weak_mask.sum()
    n_reject = reject_mask.sum()

    print("\n[ACTIONABLE DATASET DECISION BREAKDOWN]")
    print(f"  1. Clean High-Confidence Labels (Dice >= 0.82)        : {n_clean} cases ({n_clean/valid_cases*100:.1f}%)")
    print(f"  2. Weak / Coarse Labels (0.50 <= Dice < 0.82)         : {n_weak} cases ({n_weak/valid_cases*100:.1f}%)")
    print(f"  3. High Noise / Topological Failure (|Δβ0| > 5 / <0.5): {n_reject} cases ({n_reject/valid_cases*100:.1f}%)")
    
    print("\n[NEXT STEP GUIDANCE]")
    if overall_mean_dice >= 0.82 and (n_reject / valid_cases) < 0.05:
        print(" -> OUTCOME A: Dataset aligns strongly with multi-center consensus. Proceed directly to standard supervised training.")
    elif overall_mean_dice >= 0.65:
        print(" -> OUTCOME B: Dataset exhibits moderate annotator bias/coarseness. Proceed to Phase 3 (Hard-Threshold Auto-labeling with 'ignore' class).")
    else:
        print(" -> OUTCOME C: Severe annotator divergence. Discard manual masks and rely exclusively on TotalSegmentator Teacher GKD Distillation (Phase 2).")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    csv_p = sys.argv[1] if len(sys.argv) > 1 else "/mnt/scratch/user/chrsong/mp-factory/results/phase1_audit_summary.csv"
    generate_audit_report(csv_p)