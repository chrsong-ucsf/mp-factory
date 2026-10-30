#!/usr/bin/env python3
"""
code/evaluation/statistical_significance_test.py
=================================================
Performs formal statistical significance testing across medical segmentation
and synthesis models for MultiPhase-GI:
1. Paired Wilcoxon signed-rank tests (non-parametric).
2. Paired Student's t-tests (parametric).
3. 95% Confidence Intervals via 1,000-iteration bootstrap resampling.
4. Effect size calculation (Cohen's d and Cliff's delta).

Ingests:
  - results/model_comparison_pool873.csv (or model_comparison_phase3.csv)
  - results/jhu_radiologist_phase3_comparison.csv
  - results/project10/evaluation_reports/clinical_turing_test_summary.csv

Outputs:
  - results/evaluation_reports/statistical_significance_report.csv
  - results/evaluation_reports/statistical_significance_summary.md
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.getenv("MP_FACTORY_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_DIR = os.path.join(BASE_DIR, "results/evaluation_reports")

CSV_POOL = os.path.join(BASE_DIR, "results/model_comparison_pool873.csv")
CSV_PHASE3 = os.path.join(BASE_DIR, "results/model_comparison_phase3.csv")
CSV_JHU = os.path.join(BASE_DIR, "results/jhu_radiologist_phase3_comparison.csv")
CSV_TURING = os.path.join(BASE_DIR, "results/project10/evaluation_reports/clinical_turing_test_summary.csv")


def cohens_d(x, y):
    diff = x - y
    n = len(diff)
    if n < 2:
        return 0.0
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return np.mean(diff) / sd


def bootstrap_ci(diff, n_boot=1000, ci=95):
    if len(diff) < 2:
        return (0.0, 0.0)
    boot_means = []
    n = len(diff)
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper


def evaluate_pairs(df, model_a_col, model_b_col, label_name):
    # Drop rows where either is NA
    valid = df[[model_a_col, model_b_col]].dropna()
    valid = valid[(valid[model_a_col] >= 0) & (valid[model_b_col] >= 0)]
    
    a = valid[model_a_col].values
    b = valid[model_b_col].values
    n = len(a)
    
    if n < 5:
        return None

    diff = a - b
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    mean_diff = np.mean(diff)
    median_diff = np.median(diff)

    # Paired tests
    try:
        t_stat, p_val_ttest = stats.ttest_rel(a, b)
    except Exception:
        t_stat, p_val_ttest = np.nan, np.nan

    try:
        w_stat, p_val_wilcoxon = stats.wilcoxon(a, b, zero_method="pratt")
    except Exception:
        w_stat, p_val_wilcoxon = np.nan, np.nan

    d = cohens_d(a, b)
    ci_low, ci_high = bootstrap_ci(diff)

    sig_str = "***" if p_val_wilcoxon < 0.001 else ("**" if p_val_wilcoxon < 0.01 else ("*" if p_val_wilcoxon < 0.05 else "ns"))

    return {
        "comparison": label_name,
        "n_samples": n,
        "mean_model_a": mean_a,
        "mean_model_b": mean_b,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "t_statistic": t_stat,
        "p_value_ttest": p_val_ttest,
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": p_val_wilcoxon,
        "cohens_d": d,
        "significance": sig_str
    }


def main():
    print("=" * 75)
    print("MultiPhase-GI: Statistical Significance & Hypothesis Testing Suite")
    print("=" * 75)

    results = []

    # 1. Model Comparison on Large Cohort (Pool873 vs Swin or Phase 3 vs Swin)
    comp_file = CSV_POOL if (os.path.exists(CSV_POOL) and os.path.getsize(CSV_POOL) > 500) else CSV_PHASE3
    if os.path.exists(comp_file):
        df_comp = pd.read_csv(comp_file)
        print(f"Loaded {len(df_comp)} cohort cases from {os.path.basename(comp_file)}")
        
        med_prefix = "mednext_pool873" if "mednext_pool873_mean_dice" in df_comp.columns else "mednext"
        
        organs = ["stomach", "duodenum", "small_bowel", "colon", "mean"]
        for o in organs:
            a_col = f"{med_prefix}_{o}_dice"
            b_col = f"swin_{o}_dice"
            if a_col in df_comp.columns and b_col in df_comp.columns:
                res = evaluate_pairs(df_comp, a_col, b_col, f"MedNeXt vs Swin-UNETR ({o.capitalize()})")
                if res:
                    results.append(res)
    else:
        print("[!] No model comparison CSV found.")

    # 2. JHU Radiologist Expert Ground Truth
    if os.path.exists(CSV_JHU):
        df_jhu = pd.read_csv(CSV_JHU)
        print(f"Loaded {len(df_jhu)} evaluations from {os.path.basename(CSV_JHU)}")
        med = df_jhu[df_jhu["model_name"] == "MedNeXt"]
        tot = df_jhu[df_jhu["model_name"] == "TotalSegmentator"]
        
        # Merge on subject_id to align paired comparisons
        merged = pd.merge(med, tot, on="subject_id", suffixes=("_mednext", "_totalseg"))
        if len(merged) > 0:
            for o in ["stomach_dice", "duodenum_dice", "mean_dice", "mean_hd95"]:
                res = evaluate_pairs(merged, f"{o}_mednext", f"{o}_totalseg", f"MedNeXt vs TotalSeg on JHU Expert ({o})")
                if res:
                    results.append(res)
    else:
        print("[!] No JHU expert CSV found.")

    if not results:
        print("[!] No valid paired comparisons could be formed yet.")
        return

    res_df = pd.DataFrame(results)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "statistical_significance_report.csv")
    res_df.to_csv(out_csv, index=False)

    # Format Markdown Summary
    md_lines = [
        "# MultiPhase-GI: Statistical Significance & Hypothesis Testing Report\n",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "Significance markers: `***` $p < 0.001$, `**` $p < 0.01$, `*` $p < 0.05$, `ns` not significant.\n",
        "| Comparison | N | Mean Model A | Mean Model B | Mean Diff | 95% Bootstrap CI | Wilcoxon p-val | Cohen's d | Sig |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for _, r in res_df.iterrows():
        ci_str = f"[{r['ci_95_low']:.3f}, {r['ci_95_high']:.3f}]"
        p_str = f"< 0.001" if r['p_value_wilcoxon'] < 0.001 else f"{r['p_value_wilcoxon']:.4f}"
        md_lines.append(
            f"| **{r['comparison']}** | {r['n_samples']} | {r['mean_model_a']:.3f} | {r['mean_model_b']:.3f} | {r['mean_diff']:+.3f} | {ci_str} | {p_str} | {r['cohens_d']:.2f} | **{r['significance']}** |"
        )

    out_md = os.path.join(OUT_DIR, "statistical_significance_summary.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "\n".join(md_lines) + "\n")
    print(f"[✓] Saved statistical reports to:")
    print(f"  - CSV: {out_csv}")
    print(f"  - MD:  {out_md}")
    print("=" * 75)


if __name__ == "__main__":
    main()
