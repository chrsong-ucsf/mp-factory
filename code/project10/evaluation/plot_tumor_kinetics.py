#!/usr/bin/env python3
"""
code/project10/evaluation/plot_tumor_kinetics.py
================================================
Generates publication-quality figures illustrating the multi-phase contrast
enhancement kinetics (NCCT -> Arterial -> Venous) across the 1,977 procedural
3D tumors generated for Project 10 (HCC, PDAC, Hemangioma, and Metastasis).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANIFEST_PATH = "/mnt/scratch/user/chrsong/mp-factory/results/project10/synthetic_tumors/tumor_manifest_master.csv"
OUT_FIG_PNG = "/mnt/scratch/user/chrsong/mp-factory/docs/figures/project10_tumor_kinetics_curves.png"
OUT_REPORT_PNG = "/mnt/scratch/user/chrsong/mp-factory/results/project10/evaluation_reports/tumor_kinetics_curves.png"

TUMOR_COLORS = {
    "hcc": "#E53935",         # Crimson
    "pdac": "#8E24AA",        # Deep Purple
    "hemangioma": "#FFB300",   # Amber / Gold
    "metastasis": "#1E88E5",   # Royal Blue
}

TUMOR_TITLES = {
    "hcc": "Hepatocellular Carcinoma (HCC)\nLiver Hyperenhancement & Washout",
    "pdac": "Pancreatic Ductal Adeno. (PDAC)\nPersistent Hypovascularity",
    "hemangioma": "Cavernous Hemangioma\nCentripetal Nodular Fill-in",
    "metastasis": "Liver Metastasis\nPeripheral Ring / Hypovascular",
}


def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded {len(df)} tumor records from {MANIFEST_PATH}")

    os.makedirs(os.path.dirname(OUT_FIG_PNG), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_REPORT_PNG), exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=300, sharey=True)
    phases = ["NCCT", "Arterial", "Venous"]

    for idx, (t_type, ax) in enumerate(zip(["hcc", "pdac", "hemangioma", "metastasis"], axes)):
        sub = df[df["Tumor_Type"] == t_type]
        color = TUMOR_COLORS[t_type]

        data = [
            sub["Ncct_Mean_HU"].values,
            sub["Arterial_Mean_HU"].values,
            sub["Venous_Mean_HU"].values,
        ]

        # Boxplot with semi-transparent points
        bp = ax.boxplot(
            data,
            tick_labels=phases,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            medianprops=dict(color="black", linewidth=1.5),
            meanprops=dict(color="red", linewidth=2.0, linestyle="--"),
            boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, linewidth=2),
            whiskerprops=dict(color=color, linewidth=1.5),
            capprops=dict(color=color, linewidth=1.5),
            flierprops=dict(marker="o", color=color, alpha=0.2, markersize=3),
        )

        # Plot mean trajectory line
        means = [np.mean(d) for d in data]
        ax.plot([1, 2, 3], means, color=color, linewidth=2.5, marker="o", markersize=7, label=f"Mean HU: {means[0]:.1f} -> {means[1]:.1f} -> {means[2]:.1f}")

        ax.set_title(TUMOR_TITLES[t_type], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Dynamic Contrast Phase", fontsize=11, fontweight="semibold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    axes[0].set_ylabel("Attenuation (Hounsfield Units, HU)", fontsize=12, fontweight="semibold")
    fig.suptitle(
        "Project 10: Multi-Phase Pharmacokinetic CT Contrast Enhancement Curves (N = 1,977 Lesions)\n"
        "Procedural 3D Tumors Generated with Physiologically Constrained LI-RADS / PDAC Kinetics",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(OUT_FIG_PNG, bbox_inches="tight")
    plt.savefig(OUT_REPORT_PNG, bbox_inches="tight")
    plt.close()

    print(f"Saved publication curves to:")
    print(f"  - {OUT_FIG_PNG}")
    print(f"  - {OUT_REPORT_PNG}")


if __name__ == "__main__":
    main()
