#!/usr/bin/env python3
"""
plot_scaling_curves.py

Generates publication-quality scaling law figures from post-hoc evaluation CSVs.
Produces:
  1. Mean Dice vs. Cohort Size (with seed std bands)
  2. Per-organ Dice breakdown vs. Cohort Size
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_curves(csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Clean cohort size string to numeric (e.g. N0004 -> 4)
    df["N"] = df["cohort"].str.replace("N", "").astype(int)

    # Aggregate by N
    agg = df.groupby("N").agg({
        "mean_dice": ["mean", "std"],
        "dice_stomach": ["mean", "std"],
        "dice_duodenum": ["mean", "std"],
        "dice_colon": ["mean", "std"],
        "dice_small_bowel": ["mean", "std"]
    }).sort_index()

    n_vals = agg.index.values

    # 1. Overall Mean Dice
    plt.figure(figsize=(7, 5), dpi=300)
    means = agg["mean_dice"]["mean"].values
    stds = agg["mean_dice"]["std"].values

    plt.plot(n_vals, means, marker='o', linewidth=2.5, color='#1f77b4', label='Mean Dice across Seeds')
    plt.fill_between(n_vals, means - stds, means + stds, color='#1f77b4', alpha=0.2, label='±1 Std Dev')

    plt.title("MedNeXt-B Phase 2 Scaling Law: Dice vs. Training Cohort Size", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Training Cohort Size (N)", fontsize=11, labelpad=8)
    plt.ylabel("Mean Dice Score", fontsize=11, labelpad=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_main = os.path.join(out_dir, "scaling_curve_overall.png")
    plt.savefig(out_main)
    plt.close()
    print(f"Saved: {out_main}")

    # 2. Per-Organ Breakdown
    plt.figure(figsize=(8, 5.5), dpi=300)
    colors = {
        "Stomach": ("#2ca02c", agg["dice_stomach"]["mean"].values),
        "Duodenum": ("#ff7f0e", agg["dice_duodenum"]["mean"].values),
        "Colon": ("#9467bd", agg["dice_colon"]["mean"].values),
        "Small Bowel": ("#d62728", agg["dice_small_bowel"]["mean"].values),
    }

    for name, (col, organ_means) in colors.items():
        plt.plot(n_vals, organ_means, marker='s', linewidth=2.0, color=col, label=name)

    plt.title("Per-Organ Generalization Scaling (MedNeXt-B)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Training Cohort Size (N)", fontsize=11, labelpad=8)
    plt.ylabel("Dice Score", fontsize=11, labelpad=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_organs = os.path.join(out_dir, "scaling_curve_per_organ.png")
    plt.savefig(out_organs)
    plt.close()
    print(f"Saved: {out_organs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot scaling sweep curves")
    parser.add_argument("--csv", type=str, required=True, help="Path to evaluation CSV")
    parser.add_argument("--out_dir", type=str, default="figures", help="Output directory for plots")
    args = parser.parse_args()
    plot_curves(args.csv, args.out_dir)
