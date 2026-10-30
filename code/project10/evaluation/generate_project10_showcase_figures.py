#!/usr/bin/env python3
"""
code/project10/evaluation/generate_project10_showcase_figures.py
===============================================================
Generates high-resolution publication-quality showcase figures for Project 10:
  1. Clinical Turing Comparison (NCCT -> Synthesised Contrast -> Real Contrast -> Residual)
     Rendered in standardized abdominal soft-tissue windows
     (WL: 40, WW: 400 -> [-160, +240] HU) with dynamic empirical metrics.
  2. Multi-Phase Procedural Tumor Hemodynamics (HCC wash-in & venous washout)
     with clear background abdominal anatomy.
"""

import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ResizeWithPadOrCropd,
)

BASE_DIR = Path(__file__).resolve().parents[3]
OUT_FIG_DIR = os.path.join(BASE_DIR, "docs/figures")
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def apply_ct_window(arr: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
    """
    Applies standard diagnostic CT windowing and normalizes to [0, 1].
    Default is Abdominal Soft Tissue: Window Level (WL) = +40 HU, Window Width (WW) = 400 HU.
    """
    vmin = wl - (ww / 2.0)  # default: -160 HU
    vmax = wl + (ww / 2.0)  # default: +240 HU
    clipped = np.clip(arr, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin)


def generate_comparison_figure():
    """Renders 4-panel comparison of true native NCCT, synthesis, real contrast, and residual."""
    pair_id = "CV_00000042--CV_00007115"
    synth_path = os.path.join(BASE_DIR, f"results/project10/synthetic_volumes/{pair_id}_synthetic_contrast.nii.gz")
    reg_path = os.path.join(BASE_DIR, f"results/project10/registered_volumes/{pair_id}_reg.nii.gz")
    ncct_path = os.path.join(BASE_DIR, "CancerVerse/CancerVerse/CV_00000042/ct.nii.gz")
    if not os.path.exists(ncct_path):
        ncct_path = os.path.join(BASE_DIR, "CancerVerse_dbox/CV_00000042/ct.nii.gz")

    print(f"Rendering comparison figure for: {pair_id}")
    synth_nii = nib.load(synth_path)
    synth_arr = np.asanyarray(synth_nii.dataobj).astype(np.float32)

    # Transform native NCCT and registered target CT into exact model patch space
    t = Compose([
        LoadImaged(keys=["ncct", "reg"], image_only=True),
        EnsureChannelFirstd(keys=["ncct", "reg"]),
        Spacingd(keys=["ncct", "reg"], pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "bilinear")),
        ResizeWithPadOrCropd(keys=["ncct", "reg"], spatial_size=(128, 128, 128), mode="constant", value=-1000.0)
    ])

    data = t({"ncct": ncct_path, "reg": reg_path})
    ncct_arr = data["ncct"][0].numpy()
    reg_arr = data["reg"][0].numpy()

    # Residual map
    residual_arr = synth_arr - reg_arr

    # Empirical correlation and SSIM on visceral tissue
    mask = (reg_arr > -200.0) & (reg_arr < 300.0)
    r_val, _ = pearsonr(synth_arr[mask].flatten(), reg_arr[mask].flatten())

    synth_norm = np.clip((synth_arr - (-160.0)) / 400.0, 0.0, 1.0)
    reg_norm = np.clip((reg_arr - (-160.0)) / 400.0, 0.0, 1.0)
    s_val = ssim(synth_norm, reg_norm, data_range=1.0)

    # Find slice with maximum visceral tissue
    body_mask = (reg_arr > -500.0)
    best_z = int(np.argmax(body_mask.sum(axis=(0, 1))))
    if best_z < 10 or best_z > synth_arr.shape[2] - 10:
        best_z = synth_arr.shape[2] // 2

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor='white', dpi=300)

    # 1. Non-Contrast Input
    axes[0].imshow(apply_ct_window(ncct_arr[:, :, best_z]).T, cmap='gray', origin='lower')
    axes[0].set_title(f"A. Non-Contrast Input CT (NCCT)\n(True Native Acquisition, Slice Z={best_z})", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2. Project 10 Generative Synthesis
    axes[1].imshow(apply_ct_window(synth_arr[:, :, best_z]).T, cmap='gray', origin='lower')
    axes[1].set_title(f"B. Generative Contrast Synthesis\n(Project 10 Latent Diffusion)", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 3. Ground-Truth Real Registered Contrast CT
    axes[2].imshow(apply_ct_window(reg_arr[:, :, best_z]).T, cmap='gray', origin='lower')
    axes[2].set_title(f"C. Ground-Truth Real Contrast CT\n(Clinical Reference Standard)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # 4. Residual Error Map
    res_slice = residual_arr[:, :, best_z].T
    im4 = axes[3].imshow(res_slice, cmap='coolwarm', vmin=-150.0, vmax=+150.0, origin='lower')
    axes[3].set_title(f"D. Attenuation Residual (ΔHU)\n(Synthetic - Real CT)", fontsize=12, fontweight='bold')
    axes[3].axis('off')

    cbar = fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("Attenuation Difference (HU)", fontsize=10)

    fig.suptitle(
        f"Project 10 Generative Contrast CT Verification: Subject {pair_id}\n"
        f"Standard Abdominal Soft-Tissue Window (WL: 40, WW: 400 HU) | Empirical Pearson R = {r_val:.3f}, SSIM = {s_val:.3f}",
        fontsize=14, fontweight='bold', y=0.98
    )

    out_path = os.path.join(OUT_FIG_DIR, "project10_clinical_turing_comparison.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved updated comparison figure to: {out_path}")


def generate_tumor_showcase_figure():
    """Generates 4D tumor hemodynamic series with clear background liver parenchyma."""
    pair_id = "CV_00000042--CV_00007115_synthetic_contrast"
    tumor_dir = os.path.join(BASE_DIR, "results/project10/synthetic_tumors")
    mask_file = os.path.join(tumor_dir, f"{pair_id}_hcc_mask.nii.gz")
    ncct_file = os.path.join(tumor_dir, f"{pair_id}_hcc_ncct.nii.gz")
    art_file = os.path.join(tumor_dir, f"{pair_id}_hcc_arterial.nii.gz")
    ven_file = os.path.join(tumor_dir, f"{pair_id}_hcc_venous.nii.gz")

    if not os.path.exists(mask_file):
        print(f"[!] Tumor files missing for {pair_id}, skipping tumor showcase.")
        return

    m_nii = nib.load(mask_file)
    mask_arr = np.asanyarray(m_nii.dataobj).astype(bool)
    ncct_arr = np.asanyarray(nib.load(ncct_file).dataobj).astype(np.float32)
    art_arr = np.asanyarray(nib.load(art_file).dataobj).astype(np.float32)
    ven_arr = np.asanyarray(nib.load(ven_file).dataobj).astype(np.float32)

    # Find slice with maximum tumor cross-section
    slice_z = int(np.argmax(mask_arr.sum(axis=(0, 1))))

    # Attenuation measurements inside tumor
    t_ncct = ncct_arr[mask_arr].mean()
    t_art = art_arr[mask_arr].mean()
    t_ven = ven_arr[mask_arr].mean()
    washout = art_arr - ven_arr

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), facecolor='white', dpi=300)

    # 1. Native NCCT
    axes[0].imshow(apply_ct_window(ncct_arr[:, :, slice_z]).T, cmap='gray', origin='lower')
    axes[0].contour(mask_arr[:, :, slice_z].T, levels=[0.5], colors=['#E74C3C'], linewidths=1.5, origin='lower')
    axes[0].set_title(f"A. Native Baseline (NCCT)\nLesion Mean: {t_ncct:.1f} HU", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2. Arterial Phase
    axes[1].imshow(apply_ct_window(art_arr[:, :, slice_z]).T, cmap='gray', origin='lower')
    axes[1].contour(mask_arr[:, :, slice_z].T, levels=[0.5], colors=['#E74C3C'], linewidths=1.5, origin='lower')
    axes[1].set_title(f"B. Arterial Phase (Wash-in)\nLesion Mean: {t_art:.1f} HU", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 3. Portal Venous Phase
    axes[2].imshow(apply_ct_window(ven_arr[:, :, slice_z]).T, cmap='gray', origin='lower')
    axes[2].contour(mask_arr[:, :, slice_z].T, levels=[0.5], colors=['#E74C3C'], linewidths=1.5, origin='lower')
    axes[2].set_title(f"C. Portal Venous (Washout)\nLesion Mean: {t_ven:.1f} HU", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # 4. Arterial - Venous Subtraction (Washout Clearance)
    wash_slice = washout[:, :, slice_z].T
    im4 = axes[3].imshow(wash_slice, cmap='magma', vmin=0, vmax=120, origin='lower')
    axes[3].contour(mask_arr[:, :, slice_z].T, levels=[0.5], colors=['#00FFCC'], linewidths=1.5, origin='lower')
    axes[3].set_title(f"D. Washout Clearance Map\n(Arterial - Portal Venous ΔHU)", fontsize=12, fontweight='bold')
    axes[3].axis('off')

    cbar = fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label("Washout Clearance (ΔHU)", fontsize=10)

    fig.suptitle(
        f"Project 10 Procedural 4D Hemodynamic Kinetics: Hepatocellular Carcinoma (HCC)\n"
        f"Simulated Multi-Phase Contrast Perfusion Calibrated to LI-RADS Attenuation Dynamics",
        fontsize=14, fontweight='bold', y=0.98
    )

    out_path = os.path.join(OUT_FIG_DIR, "project10_4d_tumor_hemodynamics_showcase.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Saved updated tumor showcase figure to: {out_path}")


if __name__ == "__main__":
    generate_comparison_figure()
    generate_tumor_showcase_figure()
