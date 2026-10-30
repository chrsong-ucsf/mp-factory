#!/usr/bin/env python3
"""
code/project10/evaluation/evaluate_synthesis.py
===============================================
GPU-Accelerated 8-Worker Parallel Evaluation of Synthesized Multi-Phase CT Volumes.

Computes SSIM, PSNR, MAE (L1), RMSE, Body L1 (soft-tissue), and Pearson Correlation
against ground-truth registered CT volumes resampled to the 1.5mm / 128^3 latent grid.
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_synthesis")


def load_nifti(filepath: str | Path) -> np.ndarray:
    """Load a NIfTI file and return its data array as a float32 numpy array."""
    img = nib.load(str(filepath))
    return img.get_fdata().astype(np.float32)


def resample_real_to_target(
    real_path: str | Path,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    spacing_mm: float = 1.5,
) -> np.ndarray:
    """
    Resample ground-truth registered CT volume to 1.5mm isotropic, pad/crop with -1000 HU air,
    matching the exact spatial grid used by the 3D latent diffusion model.
    """
    img = sitk.ReadImage(str(real_path))
    cur_spacing = img.GetSpacing()
    cur_size = img.GetSize()

    new_spacing = (spacing_mm, spacing_mm, spacing_mm)
    new_size = [
        int(round(cur_size[i] * cur_spacing[i] / new_spacing[i])) for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000.0)
    resampled = resampler.Execute(img)

    arr = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)  # (X, Y, Z)

    # Pad or crop symmetrically to target_shape with air background (-1000 HU)
    vol = np.full(target_shape, -1000.0, dtype=np.float32)
    min_x = min(arr.shape[0], target_shape[0])
    min_y = min(arr.shape[1], target_shape[1])
    min_z = min(arr.shape[2], target_shape[2])

    sx = max(0, (target_shape[0] - arr.shape[0]) // 2)
    sy = max(0, (target_shape[1] - arr.shape[1]) // 2)
    sz = max(0, (target_shape[2] - arr.shape[2]) // 2)

    cx = max(0, (arr.shape[0] - target_shape[0]) // 2)
    cy = max(0, (arr.shape[1] - target_shape[1]) // 2)
    cz = max(0, (arr.shape[2] - target_shape[2]) // 2)

    vol[sx : sx + min_x, sy : sy + min_y, sz : sz + min_z] = arr[cx : cx + min_x, cy : cy + min_y, cz : cz + min_z]

    return np.clip(vol, -1000.0, 2000.0)


def compute_metrics_3d(
    real_vol: np.ndarray,
    fake_vol: np.ndarray,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute SSIM, PSNR, L1 (Overall MAE), Body L1 (> -500 HU), RMSE, and Pearson Correlation.
    """
    if real_vol.shape != fake_vol.shape:
        raise ValueError(f"Shape mismatch: {real_vol.shape} vs {fake_vol.shape}")

    # Standard CT attenuation dynamic range [-1000, 2000] HU -> 3000 HU
    data_range = 3000.0

    # Overall L1 and L2
    diff = real_vol - fake_vol
    l1_error = float(np.mean(np.abs(diff)))
    l2_error = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(l2_error))

    # Soft tissue / body region (> -500 HU)
    body_mask = (real_vol > -500.0) | (fake_vol > -500.0)
    if body_mask.any():
        body_l1 = float(np.mean(np.abs(real_vol[body_mask] - fake_vol[body_mask])))
    else:
        body_l1 = l1_error

    # Pearson correlation
    real_flat = real_vol.flatten()
    fake_flat = fake_vol.flatten()
    std_prod = np.std(real_flat) * np.std(fake_flat)
    if std_prod > 1e-6:
        corr = float(np.corrcoef(real_flat, fake_flat)[0, 1])
    else:
        corr = 0.0

    # Slice-by-slice SSIM & PSNR across axial dimension Z
    ssim_vals = []
    psnr_vals = []
    z_dim = real_vol.shape[-1]
    for z in range(z_dim):
        r_slice = real_vol[..., z]
        f_slice = fake_vol[..., z]

        slice_ssim = ssim(r_slice, f_slice, data_range=data_range)
        slice_psnr = psnr(r_slice, f_slice, data_range=data_range)
        ssim_vals.append(slice_ssim)
        psnr_vals.append(slice_psnr)

    return {
        "L1_Error": l1_error,
        "Body_L1_Error": body_l1,
        "L2_Error": l2_error,
        "RMSE": rmse,
        "SSIM": float(np.mean(ssim_vals)),
        "PSNR": float(np.mean(psnr_vals)),
        "Pearson_R": corr,
    }


def find_matching_real_file(fake_path: Path, real_dir: Path) -> Optional[Path]:
    """Find matching registered ground truth file in real_dir."""
    filename = fake_path.name
    # Direct match
    cand = real_dir / filename
    if cand.exists():
        return cand

    # Map *_synthetic_contrast.nii.gz to *_reg.nii.gz
    stem = filename.replace("_synthetic_contrast.nii.gz", "").replace("_synthetic_contrast.nii", "")
    for ext in [".nii.gz", ".nii"]:
        cand = real_dir / f"{stem}_reg{ext}"
        if cand.exists():
            return cand

    # Map *_reg.nii.gz directly
    for ext in [".nii.gz", ".nii"]:
        cand = real_dir / f"{stem}{ext}"
        if cand.exists():
            return cand

    return None


def merge_worker_csvs(output_csv: Path, total_workers: int):
    """Combine individual worker CSV files into a unified report."""
    csv_pattern = f"{output_csv.parent}/{output_csv.stem}_worker_*{output_csv.suffix}"
    worker_csvs = sorted(glob.glob(csv_pattern))
    if not worker_csvs:
        logger.warning(f"No worker CSVs found matching pattern: {csv_pattern}")
        return

    logger.info(f"Merging {len(worker_csvs)} worker CSVs into {output_csv}...")
    dfs = [pd.read_csv(p) for p in worker_csvs]
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["Subject"]).sort_values("Subject")
    merged_df.to_csv(output_csv, index=False)

    logger.info(f"Merged CSV saved to {output_csv} ({len(merged_df)} total scans).")
    numeric_cols = [c for c in merged_df.columns if c != "Subject"]
    summary = merged_df[numeric_cols].mean()
    logger.info("================== Unified Summary Metrics ==================")
    for metric, value in summary.items():
        logger.info(f"  Mean {metric:15s}: {value:.4f}")
    logger.info("=============================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate synthesized multi-phase CTs against ground truth."
    )
    parser.add_argument(
        "--fake_dir",
        type=str,
        default="results/project10/synthetic_volumes",
        help="Directory containing synthesized (fake) CT NIfTI files.",
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        default="results/project10/registered_volumes",
        help="Directory containing ground truth (real) CT NIfTI files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/project10/evaluation_reports/synthesis_metrics.csv",
        help="Path to save the output CSV report.",
    )
    parser.add_argument(
        "--array_id",
        type=int,
        default=0,
        help="SLURM array task ID (0-indexed worker).",
    )
    parser.add_argument(
        "--total_workers",
        type=int,
        default=1,
        help="Total number of parallel workers.",
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Merge existing worker CSVs and exit.",
    )

    args = parser.parse_args()

    fake_dir = Path(args.fake_dir)
    real_dir = Path(args.real_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_worker_csvs(output_csv, args.total_workers)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Evaluation Worker [{args.array_id}/{args.total_workers}] starting on device: {device}"
    )

    if not fake_dir.is_dir():
        logger.error(f"Fake directory does not exist: {fake_dir}")
        return

    if not real_dir.is_dir():
        logger.error(f"Real directory does not exist: {real_dir}")
        return

    # Discover fake files
    fake_files = sorted(
        list(fake_dir.glob("*_synthetic_contrast.nii.gz"))
        + list(fake_dir.glob("*.nii.gz"))
        + list(fake_dir.glob("*.nii"))
    )
    fake_files = sorted(list(set(fake_files)))

    if not fake_files:
        logger.warning(f"No NIfTI files found in {fake_dir}")
        return

    total_discovered = len(fake_files)
    chunk_size = (total_discovered + args.total_workers - 1) // args.total_workers
    start_idx = args.array_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_discovered)
    my_files = fake_files[start_idx:end_idx]

    logger.info(
        f"Total synthetic volumes: {total_discovered} | Assigned to worker {args.array_id}: {len(my_files)}"
    )

    # Worker output CSV
    if args.total_workers > 1:
        worker_csv = output_csv.parent / f"{output_csv.stem}_worker_{args.array_id}{output_csv.suffix}"
    else:
        worker_csv = output_csv

    # Check for existing partial progress
    results = []
    processed_subjects = set()
    if worker_csv.exists():
        try:
            prev_df = pd.read_csv(worker_csv)
            if "Subject" in prev_df.columns:
                results = prev_df.to_dict(orient="records")
                processed_subjects = set(prev_df["Subject"].tolist())
                logger.info(f"Resuming worker {args.array_id}: loaded {len(results)} previously evaluated scans.")
        except Exception as e:
            logger.warning(f"Could not read previous worker CSV: {e}")

    for idx, fake_path in enumerate(my_files, start=1):
        filename = fake_path.name
        if filename in processed_subjects:
            continue

        real_path = find_matching_real_file(fake_path, real_dir)
        if real_path is None or not real_path.exists():
            logger.warning(f"Missing corresponding real file for {filename} in {real_dir}. Skipping.")
            continue

        try:
            fake_vol = load_nifti(fake_path)
            real_vol = resample_real_to_target(real_path, target_shape=fake_vol.shape)

            metrics = compute_metrics_3d(real_vol, fake_vol, device=device)
            metrics["Subject"] = filename
            results.append(metrics)
            processed_subjects.add(filename)

            logger.info(
                f"[{idx}/{len(my_files)}] {filename} -> SSIM: {metrics['SSIM']:.4f} | "
                f"PSNR: {metrics['PSNR']:.2f} dB | L1: {metrics['L1_Error']:.1f} HU | "
                f"Body L1: {metrics['Body_L1_Error']:.1f} HU | Corr: {metrics['Pearson_R']:.3f}"
            )

            # Incremental save every 10 cases
            if len(results) % 10 == 0:
                df = pd.DataFrame(results)
                cols = ["Subject"] + [c for c in df.columns if c != "Subject"]
                df[cols].to_csv(worker_csv, index=False)

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}", exc_info=True)
            continue

    if not results:
        logger.warning(f"Worker {args.array_id}: No files were successfully evaluated.")
        return

    # Final save for this worker
    df = pd.DataFrame(results)
    cols = ["Subject"] + [c for c in df.columns if c != "Subject"]
    df[cols].to_csv(worker_csv, index=False)
    logger.info(f"Worker {args.array_id} complete. Saved {len(df)} records to {worker_csv}")

    # If single worker, print summary
    if args.total_workers == 1:
        summary = df.drop(columns=["Subject"]).mean()
        logger.info("=== Overall Summary Metrics ===")
        for metric, value in summary.items():
            logger.info(f"Mean {metric:15s}: {value:.4f}")


if __name__ == "__main__":
    main()
