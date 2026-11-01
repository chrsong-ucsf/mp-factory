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
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_synthesis")

METRIC_COLUMNS = ["SSIM", "PSNR", "L1_Error", "Body_L1_Error", "RMSE", "Pearson_R", "Reference_Coverage"]
METRIC_VERSION = "v2_canonical_spatial"


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_nifti(filepath: str | Path) -> np.ndarray:
    """Load a NIfTI file and return its data array as a float32 numpy array."""
    img = nib.load(str(filepath))
    return img.get_fdata().astype(np.float32)


def resample_real_to_target(
    real_path: str | Path,
    target_ref: Union[str, Path, nib.Nifti1Image, Tuple[int, int, int]] = (128, 128, 128),
    target_shape: Optional[Tuple[int, int, int]] = None,
    spacing_mm: float = 1.5,
    min_overlap: float = 0.10,
    return_support: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Resample ground-truth registered CT volume onto the exact physical coordinate grid
    of the synthetic CT target. Checks physical overlap and rejects disjoint inputs.
    Optionally returns the binary support mask of valid reference voxels.
    """
    import nibabel.processing as nip

    real_path = Path(real_path)
    real_nii = nib.load(str(real_path))

    # If target_ref is a file path or NIfTI image, use its exact physical geometry
    if isinstance(target_ref, (str, Path)) or isinstance(target_ref, nib.Nifti1Image):
        if isinstance(target_ref, nib.Nifti1Image):
            fake_nii = target_ref
        else:
            fake_nii = nib.load(str(target_ref))

        # Check physical overlap using a unit support mask in real_nii space
        real_support = nib.Nifti1Image(np.ones(real_nii.shape[:3], dtype=np.float32), real_nii.affine)
        overlap_nii = nip.resample_from_to(real_support, fake_nii, order=0, cval=0.0)
        overlap_arr = np.asanyarray(overlap_nii.dataobj)
        total_voxels = overlap_arr.size
        overlap_voxels = int(np.count_nonzero(overlap_arr > 0.5))
        overlap_fraction = overlap_voxels / float(total_voxels) if total_voxels > 0 else 0.0

        if overlap_voxels == 0 or overlap_fraction < min_overlap:
            raise ValueError(
                f"Insufficient physical overlap between real CT ({real_path}) and target ({getattr(target_ref, 'name', target_ref)}): "
                f"{overlap_fraction:.2%} overlap (threshold: {min_overlap:.2%})"
            )

        resampled_real = nip.resample_from_to(real_nii, fake_nii, order=1, cval=-1000.0)
        vol = np.clip(np.asanyarray(resampled_real.dataobj).astype(np.float32), -1000.0, 2000.0)
        support_mask = (overlap_arr > 0.5)
        if return_support:
            return vol, support_mask
        return vol

    # Legacy fallback when target_ref is a shape tuple
    t_shape = target_shape if target_shape is not None else target_ref
    from code.project10.common_spatial import preprocess_single_volume
    vol, _ = preprocess_single_volume(
        str(real_path),
        target_shape=t_shape,
        spacing=(spacing_mm, spacing_mm, spacing_mm),
        is_mask=False,
    )
    clipped = np.clip(vol, -1000.0, 2000.0)
    if return_support:
        return clipped, np.ones(clipped.shape, dtype=bool)
    return clipped


def compute_metrics_3d(
    real_vol: np.ndarray,
    fake_vol: np.ndarray,
    support_mask: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute SSIM, PSNR, L1 (Overall MAE), Body L1 (> -500 HU), RMSE, Pearson Correlation,
    and Reference Coverage restricted strictly to valid reference support.
    """
    if real_vol.shape != fake_vol.shape:
        raise ValueError(f"Shape mismatch: {real_vol.shape} vs {fake_vol.shape}")

    if support_mask is None:
        support_mask = np.ones(real_vol.shape, dtype=bool)
    else:
        support_mask = np.asanyarray(support_mask, dtype=bool)

    if not np.any(support_mask):
        raise ValueError("support_mask has zero supported voxels; cannot compute metrics.")

    # Standard CT attenuation dynamic range [-1000, 2000] HU -> 3000 HU
    data_range = 3000.0

    # Overall L1 and L2 on supported region
    diff = (real_vol - fake_vol)[support_mask]
    l1_error = float(np.mean(np.abs(diff)))
    l2_error = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(l2_error))

    # Soft tissue / body region (> -500 HU) restricted to supported region
    body_mask = ((real_vol > -500.0) | (fake_vol > -500.0)) & support_mask
    if body_mask.any():
        body_l1 = float(np.mean(np.abs(real_vol[body_mask] - fake_vol[body_mask])))
    else:
        body_l1 = l1_error

    # Pearson correlation on supported region
    real_flat = real_vol[support_mask]
    fake_flat = fake_vol[support_mask]
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
        s_slice = support_mask[..., z]
        if not s_slice.any():
            continue

        r_slice = real_vol[..., z]
        f_slice = fake_vol[..., z]

        # PSNR on supported slice voxels
        mse_slice = float(np.mean((r_slice[s_slice] - f_slice[s_slice]) ** 2))
        if mse_slice <= 1e-12:
            slice_psnr = float("inf")
        else:
            slice_psnr = float(10 * np.log10((data_range ** 2) / mse_slice))
        psnr_vals.append(slice_psnr)

        # SSIM with valid-window bounding box
        if s_slice.all():
            slice_ssim = ssim(r_slice, f_slice, data_range=data_range)
            ssim_vals.append(slice_ssim)
        else:
            rows = np.where(s_slice.any(axis=1))[0]
            cols = np.where(s_slice.any(axis=0))[0]
            if len(rows) >= 7 and len(cols) >= 7:
                r_min, r_max = rows[0], rows[-1] + 1
                c_min, c_max = cols[0], cols[-1] + 1
                win_size = min(7, r_max - r_min, c_max - c_min)
                if win_size % 2 == 0:
                    win_size -= 1
                if win_size >= 3:
                    crop_r = r_slice[r_min:r_max, c_min:c_max]
                    crop_f = f_slice[r_min:r_max, c_min:c_max]
                    try:
                        slice_ssim = ssim(crop_r, crop_f, data_range=data_range, win_size=win_size)
                        ssim_vals.append(slice_ssim)
                    except Exception:
                        pass

    coverage = float(np.mean(support_mask))

    return {
        "L1_Error": l1_error,
        "Body_L1_Error": body_l1,
        "L2_Error": l2_error,
        "RMSE": rmse,
        "SSIM": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
        "PSNR": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
        "Pearson_R": corr,
        "Reference_Coverage": coverage,
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
    output_csv = Path(output_csv)
    if total_workers < 1:
        raise ValueError(f"total_workers must be >= 1 (got {total_workers})")

    # Build expected worker shard paths
    expected_shards = []
    for i in range(total_workers):
        shard_path = output_csv.parent / f"{output_csv.stem}_worker_{i}{output_csv.suffix}"
        if not shard_path.exists() and total_workers == 1 and output_csv.exists():
            shard_path = output_csv
        expected_shards.append(shard_path)

    # Check for missing expected shards
    missing = [p for p in expected_shards if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Cannot merge worker CSVs: missing {len(missing)} of {total_workers} expected shard(s): "
            f"{[p.name for p in missing]}"
        )

    # Check for unmanifested / obsolete shards
    all_found = sorted(output_csv.parent.glob(f"{output_csv.stem}_worker_*{output_csv.suffix}"))
    extra = [p for p in all_found if p not in expected_shards]
    if extra:
        logger.warning(
            f"Ignoring {len(extra)} unmanifested / obsolete shard(s) outside expected workers [0..{total_workers-1}]: "
            f"{[p.name for p in extra]}"
        )

    logger.info(f"Merging {len(expected_shards)} worker CSVs into {output_csv}...")
    dfs = []
    for shard in expected_shards:
        df = pd.read_csv(shard)
        if "metric_version" not in df.columns:
            raise ValueError(
                f"Shard {shard.name} is missing 'metric_version'. Stale/unversioned shards cannot be merged."
            )
        bad_versions = df[df["metric_version"] != METRIC_VERSION]
        if not bad_versions.empty:
            raise ValueError(
                f"Shard {shard.name} contains invalid/stale metric_version entries "
                f"(expected '{METRIC_VERSION}', found {bad_versions['metric_version'].unique().tolist()})."
            )
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if merged_df.empty:
        logger.info(f"All worker shards were empty; empty output CSV written to {output_csv}.")
        merged_df.to_csv(output_csv, index=False)
        return

    # Check for conflicting duplicate subjects across shards
    if "Subject" in merged_df.columns:
        dup_mask = merged_df.duplicated(subset=["Subject"], keep=False)
        if dup_mask.any():
            dup_subjects = merged_df[dup_mask]["Subject"].unique()
            for subj in dup_subjects:
                group = merged_df[merged_df["Subject"] == subj]
                # Check hash consistency
                for hash_col in ["fake_sha256", "real_sha256"]:
                    if hash_col in group.columns and group[hash_col].nunique() > 1:
                        raise ValueError(
                            f"Conflicting {hash_col} found for duplicate subject '{subj}' across shards "
                            f"({group[hash_col].tolist()}). Shards originate from conflicting/stale runs and cannot be merged."
                        )
                # Check metric consistency
                for col in METRIC_COLUMNS:
                    if col in group.columns:
                        vals = group[col].dropna()
                        if len(vals) > 1 and not np.allclose(vals, vals.iloc[0], atol=1e-4):
                            raise ValueError(
                                f"Conflicting '{col}' values ({vals.tolist()}) found for duplicate subject '{subj}' across shards. "
                                f"Shards originate from conflicting/stale runs and cannot be merged."
                            )
            logger.warning(
                f"Found {len(dup_subjects)} identical duplicate subject(s) across shards; deduplicating identical entries."
            )
            merged_df = merged_df.drop_duplicates(subset=["Subject"])

    # Verify common run_id if present
    if "run_id" in merged_df.columns:
        run_ids = merged_df["run_id"].dropna().unique()
        if len(run_ids) > 1:
            raise ValueError(
                f"Worker shards originate from conflicting runs: multiple run_ids found {run_ids.tolist()}."
            )

    merged_df = merged_df.sort_values("Subject")
    merged_df.to_csv(output_csv, index=False)

    logger.info(f"Merged CSV saved to {output_csv} ({len(merged_df)} total scans).")
    metric_cols = [c for c in METRIC_COLUMNS if c in merged_df.columns]
    if not metric_cols:
        excluded = {"Subject", "metric_version", "fake_mtime", "real_mtime", "code_version", "fake_sha256", "real_sha256", "real_file"}
        metric_cols = [c for c in merged_df.select_dtypes(include=[np.number]).columns if c not in excluded]
    summary = merged_df[metric_cols].mean()
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force recomputation of all metrics, invalidating cached CSV records.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Defaults to auto-detect.",
    )

    args = parser.parse_args()

    fake_dir = Path(args.fake_dir)
    real_dir = Path(args.real_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_worker_csvs(output_csv, args.total_workers)
        return

    if args.device is not None:
        device = torch.device(args.device)
    else:
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

    if not my_files:
        logger.info(
            f"Worker {args.array_id}: 0 files assigned (idle worker out of {args.total_workers} total workers). "
            f"Writing empty schema-bearing shard to {worker_csv}."
        )
        schema_cols = [
            "Subject", "metric_version", "fake_sha256", "real_sha256", "real_file", "fake_mtime"
        ] + METRIC_COLUMNS
        empty_df = pd.DataFrame(columns=schema_cols)
        empty_df.to_csv(worker_csv, index=False)
        return


    # Check for existing valid progress
    results = []
    processed_subjects = {}
    if worker_csv.exists() and not args.overwrite:
        try:
            prev_df = pd.read_csv(worker_csv)
            if "Subject" in prev_df.columns:
                for rec in prev_df.to_dict(orient="records"):
                    subj = str(rec.get("Subject", ""))
                    # Invalidate stale rows: must have matching metric_version and verified input identities
                    if rec.get("metric_version") == METRIC_VERSION:
                        f_cand = fake_dir / subj
                        real_cand = find_matching_real_file(f_cand, real_dir)
                        if f_cand.exists() and real_cand is not None and real_cand.exists():
                            f_sha = compute_file_sha256(f_cand)
                            r_sha = compute_file_sha256(real_cand)
                            if (
                                rec.get("fake_sha256") == f_sha
                                and rec.get("real_sha256") == r_sha
                                and rec.get("real_file") == real_cand.name
                            ):
                                processed_subjects[subj] = rec
                results = list(processed_subjects.values())
                logger.info(
                    f"Resuming worker {args.array_id}: loaded {len(results)} valid verified evaluated scans (version {METRIC_VERSION})."
                )
        except Exception as e:
            logger.warning(f"Could not read previous worker CSV: {e}")

    error_count = 0
    scored_count = len(results)

    for idx, fake_path in enumerate(my_files, start=1):
        filename = fake_path.name
        if filename in processed_subjects and not args.overwrite:
            continue

        real_path = find_matching_real_file(fake_path, real_dir)
        if real_path is None or not real_path.exists():
            logger.warning(f"Missing corresponding real file for {filename} in {real_dir}. Skipping.")
            error_count += 1
            continue

        try:
            fake_vol = load_nifti(fake_path)
            real_vol, support_mask = resample_real_to_target(real_path, target_ref=fake_path, return_support=True)

            metrics = compute_metrics_3d(real_vol, fake_vol, support_mask=support_mask, device=device)
            metrics["Subject"] = filename
            metrics["fake_sha256"] = compute_file_sha256(fake_path)
            metrics["real_sha256"] = compute_file_sha256(real_path)
            metrics["real_file"] = real_path.name
            metrics["fake_mtime"] = float(fake_path.stat().st_mtime)
            metrics["metric_version"] = METRIC_VERSION

            # Replace or append metric record
            results = [r for r in results if r.get("Subject") != filename]
            results.append(metrics)
            processed_subjects[filename] = metrics
            scored_count += 1

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
            error_count += 1
            logger.error(f"Error processing {filename}: {e}", exc_info=True)
            continue

    if not results:
        logger.warning(f"Worker {args.array_id}: No files were successfully evaluated.")
        if error_count > 0:
            raise RuntimeError(f"All {error_count} assigned cases failed evaluation on worker {args.array_id}.")
        raise RuntimeError(f"Worker {args.array_id} had {len(my_files)} assigned cases but produced zero results.")

    # Final save for this worker
    df = pd.DataFrame(results)
    cols = ["Subject"] + [c for c in df.columns if c != "Subject"]
    df[cols].to_csv(worker_csv, index=False)
    logger.info(
        f"Worker {args.array_id} complete. Assigned: {len(my_files)}, "
        f"Scored: {len(df)}, Errors: {error_count}. Saved to {worker_csv}"
    )

    if error_count > 0:
        raise RuntimeError(
            f"{error_count}/{len(my_files)} assigned cases failed evaluation on worker {args.array_id}."
        )

    # If single worker, print summary
    if args.total_workers == 1:
        metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
        if not metric_cols:
            excluded = {"Subject", "metric_version", "fake_mtime", "real_mtime", "code_version", "fake_sha256", "real_sha256", "real_file"}
            metric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
        summary = df[metric_cols].mean()
        logger.info("=== Overall Summary Metrics ===")
        for metric, value in summary.items():
            logger.info(f"Mean {metric:15s}: {value:.4f}")


if __name__ == "__main__":
    main()
