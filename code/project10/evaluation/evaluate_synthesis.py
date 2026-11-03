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
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
METRIC_VERSION = "v4_audited_support"


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hash of a file with streaming 1MB blocks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1048576):
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

        # Check physical overlap using observed anatomy support mask (F05)
        supp_path = str(real_path).replace(".nii.gz", "_support.nii.gz")
        if os.path.isfile(supp_path):
            supp_nii = nib.load(supp_path)
            real_support = nib.Nifti1Image(np.asanyarray(supp_nii.dataobj).astype(np.float32), supp_nii.affine)
        else:
            # When explicit support mask is absent, exclude -1000 HU registration padding
            real_arr = np.asanyarray(real_nii.dataobj)
            real_support_data = (real_arr > -999.0).astype(np.float32)
            real_support = nib.Nifti1Image(real_support_data, real_nii.affine)

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

        # SSIM strictly evaluated on fully supported window footprints
        win_size = 7
        if min(r_slice.shape[:2]) >= win_size:
            from scipy.ndimage import binary_erosion
            # Compute full 2D local SSIM map
            _, ssim_map = ssim(r_slice, f_slice, data_range=data_range, win_size=win_size, full=True)
            # A pixel (r, c) is the center of a fully-supported win_size x win_size window
            # if and only if all pixels in that window have s_slice == True
            valid_win = binary_erosion(s_slice, structure=np.ones((win_size, win_size), dtype=bool))
            if valid_win.any():
                slice_ssim = float(np.mean(ssim_map[valid_win]))
                ssim_vals.append(slice_ssim)

    coverage = float(np.mean(support_mask))

    return {
        "L1_Error": l1_error,
        "Body_L1_Error": body_l1,
        "L2_Error": l2_error,
        "RMSE": rmse,
        "SSIM": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
        "PSNR": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
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

    # Map *_synthetic_contrast.nii.gz to *_reg.nii.gz, or general {stem}_reg.nii.gz
    stem = filename
    for sfx in ["_synthetic_contrast.nii.gz", "_synthetic_contrast.nii", "_reg.nii.gz", "_reg.nii", ".nii.gz", ".nii"]:
        if stem.endswith(sfx):
            stem = stem[:-len(sfx)]
            break

    for ext in [".nii.gz", ".nii"]:
        cand = real_dir / f"{stem}_reg{ext}"
        if cand.exists():
            return cand

    for ext in [".nii.gz", ".nii"]:
        cand = real_dir / f"{stem}{ext}"
        if cand.exists():
            return cand

    return None


def build_evaluation_cohort_manifest(
    fake_dir: Path,
    real_dir: Path,
    fake_files: List[Path],
    total_workers: int,
    metric_version: str = METRIC_VERSION,
) -> Dict[str, Any]:
    """
    Builds an immutable manifest of all input pairs and worker assignments,
    capturing file content SHA-256 hashes so any change to input volumes
    produces a distinct cohort snapshot identity.
    """
    total_discovered = len(fake_files)
    chunk_size = (total_discovered + total_workers - 1) // total_workers if total_workers > 0 else total_discovered

    items = []
    sig_builder = hashlib.sha256()
    sig_builder.update(f"{metric_version}:{total_discovered}:{total_workers}:".encode("utf-8"))

    for idx, f in enumerate(sorted(fake_files)):
        fake_sha = compute_file_sha256(f)
        real_cand = find_matching_real_file(f, real_dir)
        real_sha = compute_file_sha256(real_cand) if (real_cand and real_cand.exists()) else None
        supp_cand = Path(str(real_cand).replace(".nii.gz", "_support.nii.gz")) if real_cand else None
        supp_sha = compute_file_sha256(supp_cand) if (supp_cand and supp_cand.exists()) else None
        worker_id = idx // chunk_size if chunk_size > 0 else 0

        item = {
            "fake_file": f.name,
            "fake_sha256": fake_sha,
            "real_file": real_cand.name if real_cand else None,
            "real_sha256": real_sha,
            "support_file": supp_cand.name if (supp_cand and supp_cand.exists()) else None,
            "support_sha256": supp_sha,
            "worker_id": worker_id,
        }
        items.append(item)
        sig_builder.update(
            f"{f.name}:{fake_sha}:{item['real_file']}:{real_sha}:{item['support_file']}:{supp_sha}:{worker_id};".encode("utf-8")
        )

    snapshot_hash = sig_builder.hexdigest()
    run_id = f"run_{snapshot_hash[:16]}"

    manifest = {
        "run_id": run_id,
        "metric_version": metric_version,
        "fake_dir": str(fake_dir.resolve()),
        "real_dir": str(real_dir.resolve()),
        "total_files": total_discovered,
        "total_workers": total_workers,
        "items": items,
    }
    return manifest


def merge_worker_csvs(
    output_csv: Path,
    total_workers: int,
    expected_run_id: Optional[str] = None,
    fake_dir: Optional[Union[str, Path]] = None,
    real_dir: Optional[Union[str, Path]] = None,
    allow_partial: bool = False,
):
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

    # Attempt to load cohort manifest if present
    manifest_path = output_csv.parent / f"{output_csv.stem}_manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            if fake_dir is None and "fake_dir" in manifest:
                fake_dir = Path(manifest["fake_dir"])
            if real_dir is None and "real_dir" in manifest:
                real_dir = Path(manifest["real_dir"])
            if expected_run_id is None and "run_id" in manifest:
                expected_run_id = str(manifest["run_id"]).strip()
        except Exception as e:
            logger.warning(f"Could not load manifest {manifest_path}: {e}")

    logger.info(f"Merging {len(expected_shards)} worker CSVs into {output_csv}...")
    dfs = []
    shard_metas = []
    shard_runs = []
    for i, shard in enumerate(expected_shards):
        df = pd.read_csv(shard)

        # Check worker completion metadata JSON if present
        meta_file = shard.with_suffix(".meta.json")
        if not meta_file.exists():
            alt_meta = shard.with_suffix(".json")
            if alt_meta.exists():
                meta_file = alt_meta

        meta = None
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                meta_run = str(meta.get("run_id", "")).strip()
                if not meta_run:
                    raise ValueError(f"Shard {shard.name} metadata contains empty or blank 'run_id'.")
                if expected_run_id is not None and meta_run != expected_run_id:
                    raise ValueError(
                        f"Shard {shard.name} metadata run_id '{meta_run}' does not match expected run_id '{expected_run_id}'."
                    )
                shard_runs.append(meta_run)
            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                logger.warning(f"Could not load metadata for shard {shard.name}: {e}")
        shard_metas.append(meta)

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
        if "run_id" not in df.columns:
            raise ValueError(
                f"Shard {shard.name} is missing required 'run_id' provenance. Unidentified shards cannot be merged."
            )
        if "Reference_Coverage" not in df.columns:
            raise ValueError(
                f"Shard {shard.name} is missing required 'Reference_Coverage' column (version '{METRIC_VERSION}' required)."
            )

        # Strict data row provenance check: every row must have a non-empty, non-whitespace run_id
        if not df.empty:
            blank_mask = df["run_id"].isna() | (df["run_id"].astype(str).str.strip() == "")
            if blank_mask.any():
                bad_count = int(blank_mask.sum())
                raise ValueError(
                    f"Shard {shard.name} contains {bad_count} row(s) with blank/missing 'run_id' provenance. "
                    "Every data record must have a valid non-empty run identity."
                )
            row_runs = df["run_id"].astype(str).str.strip().unique()
            if len(row_runs) != 1:
                raise ValueError(
                    f"Shard {shard.name} contains conflicting run IDs within shard: {row_runs.tolist()}."
                )
            shard_run = row_runs[0]
            if expected_run_id is not None and shard_run != expected_run_id:
                raise ValueError(
                    f"Shard {shard.name} run_id '{shard_run}' does not match expected run_id '{expected_run_id}'."
                )
            if meta is not None and meta.get("run_id") and str(meta["run_id"]).strip() != shard_run:
                raise ValueError(
                    f"Shard {shard.name} data run_id '{shard_run}' does not match worker metadata run_id '{meta['run_id']}'."
                )
            shard_runs.append(shard_run)

        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # Enforce cohort completion proof (manifest, worker metadata sidecars, and coverage)
    if not allow_partial:
        if manifest is None:
            raise FileNotFoundError(
                f"Cannot perform verified merge: missing required cohort manifest '{manifest_path.name}'. "
                "Verified default merging requires manifest proof of the complete cohort; specify allow_partial=True to allow unverified/partial export."
            )
        if manifest.get("total_workers") is not None and manifest["total_workers"] != total_workers:
            raise ValueError(
                f"Merge total_workers ({total_workers}) does not match manifest total_workers ({manifest['total_workers']}). "
                f"The run was partitioned for {manifest['total_workers']} workers; re-partition or merge with total_workers={manifest['total_workers']}."
            )
        for i, shard in enumerate(expected_shards):
            meta = shard_metas[i]
            if meta is None:
                meta_file = shard.with_suffix(".meta.json")
                raise FileNotFoundError(
                    f"Cannot perform verified merge: missing required completion metadata sidecar '{meta_file.name}' for shard '{shard.name}'. "
                    "Verified default merging requires worker completion proof; specify allow_partial=True to allow unverified/partial export."
                )
            meta_status = meta.get("status", "")
            meta_errors = int(meta.get("error_count", 0))
            meta_total = meta.get("total_workers")
            if meta_total is not None and meta_total != total_workers:
                raise ValueError(
                    f"Shard {shard.name} metadata total_workers ({meta_total}) does not match merge total_workers ({total_workers})."
                )
            num_assigned = meta.get("num_assigned")
            num_scored = meta.get("num_scored")
            if meta_status != "completed":
                raise ValueError(
                    f"Shard {shard.name} has non-completed status '{meta_status}' (recorded {meta_errors} error(s)). "
                    "Worker run failed or is incomplete; cannot publish partial cohort without allow_partial=True."
                )
            if meta_errors > 0:
                raise ValueError(
                    f"Shard {shard.name} recorded {meta_errors} evaluation error(s); cannot publish partial cohort."
                )
            if num_assigned is not None and num_scored is not None and num_scored < num_assigned:
                raise ValueError(
                    f"Shard {shard.name} evaluated only {num_scored} of {num_assigned} assigned cases; cannot publish partial cohort."
                )
            if num_assigned is not None and num_scored is not None and num_scored > num_assigned:
                raise ValueError(
                    f"Shard {shard.name} scored {num_scored} cases but was assigned only {num_assigned}; unpartitioned/stale records detected."
                )
            worker_expected_items = [it for it in manifest.get("items", []) if it.get("worker_id") == i]
            if num_assigned is not None and num_assigned != len(worker_expected_items):
                raise ValueError(
                    f"Worker {i} metadata num_assigned ({num_assigned}) contradicts manifest assignment count ({len(worker_expected_items)})."
                )
            if num_scored is not None and num_scored != len(worker_expected_items):
                raise ValueError(
                    f"Worker {i} metadata num_scored ({num_scored}) contradicts manifest assignment count ({len(worker_expected_items)})."
                )

    # Revalidate input directories
    if fake_dir is not None or real_dir is not None:
        if fake_dir is None or not Path(fake_dir).is_dir():
            raise FileNotFoundError(
                f"Input verification failed: fake directory '{fake_dir}' does not exist or is not a directory."
            )
        if real_dir is None or not Path(real_dir).is_dir():
            raise FileNotFoundError(
                f"Input verification failed: real directory '{real_dir}' does not exist or is not a directory."
            )

    # Enforce complete manifest cohort coverage unless partial merge is explicitly authorized
    if manifest is not None and not allow_partial:
        expected_items = manifest.get("items", [])
        if expected_items:
            expected_subjects = {item["fake_file"] for item in expected_items}
            merged_subjects = set(merged_df["Subject"]) if ("Subject" in merged_df.columns and not merged_df.empty) else set()
            missing_subjects = expected_subjects - merged_subjects
            if missing_subjects:
                raise ValueError(
                    f"Incomplete cohort: merged dataset contains {len(merged_subjects)} of {len(expected_subjects)} "
                    f"expected subjects from manifest (missing {len(missing_subjects)} cases: {sorted(list(missing_subjects))[:5]}). "
                    "Failed jobs cannot become an apparently complete benchmark through merge."
                )

            # Check per-worker assignment coverage
            for i, shard in enumerate(expected_shards):
                worker_expected = {item["fake_file"] for item in expected_items if item.get("worker_id") == i}
                shard_df = dfs[i]
                shard_subjects = set(shard_df["Subject"]) if ("Subject" in shard_df.columns and not shard_df.empty) else set()
                missing_from_worker = worker_expected - shard_subjects
                if missing_from_worker:
                    raise ValueError(
                        f"Worker {i} shard {shard.name} is missing {len(missing_from_worker)} assigned cases: "
                        f"{sorted(list(missing_from_worker))[:5]}..."
                    )
                unexpected_in_worker = shard_subjects - worker_expected
                if unexpected_in_worker:
                    raise ValueError(
                        f"Worker {i} shard {shard.name} contains {len(unexpected_in_worker)} unexpected cases not assigned to worker {i}: "
                        f"{sorted(list(unexpected_in_worker))[:5]}..."
                    )

    # Check for conflicting duplicate subjects across shards
    if "Subject" in merged_df.columns and not merged_df.empty:
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
                for m_col in METRIC_COLUMNS:
                    if m_col in group.columns and group[m_col].nunique() > 1:
                        unique_vals = group[m_col].dropna().unique()
                        if len(unique_vals) > 1 and not np.allclose(unique_vals, unique_vals[0], atol=1e-5):
                            raise ValueError(
                                f"Conflicting duplicate evaluation for subject '{subj}': differing {m_col} scores "
                                f"{unique_vals.tolist()} across worker shards. Merging conflicting evaluations is prohibited."
                            )
            if not allow_partial:
                raise ValueError(
                    f"Duplicate subject(s) found across shards: {sorted(list(dup_subjects))[:5]}. "
                    "Each subject must appear in exactly one worker shard."
                )
            logger.warning(
                f"Found {len(dup_subjects)} duplicate subject(s) across shards; deduplicating identical entries."
            )
            merged_df = merged_df.drop_duplicates(subset=["Subject"])

    if merged_df.empty:
        logger.info(f"All worker shards were empty; empty output CSV written to {output_csv}.")
        merged_df.to_csv(output_csv, index=False)
        return

    # Enforce non-blank run_id across all merged rows
    blank_merged = merged_df["run_id"].isna() | (merged_df["run_id"].astype(str).str.strip() == "")
    if blank_merged.any():
        bad_count = int(blank_merged.sum())
        raise ValueError(
            f"Merged dataset contains {bad_count} row(s) with blank/missing 'run_id' provenance."
        )

    # Verify single consistent run_id across all shards
    all_run_ids = merged_df["run_id"].astype(str).str.strip().unique()
    if len(all_run_ids) != 1:
        raise ValueError(
            f"Worker shards originate from conflicting runs: found multiple or zero run_ids {all_run_ids.tolist()}. "
            f"All shards must share a single verified run identity to be merged."
        )
    merged_run = all_run_ids[0]
    if expected_run_id is not None and merged_run != expected_run_id:
        raise ValueError(
            f"Merged run_id '{merged_run}' does not match expected run_id '{expected_run_id}'."
        )
    if manifest is not None and manifest.get("run_id") and manifest["run_id"] != merged_run:
        raise ValueError(
            f"Merged run_id '{merged_run}' does not match input manifest run_id '{manifest['run_id']}'. "
            "Evaluation shards originate from an outdated run manifest; re-evaluation is required."
        )

    # Revalidate merged rows against current inputs on disk
    if fake_dir is not None and real_dir is not None:
        f_dir = Path(fake_dir)
        r_dir = Path(real_dir)
        for _, row in merged_df.iterrows():
            subj = str(row["Subject"])
            f_cand = f_dir / subj
            r_cand = find_matching_real_file(f_cand, r_dir)
            if not f_cand.exists():
                raise ValueError(
                    f"Input synthetic volume '{f_cand}' no longer exists on disk; cannot merge stale row for '{subj}'."
                )
            if r_cand is None or not r_cand.exists():
                raise ValueError(
                    f"Input real reference volume for '{subj}' no longer exists in {r_dir}; cannot merge stale row."
                )
            if "fake_sha256" in row and pd.notna(row["fake_sha256"]):
                curr_f_sha = compute_file_sha256(f_cand)
                if str(row["fake_sha256"]) != curr_f_sha:
                    raise ValueError(
                        f"Row for subject '{subj}' has stale fake_sha256 ({row['fake_sha256']} != current {curr_f_sha}). "
                        "Input synthetic file contents have changed; re-evaluation is required."
                    )
            if "real_sha256" in row and pd.notna(row["real_sha256"]):
                curr_r_sha = compute_file_sha256(r_cand)
                if str(row["real_sha256"]) != curr_r_sha:
                    raise ValueError(
                        f"Row for subject '{subj}' has stale real_sha256 ({row['real_sha256']} != current {curr_r_sha}). "
                        "Input reference file contents have changed; re-evaluation is required."
                    )
            supp_cand = Path(str(r_cand).replace(".nii.gz", "_support.nii.gz"))
            if supp_cand.exists() and "support_sha256" in row and pd.notna(row["support_sha256"]):
                curr_s_sha = compute_file_sha256(supp_cand)
                if str(row["support_sha256"]) != curr_s_sha:
                    raise ValueError(
                        f"Row for subject '{subj}' has stale support_sha256 ({row['support_sha256']} != current {curr_s_sha}). "
                        "Input support file contents have changed; re-evaluation is required."
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
        default=None,
        help="Directory containing synthesized (fake) CT NIfTI files. Defaults to results/project10/synthetic_volumes during evaluation.",
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        default=None,
        help="Directory containing ground truth (real) CT NIfTI files. Defaults to results/project10/registered_volumes during evaluation.",
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
        "--run_id",
        type=str,
        default=None,
        help="Run identifier. If omitted, a deterministic cohort run ID is generated.",
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Merge existing worker CSVs and exit.",
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Allow merging worker CSVs even if some workers failed or cohort coverage is incomplete.",
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

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_worker_csvs(
            output_csv,
            args.total_workers,
            expected_run_id=args.run_id,
            fake_dir=args.fake_dir,
            real_dir=args.real_dir,
            allow_partial=args.allow_partial,
        )
        return

    fake_dir = Path(args.fake_dir) if args.fake_dir is not None else Path("results/project10/synthetic_volumes")
    real_dir = Path(args.real_dir) if args.real_dir is not None else Path("results/project10/registered_volumes")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Evaluation Worker [{args.array_id}/{args.total_workers}] starting on device: {device}"
    )

    if not fake_dir.is_dir():
        logger.error(f"Fake directory does not exist: {fake_dir}")
        raise FileNotFoundError(
            f"Input verification failed: fake directory '{fake_dir}' does not exist or is not a directory."
        )

    if not real_dir.is_dir():
        logger.error(f"Real directory does not exist: {real_dir}")
        raise FileNotFoundError(
            f"Input verification failed: real directory '{real_dir}' does not exist or is not a directory."
        )

    # Discover fake files
    fake_files = sorted(
        list(fake_dir.glob("*_synthetic_contrast.nii.gz"))
        + list(fake_dir.glob("*.nii.gz"))
        + list(fake_dir.glob("*.nii"))
    )
    fake_files = sorted(list(set(fake_files)))

    if not fake_files:
        logger.error(f"No NIfTI files found in {fake_dir}")
        raise FileNotFoundError(
            f"Input verification failed: no NIfTI files found in fake directory '{fake_dir}'."
        )

    total_discovered = len(fake_files)
    manifest = build_evaluation_cohort_manifest(
        fake_dir=fake_dir,
        real_dir=real_dir,
        fake_files=fake_files,
        total_workers=args.total_workers,
        metric_version=METRIC_VERSION,
    )
    if args.run_id:
        run_id = str(args.run_id).strip()
        manifest["run_id"] = run_id
    else:
        run_id = manifest["run_id"]

    manifest_path = output_csv.parent / f"{output_csv.stem}_manifest.json"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    except Exception as e:
        logger.warning(f"Could not write manifest JSON to {manifest_path}: {e}")

    chunk_size = (total_discovered + args.total_workers - 1) // args.total_workers
    start_idx = args.array_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_discovered)
    my_files = fake_files[start_idx:end_idx]

    logger.info(
        f"Total synthetic volumes: {total_discovered} | Assigned to worker {args.array_id}: {len(my_files)} | run_id: {run_id}"
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
            "Subject", "metric_version", "run_id", "fake_sha256", "real_sha256", "real_file", "fake_mtime"
        ] + METRIC_COLUMNS
        empty_df = pd.DataFrame(columns=schema_cols)
        empty_df.to_csv(worker_csv, index=False)
        meta_path = worker_csv.with_suffix(".meta.json")
        try:
            meta_data = {
                "worker_id": args.array_id,
                "total_workers": args.total_workers,
                "run_id": run_id,
                "metric_version": METRIC_VERSION,
                "num_assigned": 0,
                "num_scored": 0,
                "error_count": 0,
                "status": "completed",
            }
            meta_path.write_text(json.dumps(meta_data, indent=2))
        except Exception as e:
            logger.warning(f"Could not write idle worker metadata: {e}")
        return


    # Check for existing valid progress
    results = []
    processed_subjects = {}
    if worker_csv.exists() and not args.overwrite:
        try:
            prev_df = pd.read_csv(worker_csv)
            if "Subject" in prev_df.columns:
                assigned_names = {f.name for f in my_files}
                for rec in prev_df.to_dict(orient="records"):
                    subj = str(rec.get("Subject", ""))
                    if subj not in assigned_names:
                        continue
                    # Invalidate stale rows: must have matching metric_version, valid Reference_Coverage, and matching run_id
                    rec_run = str(rec.get("run_id", "")).strip()
                    if (
                        rec.get("metric_version") == METRIC_VERSION
                        and "Reference_Coverage" in rec
                        and not pd.isna(rec.get("Reference_Coverage"))
                        and rec_run == run_id
                    ):
                        f_cand = fake_dir / subj
                        real_cand = find_matching_real_file(f_cand, real_dir)
                        if f_cand.exists() and real_cand is not None and real_cand.exists():
                            f_sha = compute_file_sha256(f_cand)
                            r_sha = compute_file_sha256(real_cand)
                            supp_cand = Path(str(real_cand).replace(".nii.gz", "_support.nii.gz"))
                            supp_sha = compute_file_sha256(supp_cand) if supp_cand.exists() else None
                            if (
                                rec.get("fake_sha256") == f_sha
                                and rec.get("real_sha256") == r_sha
                                and rec.get("real_file") == real_cand.name
                                and rec.get("support_sha256") == supp_sha
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
            metrics["run_id"] = run_id
            metrics["fake_sha256"] = compute_file_sha256(fake_path)
            metrics["real_sha256"] = compute_file_sha256(real_path)
            metrics["real_file"] = real_path.name
            supp_p = Path(str(real_path).replace(".nii.gz", "_support.nii.gz"))
            metrics["support_sha256"] = compute_file_sha256(supp_p) if supp_p.exists() else None
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
    meta_path = worker_csv.with_suffix(".meta.json")
    try:
        meta_data = {
            "worker_id": args.array_id,
            "total_workers": args.total_workers,
            "run_id": run_id,
            "metric_version": METRIC_VERSION,
            "num_assigned": len(my_files),
            "num_scored": len(df),
            "error_count": error_count,
            "status": "completed" if error_count == 0 else "failed",
        }
        meta_path.write_text(json.dumps(meta_data, indent=2))
    except Exception as e:
        logger.warning(f"Could not write worker metadata: {e}")
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
