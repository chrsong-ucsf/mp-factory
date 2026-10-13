#!/usr/bin/env python3
"""
code/project10/kinetics/generate_multiphase_tumors.py
=====================================================
Batch Multi-Phase Procedural Tumor Synthesis Generator for Project 10.

Stamps physiologically accurate 3D tumor lesions (HCC, PDAC, Hemangioma,
Metastasis) onto synthesized and registered CT volumes with multi-phase
contrast enhancement curves across NCCT, Arterial, and Venous phases.

Supports multi-GPU sharding across SLURM arrays (--array_id, --total_workers).
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

BASE_DIR = Path(__file__).resolve().parents[3]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from code.project10.kinetics.multiphase_tumor_kinetics import (
    ALL_PHASES,
    MultiPhaseTumorAugment,
    SUPPORTED_TUMOR_TYPES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tumor_generator")


def load_nifti_volume(filepath: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI volume and return data array (float32) and affine matrix."""
    nii = nib.load(str(filepath))
    return nii.get_fdata().astype(np.float32), nii.affine


def find_organ_mask(
    subject_id: str,
    mask_dirs: List[Path],
    target_shape: Tuple[int, int, int] = (128, 128, 128),
) -> Optional[np.ndarray]:
    """Search for TotalSegmentator or GI organ mask and resample to target_shape."""
    for mdir in mask_dirs:
        for ext in [".nii.gz", ".nii"]:
            candidate = mdir / f"{subject_id}{ext}"
            if candidate.exists():
                try:
                    img = sitk.ReadImage(str(candidate))
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetOutputSpacing((1.5, 1.5, 1.5))
                    resampler.SetSize(target_shape)
                    resampler.SetOutputOrigin(img.GetOrigin())
                    resampler.SetOutputDirection(img.GetDirection())
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler.SetDefaultPixelValue(0)
                    resampled = resampler.Execute(img)
                    arr = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)
                    return arr
                except Exception as e:
                    log.warning(f"Failed loading mask {candidate}: {e}")
    return None


def extract_host_parenchyma_mask(
    ct_volume: np.ndarray,
    organ_mask: Optional[np.ndarray],
    organ_type: str = "liver",
) -> np.ndarray:
    """
    Extract boolean host parenchyma mask. If organ mask is unavailable,
    extract abdominal soft tissue (30 to 120 HU) as an anatomical fallback.
    """
    if organ_mask is not None and np.sum(organ_mask > 0) > 500:
        return organ_mask > 0

    # Soft tissue fallback (liver/parenchymal HU window, excluding air & bone)
    soft_tissue = (ct_volume >= 20.0) & (ct_volume <= 140.0)
    # Ensure margin from edges
    pad_mask = np.zeros_like(soft_tissue, dtype=bool)
    pad_mask[16:-16, 16:-16, 16:-16] = True
    return soft_tissue & pad_mask


def main():
    parser = argparse.ArgumentParser(
        description="Project 10 Multi-Phase Tumor Kinetics Generator"
    )
    parser.add_argument(
        "--volumes_dir",
        type=str,
        default="results/project10/synthetic_volumes",
        help="Input CT volumes directory.",
    )
    parser.add_argument(
        "--mask_dirs",
        nargs="+",
        default=[
            "results/totalseg_gi_masks",
            "results/totalseg_gi_masks_bdmap",
            "results/totalseg_masks",
        ],
        help="Directories containing organ masks.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/project10/synthetic_tumors",
        help="Output directory for multi-phase lesion CT volumes and masks.",
    )
    parser.add_argument(
        "--tumor_types",
        nargs="+",
        default=["hcc", "pdac", "hemangioma", "metastasis"],
        choices=SUPPORTED_TUMOR_TYPES,
        help="Tumor types to generate.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["ncct", "arterial", "venous"],
        choices=ALL_PHASES,
        help="Contrast phases to synthesize.",
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
        "--max_scans",
        type=int,
        default=None,
        help="Max scans to process per worker.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    args = parser.parse_args()

    vols_dir = Path(args.volumes_dir)
    out_dir = Path(args.out_dir)
    mask_dirs = [Path(d) for d in args.mask_dirs]

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Tumor Kinetics Worker [{args.array_id}/{args.total_workers}] starting on device: {device}")

    # Discover candidate volumes
    all_files = sorted(list(vols_dir.glob("*.nii.gz")) + list(vols_dir.glob("*.nii")))
    if not all_files:
        log.error(f"No NIfTI volumes found in {vols_dir}")
        return

    # Shard across workers
    chunk_size = (len(all_files) + args.total_workers - 1) // args.total_workers
    start_idx = args.array_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_files))
    my_files = all_files[start_idx:end_idx]

    if args.max_scans:
        my_files = my_files[: args.max_scans]

    log.info(f"Total discovered volumes: {len(all_files)} | Assigned to worker {args.array_id}: {len(my_files)}")

    manifest_records = []
    worker_manifest = out_dir / f"tumor_manifest_worker_{args.array_id}.csv"

    for idx, ct_path in enumerate(my_files, start=1):
        filename = ct_path.name
        stem = filename.replace(".nii.gz", "").replace(".nii", "")
        # Subject identifier (e.g. CV_00000042 from CV_00000042--CV_00007115)
        subject_id = stem.split("--")[0]

        try:
            ct_vol, affine = load_nifti_volume(ct_path)
            organ_raw = find_organ_mask(subject_id, mask_dirs, target_shape=ct_vol.shape)

            for t_type in args.tumor_types:
                organ_name = "pancreas" if t_type == "pdac" else "liver"
                parenchyma_mask = extract_host_parenchyma_mask(ct_vol, organ_raw, organ_name)

                if np.sum(parenchyma_mask) < 200:
                    log.warning(f"[{stem}] Insufficient parenchyma mask voxels for {t_type}. Skipping.")
                    continue

                # Instantiate augmentor and sample geometry
                seed = args.seed + idx * 100 + args.tumor_types.index(t_type)
                aug = MultiPhaseTumorAugment(tumor_type=t_type, organ=organ_name, seed=seed)
                aug.sample_geometry(parenchyma_mask)

                tumor_mask = aug.geo_mask
                tumor_voxels = int(np.sum(tumor_mask))

                if tumor_voxels < 50:
                    continue

                # Save 3D binary ground-truth tumor mask
                mask_filename = f"{stem}_{t_type}_mask.nii.gz"
                mask_path = out_dir / mask_filename
                nib.save(nib.Nifti1Image(tumor_mask.astype(np.uint8), affine), str(mask_path))

                phase_mean_hu = {}
                # Generate and save each contrast phase
                for phase in args.phases:
                    augmented_ct = aug.apply(ct_vol, parenchyma_mask, phase=phase)
                    phase_mean_hu[phase] = float(augmented_ct[tumor_mask].mean())

                    out_filename = f"{stem}_{t_type}_{phase}.nii.gz"
                    nib.save(nib.Nifti1Image(augmented_ct.astype(np.int16), affine), str(out_dir / out_filename))

                record = {
                    "Pair_ID": stem,
                    "Subject_ID": subject_id,
                    "Tumor_Type": t_type,
                    "Organ": organ_name,
                    "Tumor_Voxels": tumor_voxels,
                    "Tumor_Volume_cc": float(tumor_voxels * (1.5**3) / 1000.0),
                    "Mask_File": mask_filename,
                }
                for phase in args.phases:
                    record[f"{phase.capitalize()}_Mean_HU"] = phase_mean_hu.get(phase, np.nan)
                manifest_records.append(record)

            if idx % 10 == 0 or idx == len(my_files):
                log.info(f"[{idx}/{len(my_files)}] Processed: {stem} (Generated {len(manifest_records)} tumor instances)")
                if manifest_records:
                    pd.DataFrame(manifest_records).to_csv(worker_manifest, index=False)

        except Exception as e:
            log.error(f"Error generating tumors for {stem}: {e}", exc_info=True)
            continue

    if manifest_records:
        df = pd.DataFrame(manifest_records)
        df.to_csv(worker_manifest, index=False)
        log.info(f"Worker {args.array_id} complete. Saved {len(df)} tumor records to {worker_manifest}")


if __name__ == "__main__":
    main()
