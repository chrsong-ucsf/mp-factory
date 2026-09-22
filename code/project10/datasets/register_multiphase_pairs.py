#!/usr/bin/env python3
"""
code/project10/datasets/register_multiphase_pairs.py
=====================================================
Phase 1 — Step 1.2: 3D Deformable Inter-Phase Co-Registration Worker

For each (P0_NCCT, P_target) pair from multiphase_pairs.csv, performs:
  1. Rigid Euler3DTransform   — bed/patient translation correction
  2. Affine                   — global organ bounding-box alignment
  3. B-Spline deformable      — soft-tissue non-rigid correction using
                                Mattes Mutual Information metric

Saves:
  results/project10/registered_volumes/<pair_id>_<subject_target>_reg.nii.gz
  results/project10/registered_volumes/<pair_id>_<subject_target>_dvf.nii.gz
  results/project10/manifests/registration_log.csv

Designed for SLURM array execution:
    python register_multiphase_pairs.py --chunk_idx $SLURM_ARRAY_TASK_ID --total_chunks $N

Usage (standalone):
    python code/project10/datasets/register_multiphase_pairs.py \
        --pairs results/project10/manifests/multiphase_pairs.csv \
        --out_dir results/project10/registered_volumes \
        [--chunk_idx 0] [--total_chunks 1]
        [--skip_existing]           # skip pairs already registered
        [--nthreads 8]              # SimpleITK multi-threading
"""

import os
import sys
import csv
import time
import argparse
import logging
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    print("[FATAL] SimpleITK not installed. Run: pip install SimpleITK")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Project 10 Step 1.2: 3D Deformable Co-Registration of Multi-Phase CT Pairs"
    )
    p.add_argument(
        "--pairs",
        type=str,
        default="results/project10/manifests/multiphase_pairs.csv",
        help="Path to multiphase_pairs.csv (output of discover_multiphase_cohort.py)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="results/project10/registered_volumes",
        help="Output directory for registered volumes and DVFs",
    )
    p.add_argument(
        "--log_csv",
        type=str,
        default="results/project10/manifests/registration_log.csv",
        help="CSV log file for registration status per pair",
    )
    p.add_argument(
        "--chunk_idx",
        type=int,
        default=0,
        help="SLURM array task index (0-based)",
    )
    p.add_argument(
        "--total_chunks",
        type=int,
        default=1,
        help="Total number of SLURM array tasks",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip pairs where output registered volume already exists",
    )
    p.add_argument(
        "--nthreads",
        type=int,
        default=8,
        help="Number of SimpleITK processing threads",
    )
    p.add_argument(
        "--same_day_only",
        action="store_true",
        help="Only process same-day pairs (minimal respiratory motion)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# SimpleITK Registration Pipeline
# ---------------------------------------------------------------------------

def load_sitk_image(path: str) -> sitk.Image:
    """Load NIfTI volume and cast to float32."""
    img = sitk.ReadImage(path, sitk.sitkFloat32)
    return img


def resample_to_fixed(
    moving: sitk.Image, fixed: sitk.Image, transform: sitk.Transform
) -> sitk.Image:
    """Apply a transform and resample moving image to fixed grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1000.0)  # Air HU for out-of-bounds
    resampler.SetTransform(transform)
    return resampler.Execute(moving)


def rigid_registration(
    fixed: sitk.Image, moving: sitk.Image
) -> sitk.Euler3DTransform:
    """Stage 1: Multi-resolution rigid registration (bones, bed)."""
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial_transform, inPlace=False)
    rigid_transform = reg.Execute(
        sitk.Normalize(fixed), sitk.Normalize(moving)
    )
    return rigid_transform


def affine_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    initial_transform: sitk.Transform,
) -> sitk.AffineTransform:
    """Stage 2: Affine registration for global organ alignment."""
    affine_init = sitk.AffineTransform(3)
    affine_init.SetMatrix(initial_transform.GetMatrix() if hasattr(initial_transform, 'GetMatrix') else sitk.AffineTransform(3).GetMatrix())
    affine_init = sitk.CompositeTransform(3)
    affine_init.AddTransform(initial_transform)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.5, numberOfIterations=150,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    affine_xfm = sitk.AffineTransform(3)
    reg.SetInitialTransform(affine_xfm, inPlace=True)
    reg.SetMovingInitialTransform(affine_init)
    affine_result = reg.Execute(
        sitk.Normalize(fixed), sitk.Normalize(moving)
    )
    # Build composite: affine ∘ rigid
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(initial_transform)
    composite.AddTransform(affine_result)
    return composite


def bspline_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    initial_transform: sitk.Transform,
    grid_spacing_mm: float = 50.0,
) -> tuple:
    """
    Stage 3: B-Spline deformable registration for soft-tissue alignment.
    Returns (final_transform, displacement_field_image).
    """
    # Set B-spline control grid
    transform_domain_mesh_size = [
        max(1, int(fixed.GetSize()[i] * fixed.GetSpacing()[i] / grid_spacing_mm))
        for i in range(3)
    ]
    bspline_transform = sitk.BSplineTransformInitializer(
        fixed, transform_domain_mesh_size, order=3
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGS2(
        solutionAccuracy=1e-2,
        numberOfIterations=150,
        deltaConvergenceTolerance=1e-3,
    )
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(bspline_transform, inPlace=True)
    reg.SetMovingInitialTransform(initial_transform)

    final_transform = reg.Execute(
        sitk.Normalize(fixed), sitk.Normalize(moving)
    )

    # Compose into full displacement field
    composite = sitk.CompositeTransform(3)
    composite.AddTransform(initial_transform)
    composite.AddTransform(final_transform)
    dvf_filter = sitk.TransformToDisplacementFieldFilter()
    dvf_filter.SetReferenceImage(fixed)
    dvf_image = dvf_filter.Execute(composite)

    return composite, dvf_image


def register_pair(
    fixed_path: str,
    moving_path: str,
    out_reg_path: str,
    out_dvf_path: str,
    nthreads: int = 8,
) -> dict:
    """
    Full 3-stage registration pipeline: rigid → affine → B-spline.
    Returns a dict with registration metadata for logging.
    """
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(nthreads)
    t0 = time.time()

    log.info(f"  Loading fixed (NCCT): {os.path.basename(fixed_path)}")
    fixed = load_sitk_image(fixed_path)
    log.info(f"  Loading moving (target): {os.path.basename(moving_path)}")
    moving = load_sitk_image(moving_path)

    # Clip HU to suppress contrast agent influence during registration
    # (registration driven by anatomy, not contrast uptake)
    fixed_clipped  = sitk.Clamp(fixed,  sitk.sitkFloat32, -200.0, 300.0)
    moving_clipped = sitk.Clamp(moving, sitk.sitkFloat32, -200.0, 300.0)

    log.info("    Stage 1: Rigid registration...")
    rigid_xfm = rigid_registration(fixed_clipped, moving_clipped)

    log.info("    Stage 2: Affine registration...")
    affine_xfm = affine_registration(fixed_clipped, moving_clipped, rigid_xfm)

    log.info("    Stage 3: B-Spline deformable registration...")
    bspline_xfm, dvf_image = bspline_registration(
        fixed_clipped, moving_clipped, affine_xfm, grid_spacing_mm=50.0
    )

    log.info("    Resampling moving to fixed grid...")
    registered = resample_to_fixed(moving, fixed, bspline_xfm)

    # Save registered volume
    os.makedirs(os.path.dirname(out_reg_path), exist_ok=True)
    sitk.WriteImage(sitk.Cast(registered, sitk.sitkFloat32), out_reg_path)
    log.info(f"    Saved registered volume -> {out_reg_path}")

    # Save displacement vector field
    sitk.WriteImage(dvf_image, out_dvf_path)
    log.info(f"    Saved DVF -> {out_dvf_path}")

    elapsed = time.time() - t0
    return {
        "elapsed_sec":      round(elapsed, 1),
        "fixed_size":       list(fixed.GetSize()),
        "fixed_spacing_mm": [round(s, 3) for s in fixed.GetSpacing()],
    }


# ---------------------------------------------------------------------------
# Main Worker
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load pairs manifest
    if not os.path.exists(args.pairs):
        log.error(f"Pairs manifest not found: {args.pairs}")
        log.error("Run discover_multiphase_cohort.py first.")
        sys.exit(1)

    pairs_df = pd.read_csv(args.pairs, dtype=str)
    log.info(f"Loaded {len(pairs_df):,} pairs from {args.pairs}")

    # Optional same-day filter
    if args.same_day_only:
        pairs_df = pairs_df[pairs_df["same_day"] == "True"]
        log.info(f"  After same-day filter: {len(pairs_df):,} pairs")

    # Chunk for SLURM array
    if args.total_chunks > 1:
        all_indices = list(range(len(pairs_df)))
        chunk = [i for i in all_indices if i % args.total_chunks == args.chunk_idx]
        pairs_df = pairs_df.iloc[chunk].reset_index(drop=True)
        log.info(f"  Chunk {args.chunk_idx}/{args.total_chunks}: {len(pairs_df)} pairs to register")

    os.makedirs(args.out_dir, exist_ok=True)

    # CSV log writer
    log_fields = [
        "pair_id", "patient_id", "subject_p0", "subject_target",
        "phase_target", "status", "elapsed_sec", "error", "out_reg_path",
    ]
    log_csv_exists = os.path.exists(args.log_csv)
    log_f = open(args.log_csv, "a", newline="")
    log_writer = csv.DictWriter(log_f, fieldnames=log_fields)
    if not log_csv_exists:
        log_writer.writeheader()

    # ------------------------------------------------------------------
    n_done = 0
    n_skip = 0
    n_fail = 0

    for idx, row in pairs_df.iterrows():
        pair_id = f"{row['subject_p0']}--{row['subject_target']}"
        phase_tag = str(row["phase_target"]).replace("/", "_")
        out_reg  = os.path.join(args.out_dir, f"{pair_id}_reg.nii.gz")
        out_dvf  = os.path.join(args.out_dir, f"{pair_id}_dvf.nii.gz")

        if args.skip_existing and os.path.exists(out_reg):
            log.info(f"[{idx+1}/{len(pairs_df)}] SKIP (exists): {pair_id}")
            n_skip += 1
            continue

        log.info(f"\n[{idx+1}/{len(pairs_df)}] Registering pair: {pair_id}")
        log.info(f"  Phase target: {row['phase_target']} | Same-day: {row.get('same_day', 'N/A')}")

        status  = "FAILED"
        elapsed = 0.0
        err_msg = ""

        try:
            meta = register_pair(
                fixed_path  = row["ct_path_p0"],
                moving_path = row["ct_path_target"],
                out_reg_path = out_reg,
                out_dvf_path = out_dvf,
                nthreads     = args.nthreads,
            )
            status  = "OK"
            elapsed = meta["elapsed_sec"]
            n_done  += 1
            log.info(f"  Completed in {elapsed:.1f}s")

        except KeyboardInterrupt:
            log.warning("Interrupted by user.")
            break

        except Exception as e:
            err_msg = str(e)
            n_fail += 1
            log.error(f"  FAILED: {err_msg}")
            log.debug(traceback.format_exc())

        log_writer.writerow({
            "pair_id":        pair_id,
            "patient_id":     row.get("patient_id", ""),
            "subject_p0":     row.get("subject_p0", ""),
            "subject_target": row.get("subject_target", ""),
            "phase_target":   row.get("phase_target", ""),
            "status":         status,
            "elapsed_sec":    elapsed,
            "error":          err_msg,
            "out_reg_path":   out_reg if status == "OK" else "",
        })
        log_f.flush()

    log_f.close()
    log.info(f"\n[Registration Complete] Done: {n_done} | Skipped: {n_skip} | Failed: {n_fail}")
    log.info(f"  Registration log -> {args.log_csv}")


if __name__ == "__main__":
    main()
