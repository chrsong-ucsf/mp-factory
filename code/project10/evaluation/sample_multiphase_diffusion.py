#!/usr/bin/env python3
"""
sample_multiphase_diffusion.py

Inference sampling script for Project 10 Stage 2:
Synthesizes 3D contrast-enhanced CT volumes from non-contrast CT (NCCT) scans
using the trained Stage 2 Latent Diffusion Model and frozen Stage 1 3D Autoencoder.

Supports multi-GPU sharding across SLURM array tasks (--array_id, --total_workers).
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

from code.project10.common_spatial import preprocess_single_volume

from code.project10.models.autoencoder_3d import (
    AutoEncoder3DConfig,
    LightweightAutoEncoder3D,
    build_autoencoder,
    normalise_hu,
    denormalise_hu,
)
from code.project10.models.synthesis_net import AnatomyConditionalSynthesisNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("diffusion_sampler")


def checkpoint_sha256(path: Path | str) -> str:
    """Calculate SHA256 hex digest of a checkpoint file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class DDIMSampler:
    """Fast DDIM deterministic reverse diffusion sampler for 3D latents."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        source_latent: torch.Tensor,
        anatomy_mask: torch.Tensor,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        phase_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run DDIM reverse diffusion trajectory from pure Gaussian noise.
        
        Args:
            model: Trained AnatomyConditionalSynthesisNet
            source_latent: Encoded NCCT latent [B, 4, D, H, W]
            anatomy_mask: Downsampled SPADE conditioning mask [B, 5, D, H, W]
            num_inference_steps: Number of DDIM sampling steps (default: 50)
            eta: Stochasticity parameter (0.0 = deterministic DDIM)
            phase_labels: Target phase integer tensor [B] (0..4)
        """
        device = source_latent.device
        B = source_latent.shape[0]
        step_ratio = self.timesteps // num_inference_steps
        time_steps = torch.arange(0, self.timesteps, step_ratio, device=device).long()
        time_steps = torch.flip(time_steps, dims=[0])  # Reverse from T down to 0

        # Start from pure Gaussian noise
        x_t = torch.randn_like(source_latent)

        for i, t in enumerate(time_steps):
            t_batch = torch.full((B,), t.item(), device=device, dtype=torch.long)
            alpha_bar_t = self.alphas_cumprod[t]

            # Next timestep t_prev
            if i < len(time_steps) - 1:
                t_prev = time_steps[i + 1]
                alpha_bar_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # Predict noise with target phase conditioning (F06)
            eps_pred = model(
                x=x_t,
                timesteps=t_batch,
                source_latent=source_latent,
                anatomy_mask=anatomy_mask,
                phase_labels=phase_labels,
            )

            # Estimate x_0 (clean latent)
            pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)

            # DDIM update formula
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma**2) * eps_pred
            noise = torch.randn_like(x_t) if (sigma > 0) else 0.0

            x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise

        return x_t


def preprocess_ct(ct_path: str, target_shape=(128, 128, 128), spacing_mm=1.5):
    """
    Resample CT volume to 1.5mm isotropic using unified MONAI spatial pipeline.
    Returns (normalised_tensor, resampled_affine).
    """
    try:
        vol, affine = preprocess_single_volume(
            ct_path,
            target_shape=target_shape,
            spacing=(spacing_mm, spacing_mm, spacing_mm),
            is_mask=False,
        )
    except Exception as e:
        log.warning(f"MONAI preprocess failed ({e}), falling back to SimpleITK resample")
        img = sitk.ReadImage(ct_path)
        cur_spacing = img.GetSpacing()
        cur_size = img.GetSize()
        new_spacing = (spacing_mm, spacing_mm, spacing_mm)
        new_size = [int(round(cur_size[i] * cur_spacing[i] / new_spacing[i])) for i in range(3)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000.0)
        resampled = resampler.Execute(img)
        arr = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)

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
        
        # Preserve true world coordinate affine in RAS
        dir_mat = np.array(resampled.GetDirection()).reshape(3, 3)
        spacing_diag = np.diag(new_spacing)
        rot_spacing = dir_mat @ spacing_diag
        crop_origin_vox = np.array([cx - sx, cy - sy, cz - sz], dtype=float)
        crop_origin_world = np.array(resampled.GetOrigin()) + dir_mat @ (crop_origin_vox * np.array(new_spacing))
        affine_lps = np.eye(4)
        affine_lps[:3, :3] = rot_spacing
        affine_lps[:3, 3] = crop_origin_world
        lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
        affine = lps_to_ras @ affine_lps

    t_vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()
    return normalise_hu(t_vol), affine


def load_anatomy_mask(
    subject_id: str,
    mask_dirs: list,
    target_shape=(128, 128, 128),
    latent_shape=(16, 16, 16),
    device=torch.device("cpu"),
    totalseg_task="auto",
    require_conditioning=False,
    reference_ct_path: Optional[str] = None,
    return_path: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[str]]]:
    """
    Load 5-organ GI mask if available, physically align to reference CT grid,
    semantically remap TotalSeg if needed, one-hot encode, and downsample to latent resolution.
    Explicit totalseg_task takes strict precedence over content heuristics.
    Falls back to neutral background prior [1, 0, 0, 0, 0] if no mask is found
    and require_conditioning is False.
    """
    candidate_suffixes = [
        "_gi_mask.nii.gz",
        "_mask.nii.gz",
        ".nii.gz",
        ".nii",
        "_consensus.nii.gz",
        "_gi_seg.nii.gz",
        "_autolabel.nii.gz",
    ]

    ref_ct_itk = None
    if reference_ct_path and os.path.exists(reference_ct_path):
        try:
            ref_ct_itk = sitk.ReadImage(str(reference_ct_path))
        except Exception as e:
            log.warning(f"Could not load reference CT from {reference_ct_path}: {e}")

    rejection_reasons = []

    for mdir in mask_dirs:
        for sfx in candidate_suffixes:
            mpath = os.path.join(mdir, f"{subject_id}{sfx}")
            if os.path.exists(mpath):
                try:
                    vol = None
                    if ref_ct_itk is not None:
                        # 1. Physical alignment: resample mask into reference CT geometry
                        mask_itk = sitk.ReadImage(mpath)
                        resampler = sitk.ResampleImageFilter()
                        resampler.SetReferenceImage(ref_ct_itk)
                        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                        resampler.SetDefaultPixelValue(0)
                        res_mask_itk = resampler.Execute(mask_itk)
                        native_arr = sitk.GetArrayFromImage(res_mask_itk)

                        # Check physical overlap with CT volume
                        if np.sum(native_arr > 0) == 0:
                            rejection_reasons.append(
                                (mpath, f"zero physical overlap with reference CT {reference_ct_path}")
                            )
                            log.warning(f"Candidate mask {mpath} has zero physical overlap with reference CT; trying next candidate.")
                            continue

                        # Save aligned native mask to temp file and preprocess with unified MONAI pipeline
                        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_f:
                            tmp_path = tmp_f.name
                        try:
                            sitk.WriteImage(res_mask_itk, tmp_path)
                            vol, _ = preprocess_single_volume(
                                tmp_path,
                                target_shape=target_shape,
                                spacing=(1.5, 1.5, 1.5),
                                is_mask=True,
                            )
                        finally:
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                    else:
                        vol, _ = preprocess_single_volume(
                            mpath,
                            target_shape=target_shape,
                            spacing=(1.5, 1.5, 1.5),
                            is_mask=True,
                        )

                    # Semantic remapping to 5-class GI scheme (0=bg, 1=stomach, 2=duodenum, 3=small bowel, 4=colon)
                    mapped = np.zeros_like(vol, dtype=np.uint8)
                    if totalseg_task == "total_v1":
                        # Explicit TotalSegmentator v1: 6=stomach, 56=duodenum, 55=small_bowel, 57=colon
                        mapped[vol == 6] = 1
                        mapped[vol == 56] = 2
                        mapped[vol == 55] = 3
                        mapped[vol == 57] = 4
                    elif totalseg_task == "total":
                        # Explicit TotalSegmentator v2 full body: 6=stomach, 19=duodenum, 18=small_bowel, 20=colon
                        mapped[vol == 6] = 1
                        mapped[vol == 19] = 2
                        mapped[vol == 18] = 3
                        mapped[vol == 20] = 4
                    elif totalseg_task == "gi":
                        # Standard GI 5-class mask (0..4)
                        for organ_id in (1, 2, 3, 4):
                            mapped[vol == organ_id] = organ_id
                    else:
                        # Auto-detection heuristic: check v1 specific IDs first
                        if (56 in vol) or (55 in vol) or (57 in vol):
                            mapped[vol == 6] = 1
                            mapped[vol == 56] = 2
                            mapped[vol == 55] = 3
                            mapped[vol == 57] = 4
                        elif (19 in vol) or (18 in vol) or (20 in vol):
                            mapped[vol == 6] = 1
                            mapped[vol == 19] = 2
                            mapped[vol == 18] = 3
                            mapped[vol == 20] = 4
                        else:
                            for organ_id in (1, 2, 3, 4):
                                mapped[vol == organ_id] = organ_id

                    if np.sum(mapped > 0) == 0:
                        rejection_reasons.append(
                            (mpath, f"zero foreground voxels for GI schema '{totalseg_task}'")
                        )
                        log.warning(f"Mask {mpath} has 0 foreground voxels for GI schema '{totalseg_task}'; trying next candidate.")
                        continue

                    mask_t = torch.from_numpy(mapped).unsqueeze(0).long().clamp(0, 4)
                    one_hot = F.one_hot(mask_t, num_classes=5).permute(0, 4, 1, 2, 3).float().to(device)
                    interp_mask = F.interpolate(one_hot, size=latent_shape, mode="nearest")
                    log.info(f"Successfully loaded and verified anatomy conditioning mask: {mpath}")
                    if return_path:
                        return interp_mask, mpath
                    return interp_mask
                except Exception as e:
                    rejection_reasons.append((mpath, str(e)))
                    log.warning(f"Failed loading candidate mask from {mpath}: {e}; trying next candidate.")
                    continue

    if require_conditioning:
        if rejection_reasons:
            reasons_summary = "; ".join(f"{p}: {r}" for p, r in rejection_reasons)
            raise ValueError(
                f"CRITICAL: Required organ conditioning mask for {subject_id} failed on all candidates: {reasons_summary}"
            )
        else:
            raise FileNotFoundError(
                f"CRITICAL: Required organ conditioning mask for {subject_id} not found in {mask_dirs}."
            )

    log.warning(f"No organ mask found for {subject_id}; generating with neutral background prior.")
    dummy = torch.zeros((1, 5, *latent_shape), device=device, dtype=torch.float32)
    dummy[:, 0] = 1.0
    if return_path:
        return dummy, None
    return dummy


PHASE_NAME_TO_ID = {
    "noncontrast": 0, "ncct": 0, "0": 0,
    "arterial_early": 1, "early_arterial": 1, "1": 1,
    "arterial_late": 2, "late_arterial": 2, "arterial": 2, "2": 2,
    "venous": 3, "portal_venous": 3, "portal": 3, "pv": 3, "3": 3,
    "delayed": 4, "4": 4,
}
ID_TO_PHASE_NAME = {
    0: "noncontrast",
    1: "arterial_early",
    2: "arterial_late",
    3: "venous",
    4: "delayed",
}


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Multiphase Diffusion Inference & Sampling")
    parser.add_argument("--registered_dir", "--data_dir", type=str,
                        dest="registered_dir",
                        default="results/project10/registered_volumes",
                        help="Directory containing registered moving/fixed CT volumes")
    parser.add_argument("--ncct_dir", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox",
                        help="Directory containing true native non-contrast (NCCT) CT scans")
    parser.add_argument("--mask_dirs", nargs="+",
                        default=["results/totalseg_gi_masks", "results/totalseg_masks"],
                        help="Directories to search for organ masks")
    parser.add_argument("--ae_checkpoint", type=str,
                        default="results/project10/checkpoints/stage1_ae/best_stage1_ae.pt",
                        help="Path to trained Stage 1 Autoencoder checkpoint")
    parser.add_argument("--diffusion_checkpoint", "--stage2_checkpoint", type=str,
                        dest="diffusion_checkpoint",
                        default="results/project10/checkpoints/stage2_synthesis/best_synthesis_net.pt",
                        help="Path to trained Stage 2 Latent Diffusion checkpoint")
    parser.add_argument("--out_dir", type=str,
                        default="results/project10/synthetic_volumes",
                        help="Output directory for generated synthetic NIfTI CT volumes")
    parser.add_argument("--pairs_csv", type=str,
                        default="results/project10/manifests/multiphase_pairs.csv",
                        help="Path to multiphase_pairs.csv for authoritative phase metadata lookup")
    parser.add_argument("--target_phase", type=str, default="auto",
                        help="Target contrast acquisition phase ('auto', 'arterial_early', 'arterial_late', 'venous', 'delayed', or 0..4)")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of DDIM sampling steps (default: 50)")
    parser.add_argument("--array_id", type=int, default=0, help="SLURM array task index")
    parser.add_argument("--total_workers", type=int, default=1, help="Total number of parallel array workers")
    parser.add_argument("--totalseg_task", type=str, default="auto", choices=["auto", "total", "total_v1", "gi"],
                        help="Producer schema for TotalSegmentator masks ('auto', 'total', 'total_v1', 'gi')")
    parser.add_argument("--allow_legacy_checkpoint", action="store_true",
                        help="Allow diffusion checkpoint without Stage 1 AE or conditioning metadata")
    parser.add_argument("--require_conditioning", action="store_true",
                        help="Strictly fail if organ conditioning mask is missing or empty")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force regenerate existing synthetic volumes and provenance sidecars")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum scans to sample per worker")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device} | Worker: [{args.array_id}/{args.total_workers}]")

    # Load authoritative pairs manifest for target acquisition phase mapping (F06)
    pairs_meta = {}
    if args.pairs_csv:
        if not os.path.exists(args.pairs_csv):
            raise FileNotFoundError(f"Specified --pairs_csv manifest not found: {args.pairs_csv}")
        try:
            pdf = pd.read_csv(args.pairs_csv, dtype=str)
            for idx, prow in pdf.iterrows():
                s_p0 = str(prow.get("subject_p0", "")).strip()
                s_tgt = str(prow.get("subject_target", "")).strip()
                p_tgt = str(prow.get("phase_target", prow.get("phase_target_raw", ""))).strip()
                if s_p0 and s_tgt:
                    for k in [(s_p0, s_tgt), f"{s_p0}--{s_tgt}", f"{s_p0}_{s_tgt}", s_tgt]:
                        if k in pairs_meta and pairs_meta[k] != p_tgt:
                            raise ValueError(
                                f"Contradictory phase mapping in {args.pairs_csv} row {idx} for '{k}': "
                                f"existing phase '{pairs_meta[k]}' vs new phase '{p_tgt}'"
                            )
                        pairs_meta[k] = p_tgt
            log.info(f"Loaded {len(pairs_meta)} pair phase mappings from {args.pairs_csv}")
        except Exception as pe:
            raise ValueError(f"Failed to load required pairs manifest from {args.pairs_csv}: {pe}")

    # Discover candidate CT volumes
    all_files = sorted(glob.glob(os.path.join(args.registered_dir, "*_reg.nii.gz")))
    if not all_files:
        log.error(f"No *_reg.nii.gz files found in {args.registered_dir}")
        return

    chunk_size = (len(all_files) + args.total_workers - 1) // args.total_workers
    start_idx = args.array_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_files))
    my_files = all_files[start_idx:end_idx]

    if args.max_samples:
        my_files = my_files[:args.max_samples]

    log.info(f"Total volumes discovered: {len(all_files)} | Assigned to worker {args.array_id}: {len(my_files)}")

    # 1. Load Stage 1 Autoencoder
    log.info(f"Loading Stage 1 Autoencoder from: {args.ae_checkpoint}")
    if not os.path.exists(args.ae_checkpoint):
        raise FileNotFoundError(f"Stage 1 Autoencoder checkpoint not found at: {args.ae_checkpoint}")

    ae_sha = checkpoint_sha256(args.ae_checkpoint)
    log.info(f"Loaded Stage 1 AE SHA256: {ae_sha[:12]}...")

    ckpt_ae = torch.load(args.ae_checkpoint, map_location=device)
    cfg_dict = ckpt_ae.get("cfg", {})
    if isinstance(cfg_dict, dict):
        ae_cfg = AutoEncoder3DConfig(**cfg_dict)
    else:
        ae_cfg = AutoEncoder3DConfig()

    ae = LightweightAutoEncoder3D(ae_cfg).to(device)
    state_dict_ae = ckpt_ae["model"] if "model" in ckpt_ae else ckpt_ae
    ae.load_state_dict(state_dict_ae, strict=True)
    ae.eval()

    # 2. Load Stage 2 Diffusion Synthesis Model
    log.info(f"Loading Stage 2 Diffusion Model from: {args.diffusion_checkpoint}")
    if not os.path.isfile(args.diffusion_checkpoint):
        raise FileNotFoundError(
            f"CRITICAL: Diffusion checkpoint not found at: {args.diffusion_checkpoint}. "
            "Aborting to prevent uninitialized random-weight generation."
        )

    diff_sha = checkpoint_sha256(args.diffusion_checkpoint)
    log.info(f"Loaded Stage 2 Diffusion SHA256: {diff_sha[:12]}...")

    ckpt_diff = torch.load(args.diffusion_checkpoint, map_location=device)

    # Validate AE Identity
    ckpt_ae_sha = ckpt_diff.get("ae_checkpoint_sha256")
    if ckpt_ae_sha is not None:
        if ckpt_ae_sha != ae_sha:
            raise ValueError(
                f"Diffusion checkpoint {args.diffusion_checkpoint} was trained with AE {ckpt_ae_sha}, but loaded AE is {ae_sha}."
            )
        log.info(f"[Provenance Check] Verified AE hash matches diffusion checkpoint: {ae_sha[:12]}...")
    else:
        if not args.allow_legacy_checkpoint:
            raise ValueError(
                f"Legacy checkpoint {args.diffusion_checkpoint} lacks Stage 1 AE identity. "
                "Explicit override required via --allow_legacy_checkpoint."
            )

    # Validate Conditioning Identity
    ckpt_cond = ckpt_diff.get("conditioning")
    if ckpt_cond is not None:
        if not args.require_conditioning:
            args.require_conditioning = True
            log.info("[Provenance Check] Enforcing --require_conditioning as checkpoint was trained with anatomy conditioning.")
        if isinstance(ckpt_cond, dict):
            train_task = ckpt_cond.get("totalseg_task")
            if args.totalseg_task == "auto" and train_task:
                args.totalseg_task = train_task
                log.info(f"[Provenance Check] Resolved totalseg_task to '{train_task}' from checkpoint metadata.")
            elif args.totalseg_task != "auto" and train_task and args.totalseg_task != train_task:
                raise ValueError(
                    f"Diffusion checkpoint was trained with totalseg_task='{train_task}', but requested '{args.totalseg_task}'."
                )
    else:
        if not args.allow_legacy_checkpoint:
            raise ValueError(
                f"Legacy checkpoint {args.diffusion_checkpoint} lacks conditioning identity. "
                "Explicit override required via --allow_legacy_checkpoint."
            )

    model = AnatomyConditionalSynthesisNet(
        spatial_dims=3,
        latent_channels=4,
        label_nc=5,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=2,
        num_head_channels=64,
    ).to(device)

    state_dict_diff = ckpt_diff["model"] if "model" in ckpt_diff else ckpt_diff
    state_dict_diff = {k.replace("module.", ""): v for k, v in state_dict_diff.items()}
    model.load_state_dict(state_dict_diff)
    model.eval()

    trained_timesteps = ckpt_diff.get("timesteps", 1000) if isinstance(ckpt_diff, dict) else 1000
    sampler = DDIMSampler(timesteps=trained_timesteps).to(device)

    # 3. Inference Loop
    attempted_count = 0
    generated_count = 0
    skipped_count = 0
    failed_count = 0

    CODE_VERSION_TAG = "stage2_diffusion_v2_canonical_spatial"

    for idx, ct_path in enumerate(my_files, start=1):
        filename = os.path.basename(ct_path)
        pair_name = filename.replace("_reg.nii.gz", "")
        out_path = os.path.join(args.out_dir, f"{pair_name}_synthetic_contrast.nii.gz")
        prov_path = os.path.join(args.out_dir, f"{pair_name}_provenance.json")

        # Extract subject ID and target ID for anatomy conditioning, true NCCT, and phase lookup
        clean_name = pair_name
        for sfx in ["_reg", "_synthetic_contrast"]:
            if clean_name.endswith(sfx):
                clean_name = clean_name[:-len(sfx)]
        if "--" in clean_name:
            parts = clean_name.split("--")
            subject_p0 = parts[0]
            subject_target = parts[1]
        elif "_" in clean_name:
            parts = clean_name.split("_")
            subject_p0 = parts[0]
            subject_target = "_".join(parts[1:])
        else:
            subject_p0 = clean_name
            subject_target = clean_name

        # Resolve true native NCCT scan first to verify input provenance
        ncct_input_file = None
        cands = []
        if args.ncct_dir and os.path.exists(args.ncct_dir):
            cands.extend([
                os.path.join(args.ncct_dir, f"{subject_p0}.nii.gz"),
                os.path.join(args.ncct_dir, subject_p0, "ct.nii.gz")
            ])
        cands.extend([
            os.path.join(str(BASE_DIR), "CancerVerse", "CancerVerse", subject_p0, "ct.nii.gz"),
            os.path.join(str(BASE_DIR), "CancerVerse_dbox", subject_p0, "ct.nii.gz")
        ])
        for c in cands:
            if os.path.exists(c):
                ncct_input_file = c
                break

        if ncct_input_file is None:
            failed_count += 1
            log.error(
                f"CRITICAL: Native NCCT file for {subject_p0} not found in {args.ncct_dir} or CancerVerse. "
                "Aborting sample to strictly prevent target leakage from registered contrast CT."
            )
            continue

        ncct_sha = checkpoint_sha256(ncct_input_file)

        # Resolve candidate anatomy conditioning mask before resume check to ensure contract validity
        try:
            anatomy_mask, mask_path = load_anatomy_mask(
                subject_id=subject_p0,
                mask_dirs=args.mask_dirs,
                target_shape=(128, 128, 128),
                latent_shape=(16, 16, 16),
                device=device,
                totalseg_task=args.totalseg_task,
                require_conditioning=args.require_conditioning,
                reference_ct_path=ncct_input_file,
                return_path=True,
            )
        except Exception as e:
            failed_count += 1
            log.error(f"Failed resolving required anatomy conditioning mask for {pair_name}: {e}")
            continue

        mask_sha = checkpoint_sha256(mask_path) if mask_path and os.path.exists(mask_path) else None

        # Resolve target phase (F06)
        target_phase_id = None
        if args.target_phase != "auto":
            phase_key = str(args.target_phase).lower()
            if phase_key not in PHASE_NAME_TO_ID:
                raise ValueError(f"Unknown --target_phase '{args.target_phase}'. Choose from {list(PHASE_NAME_TO_ID.keys())}")
            target_phase_id = PHASE_NAME_TO_ID[phase_key]
        else:
            # 1. Authoritative lookup from pairs manifest
            meta_phase = (
                pairs_meta.get((subject_p0, subject_target))
                or pairs_meta.get(pair_name)
                or pairs_meta.get(clean_name)
                or pairs_meta.get(subject_target)
            )
            if meta_phase:
                m_lower = str(meta_phase).lower()
                if "early_art" in m_lower or "arterial_early" in m_lower or "p1" in m_lower:
                    target_phase_id = 1
                elif "late_art" in m_lower or "arterial" in m_lower or "p2" in m_lower:
                    target_phase_id = 2
                elif "delayed" in m_lower or "p4" in m_lower:
                    target_phase_id = 4
                elif "venous" in m_lower or "portal" in m_lower or "pv" in m_lower or "p3" in m_lower:
                    target_phase_id = 3
                elif m_lower in PHASE_NAME_TO_ID:
                    target_phase_id = PHASE_NAME_TO_ID[m_lower]

            # 2. Check explicit phase token in candidate filename
            if target_phase_id is None:
                p_lower = pair_name.lower()
                if "arterial_early" in p_lower or "early_art" in p_lower:
                    target_phase_id = 1
                elif "arterial" in p_lower:
                    target_phase_id = 2
                elif "delayed" in p_lower:
                    target_phase_id = 4
                elif "venous" in p_lower or "portal" in p_lower or "pv" in p_lower:
                    target_phase_id = 3

            # 3. Fail closed if phase cannot be verified
            if target_phase_id is None:
                raise ValueError(
                    f"Cannot determine target acquisition phase for candidate '{pair_name}'. "
                    f"Target subject '{subject_target}' not found in {args.pairs_csv} and filename has no phase token. "
                    "Specify --target_phase explicitly or provide valid manifest."
                )
        target_phase_name = ID_TO_PHASE_NAME.get(target_phase_id, f"phase_{target_phase_id}")
        target_phase_tensor = torch.tensor([target_phase_id], device=device, dtype=torch.long)

        # Complete generation contract verification
        if os.path.exists(out_path) and not args.overwrite:
            try:
                is_valid = False
                if os.path.exists(prov_path):
                    with open(prov_path, "r") as pf:
                        prov_data = json.load(pf)
                    if (
                        prov_data.get("ncct_sha256") == ncct_sha
                        and prov_data.get("mask_sha256") == mask_sha
                        and prov_data.get("mask_path") == mask_path
                        and prov_data.get("diffusion_checkpoint_sha256") == diff_sha
                        and prov_data.get("ae_checkpoint_sha256") == ae_sha
                        and prov_data.get("totalseg_task") == args.totalseg_task
                        and prov_data.get("target_phase") == target_phase_id
                        and prov_data.get("num_steps") == args.num_steps
                        and prov_data.get("code_version") == CODE_VERSION_TAG
                    ):
                        test_nii = nib.load(out_path)
                        test_data = test_nii.get_fdata()
                        if test_nii.shape == (128, 128, 128) and np.all(np.isfinite(test_data)):
                            is_valid = True
                if is_valid:
                    log.info(f"[{idx}/{len(my_files)}] Skipping valid existing verified volume (matching input, mask, and model contract): {pair_name}")
                    skipped_count += 1
                    continue
                else:
                    log.warning(f"Existing {out_path} lacks matching provenance or has altered input/mask/model contract; regenerating...")
            except Exception as e:
                log.warning(f"Existing {out_path} unreadable ({e}), regenerating...")

        attempted_count += 1
        log.info(f"[{idx}/{len(my_files)}] Generating synthetic contrast volume for: {pair_name} (Phase {target_phase_id}: {target_phase_name})")

        try:
            # Preprocess volume to normalised range [-0.5, 1.0] and get resampled affine
            ncct_tensor, resampled_affine = preprocess_ct(ncct_input_file)
            ncct_tensor = ncct_tensor.to(device)

            with torch.no_grad():
                # Encode NCCT to latent space: (1, 4, 16, 16, 16)
                z_ncct = ae.encode(ncct_tensor)

                # Sample target latent via DDIM with target phase conditioning (F06)
                z_syn = sampler.sample(
                    model=model,
                    source_latent=z_ncct,
                    anatomy_mask=anatomy_mask,
                    num_inference_steps=args.num_steps,
                    eta=0.0,
                    phase_labels=target_phase_tensor,
                )

                # Decode latent to full 3D CT volume (1, 1, 128, 128, 128)
                fake_vol = ae.decode(z_syn)

                # Denormalize back to real Hounsfield Units [-1000, 2000]
                fake_hu = denormalise_hu(fake_vol).squeeze().cpu().numpy()
                fake_hu = np.clip(fake_hu, -1000.0, 2000.0).astype(np.int16)

                # Save as NIfTI with resampled physical affine coordinate system
                nii = nib.Nifti1Image(fake_hu, affine=resampled_affine)
                nib.save(nii, out_path)

                # Write complete provenance metadata sidecar
                prov_meta = {
                    "pair_name": pair_name,
                    "subject_p0": subject_p0,
                    "ncct_input_file": ncct_input_file,
                    "ncct_sha256": ncct_sha,
                    "mask_path": mask_path,
                    "mask_sha256": mask_sha,
                    "diffusion_checkpoint": args.diffusion_checkpoint,
                    "diffusion_checkpoint_sha256": diff_sha,
                    "ae_checkpoint": args.ae_checkpoint,
                    "ae_checkpoint_sha256": ae_sha,
                    "totalseg_task": args.totalseg_task,
                    "require_conditioning": args.require_conditioning,
                    "target_phase": target_phase_id,
                    "target_phase_name": target_phase_name,
                    "num_steps": args.num_steps,
                    "target_shape": [128, 128, 128],
                    "spacing_mm": 1.5,
                    "code_version": CODE_VERSION_TAG,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with open(prov_path, "w") as pf:
                    json.dump(prov_meta, pf, indent=2)

                log.info(f"  [+] Saved synthetic CT volume -> {out_path} (with verified provenance)")
                generated_count += 1

        except Exception as e:
            failed_count += 1
            log.error(f"Failed processing {pair_name}: {e}")

    log.info(
        f"Worker {args.array_id} summary: {len(my_files)} assigned, "
        f"{skipped_count} skipped verified, {generated_count} newly generated, "
        f"{failed_count} failed, {attempted_count} attempted."
    )
    if failed_count > 0:
        raise RuntimeError(f"Inference failed on {failed_count}/{len(my_files)} assigned scans on worker {args.array_id}.")


if __name__ == "__main__":
    main()
