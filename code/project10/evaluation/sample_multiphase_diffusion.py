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
import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE_DIR))

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
    ) -> torch.Tensor:
        """
        Run DDIM reverse diffusion trajectory from pure Gaussian noise.
        
        Args:
            model: Trained AnatomyConditionalSynthesisNet
            source_latent: Encoded NCCT latent [B, 4, D, H, W]
            anatomy_mask: Downsampled SPADE conditioning mask [B, 5, D, H, W]
            num_inference_steps: Number of DDIM sampling steps (default: 50)
            eta: Stochasticity parameter (0.0 = deterministic DDIM)
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

            # Predict noise
            eps_pred = model(
                x=x_t,
                timesteps=t_batch,
                source_latent=source_latent,
                anatomy_mask=anatomy_mask,
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


def preprocess_ct(ct_path: str, target_shape=(128, 128, 128), spacing_mm=1.5) -> torch.Tensor:
    """Resample CT volume to 1.5mm isotropic, pad/crop with -1000 HU air, and normalise."""
    img = sitk.ReadImage(ct_path)
    current_spacing = img.GetSpacing()
    current_size = img.GetSize()

    new_spacing = (spacing_mm, spacing_mm, spacing_mm)
    new_size = [
        int(round(current_size[i] * current_spacing[i] / new_spacing[i]))
        for i in range(3)
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

    # Pad or crop to target_shape with air background (-1000 HU)
    vol = np.full(target_shape, -1000.0, dtype=np.float32)
    min_x = min(arr.shape[0], target_shape[0])
    min_y = min(arr.shape[1], target_shape[1])
    min_z = min(arr.shape[2], target_shape[2])

    sx = (target_shape[0] - min_x) // 2
    sy = (target_shape[1] - min_y) // 2
    sz = (target_shape[2] - min_z) // 2
    vol[sx : sx + min_x, sy : sy + min_y, sz : sz + min_z] = arr[:min_x, :min_y, :min_z]

    t_vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()  # (1, 1, D, H, W)
    return normalise_hu(t_vol)


def load_anatomy_mask(
    subject_id: str,
    mask_dirs: list,
    target_shape=(128, 128, 128),
    latent_shape=(16, 16, 16),
    device=torch.device("cpu"),
) -> torch.Tensor:
    """
    Load 5-organ GI mask if available, one-hot encode, and downsample to latent resolution.
    Falls back to neutral background prior [1, 0, 0, 0, 0] if no mask is found.
    """
    for mdir in mask_dirs:
        for ext in [".nii.gz", ".nii"]:
            mpath = os.path.join(mdir, f"{subject_id}{ext}")
            if os.path.exists(mpath):
                try:
                    img = sitk.ReadImage(mpath)
                    resampler = sitk.ResampleImageFilter()
                    resampler.SetOutputSpacing((1.5, 1.5, 1.5))
                    resampler.SetSize(target_shape)
                    resampler.SetOutputOrigin(img.GetOrigin())
                    resampler.SetOutputDirection(img.GetDirection())
                    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler.SetDefaultPixelValue(0)
                    resampled = resampler.Execute(img)
                    arr = sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)

                    mask_t = torch.from_numpy(arr).unsqueeze(0).long().clamp(0, 4)
                    one_hot = F.one_hot(mask_t, num_classes=5).permute(0, 4, 1, 2, 3).float().to(device)
                    return F.interpolate(one_hot, size=latent_shape, mode="nearest")
                except Exception as e:
                    log.warning(f"Failed loading mask from {mpath}: {e}")

    # Neutral prior
    dummy = torch.zeros((1, 5, *latent_shape), device=device, dtype=torch.float32)
    dummy[:, 0] = 1.0
    return dummy


def main():
    parser = argparse.ArgumentParser(description="Multiphase Diffusion Inference & Sampling")
    parser.add_argument("--registered_dir", type=str,
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
    parser.add_argument("--diffusion_checkpoint", type=str,
                        default="results/project10/checkpoints/stage2_synthesis/best_synthesis_net.pt",
                        help="Path to trained Stage 2 Latent Diffusion checkpoint")
    parser.add_argument("--out_dir", type=str,
                        default="results/project10/synthetic_volumes",
                        help="Output directory for generated synthetic NIfTI CT volumes")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of DDIM sampling steps (default: 50)")
    parser.add_argument("--array_id", type=int, default=0, help="SLURM array task index")
    parser.add_argument("--total_workers", type=int, default=1, help="Total number of parallel array workers")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum scans to sample per worker")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device} | Worker: [{args.array_id}/{args.total_workers}]")

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
    model = AnatomyConditionalSynthesisNet(
        spatial_dims=3,
        latent_channels=4,
        label_nc=5,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=2,
        num_head_channels=64,
    ).to(device)

    if not os.path.isfile(args.diffusion_checkpoint):
        raise FileNotFoundError(
            f"CRITICAL: Diffusion checkpoint not found at: {args.diffusion_checkpoint}. "
            "Aborting to prevent uninitialized random-weight generation."
        )

    ckpt_diff = torch.load(args.diffusion_checkpoint, map_location=device)
    state_dict_diff = ckpt_diff["model"] if "model" in ckpt_diff else ckpt_diff
    state_dict_diff = {k.replace("module.", ""): v for k, v in state_dict_diff.items()}
    model.load_state_dict(state_dict_diff)
    model.eval()

    sampler = DDIMSampler(timesteps=1000).to(device)

    # 3. Inference Loop
    for idx, ct_path in enumerate(my_files, start=1):
        filename = os.path.basename(ct_path)
        pair_name = filename.replace("_reg.nii.gz", "")
        out_path = os.path.join(args.out_dir, f"{pair_name}_synthetic_contrast.nii.gz")

        if os.path.exists(out_path):
            log.info(f"[{idx}/{len(my_files)}] Skipping existing: {pair_name}")
            continue

        log.info(f"[{idx}/{len(my_files)}] Generating synthetic contrast volume for: {pair_name}")

        try:
            # Extract subject ID for anatomy conditioning and true NCCT resolution
            subject_p0 = pair_name.split("--")[0]

            # Resolve true native NCCT scan (preventing target leakage from registered contrast)
            ncct_input_file = None
            if args.ncct_dir and os.path.exists(args.ncct_dir):
                cands = [
                    os.path.join(args.ncct_dir, f"{subject_p0}.nii.gz"),
                    os.path.join(args.ncct_dir, subject_p0, "ct.nii.gz")
                ]
                for c in cands:
                    if os.path.exists(c):
                        ncct_input_file = c
                        break

            if ncct_input_file is None:
                log.warning(f"Native NCCT file for {subject_p0} not found in {args.ncct_dir}. Using registered pair baseline.")
                ncct_input_file = ct_path

            # Preprocess volume to normalised range [-0.5, 1.0]
            ncct_tensor = preprocess_ct(ncct_input_file).to(device)

            anatomy_mask = load_anatomy_mask(
                subject_id=subject_p0,
                mask_dirs=args.mask_dirs,
                target_shape=(128, 128, 128),
                latent_shape=(16, 16, 16),
                device=device,
            )

            with torch.no_grad():
                # Encode NCCT to latent space: (1, 4, 16, 16, 16)
                z_ncct = ae.encode(ncct_tensor)

                # Sample target latent via DDIM
                z_syn = sampler.sample(
                    model=model,
                    source_latent=z_ncct,
                    anatomy_mask=anatomy_mask,
                    num_inference_steps=args.num_steps,
                    eta=0.0,
                )

                # Decode latent to full 3D CT volume (1, 1, 128, 128, 128)
                fake_vol = ae.decode(z_syn)

                # Denormalize back to real Hounsfield Units [-1000, 2000]
                fake_hu = denormalise_hu(fake_vol).squeeze().cpu().numpy()
                fake_hu = np.clip(fake_hu, -1000.0, 2000.0).astype(np.int16)

                # Save as NIfTI
                affine = np.diag([1.5, 1.5, 1.5, 1.0])
                nii = nib.Nifti1Image(fake_hu, affine)
                nib.save(nii, out_path)
                log.info(f"  [+] Saved synthetic CT volume -> {out_path}")

        except Exception as e:
            log.error(f"Failed processing {pair_name}: {e}")

    log.info(f"Worker {args.array_id} completed all assigned inference scans!")


if __name__ == "__main__":
    main()
