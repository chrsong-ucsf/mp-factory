#!/usr/bin/env python3
"""
code/project10/training/train_stage2_synthesis.py
=================================================
Phase 3 — Stage 2: Anatomy-Conditional Latent Diffusion Synthesis Training

Trains AnatomyConditionalSynthesisNet (SPADEDiffusionModelUNet) to synthesize
target contrast phase latents (arterial/venous) from baseline NCCT source latents,
spatially conditioned on the Project 9 5-organ GI segmentation mask.

Pipeline
--------
  1. Load and validate frozen Stage 1 continuous AutoEncoder.
  2. Load co-registered multi-phase pairs via MultiphasePairedDataset:
       - Source CT (NCCT/P0 phase)
       - Registered target CT (Arterial / Portal-Venous)
       - Organ mask (5-class TotalSegmentator / Project 9 prior)
  3. Encode CT volumes into latent space using frozen Stage 1 AutoEncoder:
       - z_ncct   = AE.encode(ncct)    -> (B, 4, 16, 16, 16)
       - z_target = AE.encode(target)  -> (B, 4, 16, 16, 16)
  4. Prepare SPADE condition mask:
       - Convert mask (0..4) to one-hot (B, 5, 128, 128, 128)
       - Downsample to latent resolution -> (B, 5, 16, 16, 16)
  5. Forward Diffusion:
       - Sample random diffusion timestep t in [0, T-1]
       - Add scheduled Gaussian noise: z_t = sqrt(alpha_bar_t)*z_target + sqrt(1-alpha_bar_t)*eps
       - Predict noise: eps_pred = SynthesisNet(z_t, t, z_ncct, mask_down)
       - Loss = MSE(eps_pred, eps)
  6. Save audited checkpoints with complete resume state and conditioning identity.

Usage
-----
  python code/project10/training/train_stage2_synthesis.py \\
      --registration_csv results/project10/manifests/registration_log.csv \\
      --ae_checkpoint    results/project10/checkpoints/stage1_ae/best_stage1_ae.pt \\
      --out_dir          results/project10/checkpoints/stage2_synthesis \\
      --epochs           100 --batch_size 4 --lr 1e-4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Project root setup
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from code.project10.datasets.multiphase_paired_dataset import MultiphasePairedDataset
from code.project10.models.autoencoder_3d import (
    AutoEncoder3DConfig,
    build_autoencoder,
)
from code.project10.models.synthesis_net import AnatomyConditionalSynthesisNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint and Provenance Helpers
# ---------------------------------------------------------------------------
def checkpoint_sha256(path: Path | str) -> str:
    """Calculate SHA256 hex digest of a checkpoint file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def mask_conditioning_identity(totalseg_task: str, base_dir: Path | str, mask_dir: Path | str) -> dict:
    """Construct canonical identity metadata for spatial mask conditioning."""
    base = Path(base_dir).resolve()
    m_dir = Path(mask_dir)
    resolved_mask_dir = m_dir.resolve() if m_dir.is_absolute() else (base / m_dir).resolve()
    return {
        "totalseg_task": totalseg_task,
        "mask_dir": str(resolved_mask_dir),
    }


def load_stage1_autoencoder(path: Path | str, device: torch.device) -> nn.Module:
    """
    Load a frozen Stage 1 continuous 3D AutoEncoder.
    Rejects missing files, discrete VQ-VAE checkpoints, and partial/incompatible weights.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stage 1 AutoEncoder checkpoint not found at {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    if cfg_dict.get("use_vq", False):
        raise ValueError("Stage 1 checkpoint must be continuous (use_vq=False required for latent diffusion Stage 2)")

    cfg = AutoEncoder3DConfig(**cfg_dict) if cfg_dict else AutoEncoder3DConfig(use_vq=False)
    ae = build_autoencoder(cfg).to(device)
    try:
        ae.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Checkpoint model weights are incompatible: {e}") from e

    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False
    return ae


def stage2_checkpoint_payload(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_val_loss: float,
    ae_checkpoint_sha256: str,
    conditioning: dict,
) -> dict:
    """Build complete checkpoint payload with training state and provenance."""
    raw_model = model.module if hasattr(model, "module") else model
    return {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "ae_checkpoint_sha256": ae_checkpoint_sha256,
        "conditioning": conditioning,
    }


def restore_stage2_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    total_epochs: int,
    ae_checkpoint_sha256: str,
    conditioning: dict,
    allow_legacy_ae_resume: bool = False,
) -> Tuple[int, float]:
    """
    Restore Stage 2 training state from checkpoint with strict provenance auditing.
    Returns (epoch, best_val_loss).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found at {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # 1. AE Identity Validation
    ckpt_ae_sha = ckpt.get("ae_checkpoint_sha256")
    if ckpt_ae_sha is not None:
        if ckpt_ae_sha != ae_checkpoint_sha256:
            raise ValueError(
                f"Checkpoint {path} was trained with a different Stage 1 AE ({ckpt_ae_sha} vs {ae_checkpoint_sha256})"
            )
    else:
        if not allow_legacy_ae_resume:
            raise ValueError(
                f"Legacy checkpoint {path} lacks Stage 1 AE identity. "
                "Explicit override required via --allow_legacy_ae_resume."
            )

    # 2. Conditioning Identity Validation
    ckpt_cond = ckpt.get("conditioning")
    if ckpt_cond is not None:
        if ckpt_cond != conditioning:
            raise ValueError(
                f"Checkpoint {path} was trained with different mask conditioning: {ckpt_cond} vs {conditioning}"
            )
    else:
        if not allow_legacy_ae_resume:
            raise ValueError(
                f"Legacy checkpoint {path} has unaudited mask conditioning. "
                "Explicit override required via --allow_legacy_ae_resume."
            )

    # 3. Model weights
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model"], strict=True)

    # 4. Epoch
    epoch = ckpt.get("epoch", 0)

    # 5. Optimizer
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    # 6. Scheduler & LR state
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        scheduler.last_epoch = epoch
        base_lr = scheduler.base_lrs[0] if hasattr(scheduler, "base_lrs") else optimizer.param_groups[0]["lr"]
        eta_min = getattr(scheduler, "eta_min", 1e-6)
        expected_lr = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
        for pg in optimizer.param_groups:
            pg["lr"] = expected_lr

    # 7. Best validation loss
    if "best_val_loss" in ckpt:
        best_val_loss = float(ckpt["best_val_loss"])
    else:
        best_path = path.parent / "best_synthesis_net.pt"
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            best_val_loss = float(best_ckpt.get("val_loss", best_ckpt.get("best_val_loss", float("inf"))))
        else:
            best_val_loss = float("inf")

    return epoch, best_val_loss


# ---------------------------------------------------------------------------
# Diffusion Noise Schedule Helpers
# ---------------------------------------------------------------------------
class DiffusionSchedule:
    """Standard linear beta schedule for DDPM."""
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def to(self, device: torch.device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample q(x_t | x_0) given clean latent, timesteps, and random Gaussian noise."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 10 Stage 2: Anatomy-Conditional Diffusion Synthesis")
    p.add_argument("--registration_csv", type=str,
                   default="results/project10/manifests/registration_log.csv",
                   help="Path to registration_log.csv")
    p.add_argument("--base_dir", type=str,
                   default="/mnt/scratch/user/chrsong/mp-factory",
                   help="Base repository directory")
    p.add_argument("--mask_dir", type=str,
                   default="results/totalseg_masks",
                   help="Directory containing organ masks")
    p.add_argument("--totalseg_task", type=str, default="gi",
                   choices=["gi", "total", "total_v1"],
                   help="TotalSegmentator mask label scheme ('gi', 'total', 'total_v1')")
    p.add_argument("--ae_checkpoint", type=str,
                   default="results/project10/checkpoints/stage1_ae/best_stage1_ae.pt",
                   help="Path to Stage 1 frozen autoencoder checkpoint")
    p.add_argument("--out_dir", type=str,
                   default="results/project10/checkpoints/stage2_synthesis",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--val_interval", type=int, default=5)
    p.add_argument("--resume", type=str, default="",
                   help="Path to Stage 2 checkpoint to resume from")
    p.add_argument("--allow_legacy_ae_resume", action="store_true",
                   help="Allow resuming from legacy checkpoints without recorded AE SHA256 or conditioning identity")
    return p.parse_args()


# ---------------------------------------------------------------------------
# DDP Setup
# ---------------------------------------------------------------------------
def setup_ddp():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, is_ddp


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_master = (rank == 0)

    os.makedirs(args.out_dir, exist_ok=True)

    if is_master:
        log.info("=" * 60)
        log.info("Project 10 Stage 2: Anatomy-Conditional Synthesis Training")
        log.info(f"  Registration CSV: {args.registration_csv}")
        log.info(f"  Base Dir:         {args.base_dir}")
        log.info(f"  Stage 1 AE Ckpt:  {args.ae_checkpoint}")
        log.info(f"  Mask Dir:         {args.mask_dir}")
        log.info(f"  TotalSeg Task:    {args.totalseg_task}")
        log.info(f"  Output Dir:       {args.out_dir}")
        log.info(f"  Epochs:           {args.epochs}")
        log.info(f"  Batch size:       {args.batch_size} (effective: {args.batch_size * world_size})")
        log.info(f"  Device:           {device}")
        log.info("=" * 60)

    # 1. Stage 1 Autoencoder (Frozen Latent Compressor) — validate FIRST
    ae = load_stage1_autoencoder(args.ae_checkpoint, device)
    ae_sha = checkpoint_sha256(args.ae_checkpoint)
    conditioning = mask_conditioning_identity(args.totalseg_task, args.base_dir, args.mask_dir)
    if is_master:
        log.info(f"Loaded Stage 1 AE (SHA256: {ae_sha[:12]}...)")

    # 2. Dataset & DataLoaders
    full_ds = MultiphasePairedDataset(
        registration_csv=args.registration_csv,
        base_dir=args.base_dir,
        mask_dir=args.mask_dir,
        target_shape=(128, 128, 128),
        spacing_mm=1.5,
        totalseg_task=args.totalseg_task,
    )
    n_total = len(full_ds)
    if n_total == 0:
        log.error(f"No pairs found in {args.registration_csv}. Ensure registration array has run.")
        if is_ddp:
            dist.destroy_process_group()
        return

    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if is_ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if is_master:
        log.info(f"Loaded {n_total} pairs ({n_train} train, {n_val} val).")

    # 3. Stage 2 Generative Model: AnatomyConditionalSynthesisNet
    model = AnatomyConditionalSynthesisNet(
        spatial_dims=3,
        latent_channels=4,
        label_nc=5,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=2,
        num_head_channels=64,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_val_loss = restore_stage2_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            total_epochs=args.epochs,
            ae_checkpoint_sha256=ae_sha,
            conditioning=conditioning,
            allow_legacy_ae_resume=args.allow_legacy_ae_resume,
        )
        if is_master:
            log.info(f"Resumed from {args.resume} (epoch {start_epoch}, best_val_loss {best_val_loss:.5f})")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    diffusion = DiffusionSchedule(timesteps=args.timesteps).to(device)
    scaler = GradScaler()

    # 4. CSV Logging Setup
    log_csv = os.path.join(args.out_dir, "stage2_training_log.csv")
    if is_master and not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "val_loss", "lr", "elapsed_min",
            ]).writeheader()

    t_start = time.time()

    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss_acc = 0.0
        n_batches = 0

        for batch in train_loader:
            ncct = batch["ncct"].to(device, non_blocking=True)      # (B, 1, 128, 128, 128)
            target = batch["target"].to(device, non_blocking=True)  # (B, 1, 128, 128, 128)
            mask = batch["mask"].to(device, non_blocking=True)      # (B, 1, 128, 128, 128)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                # Encode into latent space: (B, 4, 16, 16, 16)
                z_ncct = ae.encode(ncct)
                z_target = ae.encode(target)

                # Prepare one-hot condition mask downsampled to latent grid
                mask_one_hot = F.one_hot(mask.squeeze(1).long(), num_classes=5).permute(0, 4, 1, 2, 3).float()
                mask_down = F.interpolate(mask_one_hot, size=z_target.shape[2:], mode="nearest")

            # Sample diffusion timestep
            t = torch.randint(0, diffusion.timesteps, (z_target.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(z_target)
            z_t = diffusion.q_sample(z_target, t, noise)

            with autocast():
                eps_pred = model(z_t, t, z_ncct, mask_down)
                loss = F.mse_loss(eps_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_acc += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_acc / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        # Validation
        val_loss = float("nan")
        if (epoch + 1) % args.val_interval == 0 or (epoch + 1) == args.epochs:
            model.eval()
            val_loss_acc = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    ncct = batch["ncct"].to(device)
                    target = batch["target"].to(device)
                    mask = batch["mask"].to(device)

                    z_ncct = ae.encode(ncct)
                    z_target = ae.encode(target)
                    mask_one_hot = F.one_hot(mask.squeeze(1).long(), num_classes=5).permute(0, 4, 1, 2, 3).float()
                    mask_down = F.interpolate(mask_one_hot, size=z_target.shape[2:], mode="nearest")

                    t = torch.randint(0, diffusion.timesteps, (z_target.shape[0],), device=device, dtype=torch.long)
                    noise = torch.randn_like(z_target)
                    z_t = diffusion.q_sample(z_target, t, noise)

                    with autocast():
                        eps_pred = model(z_t, t, z_ncct, mask_down)
                        v_loss = F.mse_loss(eps_pred, noise)

                    val_loss_acc += v_loss.item()
                    n_val_batches += 1

            val_loss = val_loss_acc / max(n_val_batches, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.out_dir, "best_synthesis_net.pt")
                payload = stage2_checkpoint_payload(
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val_loss=best_val_loss,
                    ae_checkpoint_sha256=ae_sha,
                    conditioning=conditioning,
                )
                torch.save(payload, best_path)
                log.info(f"  [Epoch {epoch+1:03d}] New best val_loss={val_loss:.5f} -> saved {best_path}")

        elapsed = (time.time() - t_start) / 60.0
        if is_master:
            log.info(
                f"[Epoch {epoch+1:03d}/{args.epochs}] "
                f"train_loss={avg_train_loss:.5f} val_loss={val_loss:.5f} "
                f"lr={lr_now:.2e} elapsed={elapsed:.1f}min"
            )
            with open(log_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=[
                    "epoch", "train_loss", "val_loss", "lr", "elapsed_min",
                ]).writerow({
                    "epoch": epoch + 1,
                    "train_loss": round(avg_train_loss, 6),
                    "val_loss": round(val_loss, 6) if not math.isnan(val_loss) else "",
                    "lr": f"{lr_now:.2e}",
                    "elapsed_min": round(elapsed, 2),
                })

        # Periodic checkpoint
        if is_master and (epoch + 1) % 10 == 0:
            last_path = os.path.join(args.out_dir, "last_synthesis_net.pt")
            payload = stage2_checkpoint_payload(
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
                ae_checkpoint_sha256=ae_sha,
                conditioning=conditioning,
            )
            torch.save(payload, last_path)

    if is_master:
        log.info(f"Stage 2 Synthesis Training Complete. Best val_loss = {best_val_loss:.5f}")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
