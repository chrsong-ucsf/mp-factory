#!/usr/bin/env python3
"""
code/project10/training/train_stage1_autoencoder.py
===================================================
Phase 3 — Train Stage 1: 3D CT Compression Autoencoder

Trains a VQ-VAE (or lightweight fallback) to compress 128³ CT volumes
to a 16³ × 4-channel latent representation.

Pipeline
--------
  1. Load multiphase_pairs.csv (built by discover_multiphase_cohort.py)
  2. For each subject, load and preprocess ct.nii.gz:
       - Resample to 1.5mm isotropic (SimpleITK)
       - Centre-crop / pad to 128³
       - Clip HU to [-1000, +2000], normalise to [-0.5, 1.0]
  3. Forward pass: encode → VQ → decode → reconstruction loss + VQ loss
  4. Log to CSV and save best checkpoint

Usage
-----
  python code/project10/training/train_stage1_autoencoder.py \
      --pairs   results/project10/manifests/multiphase_pairs.csv \
      --out_dir results/project10/checkpoints/stage1_ae \
      --epochs  100  --batch_size 2  --lr 1e-4

  # DDP multi-GPU (torchrun):
  torchrun --nproc_per_node=4 code/project10/training/train_stage1_autoencoder.py \
      --pairs   results/project10/manifests/multiphase_pairs.csv \
      --out_dir results/project10/checkpoints/stage1_ae \
      --epochs  100  --batch_size 2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import pandas as pd
import nibabel as nib

# Project10 modules
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from code.project10.models.autoencoder_3d import (
    AutoEncoder3DConfig,
    build_autoencoder,
    normalise_hu,
    denormalise_hu,
    reconstruction_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimpleITK resampler
# ---------------------------------------------------------------------------
def _resample_to_isotropic(ct_path: str, spacing_mm: float = 1.5) -> np.ndarray:
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(ct_path, sitk.sitkFloat32)
        orig_spacing = img.GetSpacing()
        orig_size    = img.GetSize()
        new_spacing  = (spacing_mm, spacing_mm, spacing_mm)
        new_size     = tuple(
            int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
            for i in range(3)
        )
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000.0)
        resampled = resampler.Execute(img)
        return sitk.GetArrayFromImage(resampled).transpose(2, 1, 0)  # (X, Y, Z)
    except ImportError:
        # Fallback: load as-is
        img = nib.load(ct_path)
        return img.get_fdata().astype(np.float32)


def _center_crop_pad(
    volume: np.ndarray, target: Tuple[int, int, int] = (128, 128, 128)
) -> np.ndarray:
    """Centre-crop or zero-pad each axis to target size."""
    result = np.full(target, -1000.0, dtype=np.float32)
    src_shape = volume.shape
    for ax, (s, t) in enumerate(zip(src_shape, target)):
        if s >= t:
            start = (s - t) // 2
            volume = np.take(volume, range(start, start + t), axis=ax)
        else:
            pad = t - s
            pad_before = pad // 2
            pads = [(0, 0)] * 3
            pads[ax] = (pad_before, pad - pad_before)
            volume = np.pad(volume, pads, mode="constant", constant_values=-1000.0)
    result = volume.astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MultiPhaseVolumeDataset(Dataset):
    """
    Loads unique subjects from multiphase_pairs.csv (all phases together,
    deduplicated on subject_p0 + subject_target).
    For Stage 1 autoencoder training, we use all CT volumes regardless of phase.
    """

    def __init__(
        self,
        pairs_csv: str,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        spacing_mm: float = 1.5,
        split: str = "train",        # "train" or "val"
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        df = pd.read_csv(pairs_csv)
        # Collect unique CT paths from both P0 and target columns
        paths_p0  = df[["subject_p0",     "ct_path_p0"]].rename(
            columns={"subject_p0": "subject_id", "ct_path_p0": "ct_path"})
        paths_tgt = df[["subject_target", "ct_path_target"]].rename(
            columns={"subject_target": "subject_id", "ct_path_target": "ct_path"})
        all_paths = pd.concat([paths_p0, paths_tgt], ignore_index=True)
        all_paths = all_paths.drop_duplicates("ct_path").reset_index(drop=True)

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(all_paths))
        n_val = max(1, int(len(indices) * val_fraction))
        if split == "val":
            indices = indices[:n_val]
        else:
            indices = indices[n_val:]

        self.records = all_paths.iloc[indices].reset_index(drop=True)
        self.target_shape = target_shape
        self.spacing_mm   = spacing_mm
        log.info(f"  {split} set: {len(self.records)} volumes")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records.iloc[idx]
        ct_path = row["ct_path"]

        try:
            vol = _resample_to_isotropic(ct_path, self.spacing_mm)
            vol = _center_crop_pad(vol, self.target_shape)
        except Exception as e:
            log.warning(f"Failed to load {ct_path}: {e}. Returning blank volume.")
            vol = np.full(self.target_shape, -1000.0, dtype=np.float32)

        # Add channel dim; normalise
        vol_t = torch.from_numpy(vol).unsqueeze(0).float()      # (1, H, W, D)
        vol_t = normalise_hu(vol_t, -1000.0, 2000.0, 2000.0)

        return {
            "ct":         vol_t,
            "subject_id": str(row["subject_id"]),
            "ct_path":    str(ct_path),
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Project 10 Stage 1: Train 3D Compression Autoencoder")
    p.add_argument("--pairs",      type=str, required=True,
                   help="Path to multiphase_pairs.csv")
    p.add_argument("--out_dir",    type=str, default="results/project10/checkpoints/stage1_ae",
                   help="Output directory for checkpoints and logs")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lambda_vq",  type=float, default=1.0)
    p.add_argument("--val_interval", type=int, default=5,
                   help="Run validation every N epochs")
    p.add_argument("--target_size", type=int, default=128,
                   help="Isotropic spatial dimension after resampling")
    p.add_argument("--spacing_mm",  type=float, default=1.5,
                   help="Target isotropic voxel spacing (mm)")
    p.add_argument("--start_epoch", type=int, default=0,
                   help="Resume from checkpoint epoch")
    p.add_argument("--checkpoint",  type=str, default="",
                   help="Path to a checkpoint .pt file to resume from")
    p.add_argument("--no_vq",       action="store_true",
                   help="Use plain AE without vector quantisation")
    return p.parse_args()


def setup_ddp():
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_ddp = world_size > 1
    if is_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, is_ddp


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_master = (rank == 0)

    os.makedirs(args.out_dir, exist_ok=True)
    target_shape = (args.target_size, args.target_size, args.target_size)

    if is_master:
        log.info(f"Project 10 Stage 1 — 3D CT Compression Autoencoder Training")
        log.info(f"  pairs:       {args.pairs}")
        log.info(f"  out_dir:     {args.out_dir}")
        log.info(f"  epochs:      {args.epochs}")
        log.info(f"  batch_size:  {args.batch_size} × {world_size} workers = {args.batch_size * world_size} effective")
        log.info(f"  device:      {device}")

    # ------------------------------------------------------------------
    # Data
    train_ds = MultiPhaseVolumeDataset(args.pairs, target_shape, args.spacing_mm, "train")
    val_ds   = MultiPhaseVolumeDataset(args.pairs, target_shape, args.spacing_mm, "val")

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if is_ddp else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # ------------------------------------------------------------------
    # Model
    cfg = AutoEncoder3DConfig(
        input_size=target_shape,
        use_vq=not args.no_vq,
        latent_channels=4,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2, 1),
    )
    cfg.lr           = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.lambda_vq    = args.lambda_vq

    model = build_autoencoder(cfg).to(device)
    if is_master:
        params = sum(p.numel() for p in model.parameters())
        log.info(f"  Model: {model.__class__.__name__}, parameters: {params:,}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        args.start_epoch = ckpt.get("epoch", 0)
        if is_master:
            log.info(f"  Resumed from checkpoint: {args.checkpoint} (epoch {args.start_epoch})")

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.start_epoch, eta_min=1e-6
    )
    scaler = GradScaler()

    # ------------------------------------------------------------------
    # Log file
    log_csv = os.path.join(args.out_dir, "training_log.csv")
    if is_master and not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=[
                "epoch", "train_recon", "train_vq", "train_total",
                "val_recon", "lr", "elapsed_min",
            ]).writeheader()

    best_val_recon = float("inf")
    t_start = time.time()

    # ------------------------------------------------------------------
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_recon_acc = 0.0
        train_vq_acc    = 0.0
        n_batches = 0

        for batch in train_loader:
            ct = batch["ct"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                recon, vq_loss = model(ct)
                loss_recon = reconstruction_loss(recon, ct, mode="l1")
                loss_total = loss_recon + cfg.lambda_vq * vq_loss

            scaler.scale(loss_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_recon_acc += loss_recon.item()
            train_vq_acc    += vq_loss.item() if isinstance(vq_loss, torch.Tensor) else float(vq_loss)
            n_batches += 1

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        avg_recon = train_recon_acc / max(n_batches, 1)
        avg_vq    = train_vq_acc    / max(n_batches, 1)

        # ------------------------------------------------------------------
        # Validation
        val_recon = float("nan")
        if is_master and (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_acc = 0.0
            n_val   = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vct = vbatch["ct"].to(device, non_blocking=True)
                    with autocast():
                        vrecon, _ = model(vct)
                        val_acc += reconstruction_loss(vrecon, vct, mode="l1").item()
                    n_val += 1
            val_recon = val_acc / max(n_val, 1)

            if val_recon < best_val_recon:
                best_val_recon = val_recon
                ckpt_path = os.path.join(args.out_dir, "best_stage1_ae.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model": (model.module if is_ddp else model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_recon": val_recon,
                    "cfg": vars(cfg),
                }, ckpt_path)
                log.info(f"  [Epoch {epoch+1}] New best val_recon={val_recon:.4f} → saved {ckpt_path}")

        # ------------------------------------------------------------------
        # Log
        elapsed = (time.time() - t_start) / 60.0
        if is_master:
            log.info(
                f"[Epoch {epoch+1:03d}/{args.epochs}] "
                f"recon={avg_recon:.4f} vq={avg_vq:.4f} "
                f"val_recon={val_recon:.4f} lr={lr_now:.2e} "
                f"elapsed={elapsed:.1f}min"
            )
            with open(log_csv, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=[
                    "epoch", "train_recon", "train_vq", "train_total",
                    "val_recon", "lr", "elapsed_min",
                ]).writerow({
                    "epoch": epoch + 1,
                    "train_recon": round(avg_recon, 5),
                    "train_vq":    round(avg_vq, 5),
                    "train_total": round(avg_recon + cfg.lambda_vq * avg_vq, 5),
                    "val_recon":   round(val_recon, 5) if not math.isnan(val_recon) else "",
                    "lr":          f"{lr_now:.2e}",
                    "elapsed_min": round(elapsed, 2),
                })

        # Save last checkpoint (every 10 epochs)
        if is_master and (epoch + 1) % 10 == 0:
            ckpt_last = os.path.join(args.out_dir, "last_stage1_ae.pt")
            torch.save({
                "epoch": epoch + 1,
                "model": (model.module if is_ddp else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": vars(cfg),
            }, ckpt_last)

    if is_master:
        log.info(f"\nTraining complete. Best val_recon = {best_val_recon:.4f}")
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
