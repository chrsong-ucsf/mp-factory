#!/usr/bin/env python3
"""
code/project10/models/autoencoder_3d.py
=======================================
Phase 3 — Stage 1: 3D Compression Autoencoder for Multi-Phase CT Synthesis

Compresses full-resolution 3D CT volumes (512×512×Z at 1.5mm isotropic,
resampled to 128×128×128 for GPU) into a compact latent space using a
VQ-VAE or AutoencoderKL-style encoder-decoder.

Architecture choices
--------------------
  - 8× spatial compression: 128³ → 16³ latent (sufficient to represent
    organ-level contrast dynamics while fitting 48GB L40S VRAM)
  - 4 latent channels (captures multi-phase contrast spectrum)
  - MONAI AutoEncoder backbone with configurable VQ or KL regularisation
  - Perceptual loss via a frozen 3D VGG-style feature extractor
  - Training: AdamW, cosine LR, AMP, DDP-ready

Usage
-----
  python code/project10/models/autoencoder_3d.py --help

  python code/project10/training/train_stage1_autoencoder.py \
      --data_root CancerVerse/CancerVerse \
      --out_dir   results/project10/checkpoints/stage1_ae \
      --epochs    100

Architecture diagram
--------------------
  CT volume (1, 128, 128, 128) float32 HU
        │
        ▼  Encoder (strides 2×2×2, depth 4)
  Latent z (4, 16, 16, 16) float32
        │
        ▼  VQ / KL regulariser
  Quantised / sampled ẑ (4, 16, 16, 16)
        │
        ▼  Decoder (strides 2×2×2, depth 4)
  Reconstructed CT (1, 128, 128, 128) float32 HU
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import AutoEncoder, VQVAE
    _MONAI_AE_AVAILABLE = True
except ImportError:
    _MONAI_AE_AVAILABLE = False
    warnings.warn("MONAI AutoEncoder / VQVAE not available; using built-in lightweight AE.")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class AutoEncoder3DConfig:
    """Hyperparameters for the 3D compression autoencoder."""
    # Spatial
    input_size: Tuple[int, int, int] = (128, 128, 128)  # after resampling
    in_channels: int = 1             # single-channel CT (HU)
    latent_channels: int = 4         # channels in compressed latent space
    compression_factor: int = 8      # spatial 8× downsampling (e.g. 128 → 16)

    # Architecture
    channels: Tuple[int, ...] = (32, 64, 128, 256)
    strides: Tuple[int, ...] = (2, 2, 2, 1)   # 3 stride-2 → 8× compression
    use_vq: bool = True              # True = VQ-VAE, False = vanilla AE
    num_embeddings: int = 512        # VQ codebook size
    embedding_dim: int = 4           # must match latent_channels

    # Training
    lr: float = 1e-4
    weight_decay: float = 1e-5
    lambda_perceptual: float = 0.1
    lambda_vq: float = 1.0          # VQ commitment loss weight
    recon_loss: str = "l1"          # "l1" or "l2"

    # Data
    hu_clip_min: float = -1000.0
    hu_clip_max: float = +2000.0
    hu_norm_scale: float = 2000.0   # normalise HU to roughly [-0.5, 1.0]


# ---------------------------------------------------------------------------
# Lightweight built-in fallback (when MONAI nets not found)
# ---------------------------------------------------------------------------
class _ResBlock3D(nn.Module):
    """3D residual block with GroupNorm."""
    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(min(num_groups, channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(num_groups, channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class _DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 4 if stride == 2 else 3,
                              stride=stride, padding=1, bias=False)
        self.res  = _ResBlock3D(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.conv(x))


class _UpBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_ch, out_ch, 4 if stride == 2 else 3,
                                       stride=stride, padding=1, bias=False)
        self.res  = _ResBlock3D(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.conv(x))


class LightweightAutoEncoder3D(nn.Module):
    """
    Lightweight 3D autoencoder with configurable depth and VQ bottleneck.
    Serves as a fallback when MONAI VQVAE is unavailable.
    """

    def __init__(self, cfg: AutoEncoder3DConfig):
        super().__init__()
        self.cfg = cfg
        chs = cfg.channels
        strides = cfg.strides

        # Encoder
        enc_layers: List[nn.Module] = [nn.Conv3d(cfg.in_channels, chs[0], 3, padding=1)]
        for i in range(len(chs) - 1):
            enc_layers.append(_DownBlock3D(chs[i], chs[i + 1], stride=strides[i]))
        enc_layers.append(nn.Conv3d(chs[-1], cfg.latent_channels, 1))
        self.encoder = nn.Sequential(*enc_layers)

        # VQ codebook (optional)
        self.use_vq = cfg.use_vq
        if cfg.use_vq:
            self.vq_embed = nn.Embedding(cfg.num_embeddings, cfg.embedding_dim)
            nn.init.uniform_(self.vq_embed.weight, -1 / cfg.num_embeddings, 1 / cfg.num_embeddings)

        # Decoder
        dec_layers: List[nn.Module] = [nn.Conv3d(cfg.latent_channels, chs[-1], 1)]
        for i in range(len(chs) - 1, 0, -1):
            dec_layers.append(_UpBlock3D(chs[i], chs[i - 1], stride=strides[i - 1]))
        dec_layers.append(nn.Conv3d(chs[0], cfg.in_channels, 3, padding=1))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _vq(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vector quantisation with straight-through gradient estimator."""
        # z: (B, C, H, W, D) -> flatten to (N, C)
        B, C, H, W, D = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)
        # Distances to codebook
        dist = (
            (z_flat ** 2).sum(1, keepdim=True)
            - 2 * z_flat @ self.vq_embed.weight.T
            + (self.vq_embed.weight ** 2).sum(1)
        )
        ids = dist.argmin(1)
        z_q = self.vq_embed(ids).reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)
        # Commitment loss
        vq_loss = F.mse_loss(z_q.detach(), z) + 0.25 * F.mse_loss(z_q, z.detach())
        # Straight-through
        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        recon    : reconstructed CT (same shape as x)
        aux_loss : VQ commitment loss (or 0.0 if not using VQ)
        """
        z = self.encode(x)
        if self.use_vq:
            z, vq_loss = self._vq(z)
        else:
            vq_loss = torch.tensor(0.0, device=x.device)
        recon = self.decode(z)
        return recon, vq_loss


# ---------------------------------------------------------------------------
# MONAI-backed wrapper (preferred when available)
# ---------------------------------------------------------------------------
class MONAI_VQVAE3D(nn.Module):
    """
    Thin wrapper around monai.networks.nets.VQVAE providing the same
    forward() signature as LightweightAutoEncoder3D.
    """

    def __init__(self, cfg: AutoEncoder3DConfig):
        super().__init__()
        if not _MONAI_AE_AVAILABLE:
            raise ImportError("MONAI VQVAE not available. Use LightweightAutoEncoder3D instead.")
        self.cfg = cfg
        n_down = len([s for s in cfg.strides if s == 2])
        self.vqvae = VQVAE(
            spatial_dims=3,
            in_channels=cfg.in_channels,
            out_channels=cfg.in_channels,
            channels=list(cfg.channels),
            num_res_layers=2,
            num_res_channels=list(cfg.channels),
            downsample_parameters=tuple((2, 4, 1, 1) for _ in range(n_down)),
            upsample_parameters=tuple((2, 4, 1, 1, 0) for _ in range(n_down)),
            num_embeddings=cfg.num_embeddings,
            embedding_dim=cfg.embedding_dim,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.vqvae.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vqvae.decode(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        recon, vq_loss, _ = self.vqvae(x)
        return recon, vq_loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_autoencoder(cfg: Optional[AutoEncoder3DConfig] = None) -> nn.Module:
    """
    Returns the best available 3D autoencoder for the given config.
    Prefers MONAI VQVAE; falls back to LightweightAutoEncoder3D.
    """
    if cfg is None:
        cfg = AutoEncoder3DConfig()
    if _MONAI_AE_AVAILABLE and cfg.use_vq:
        try:
            return MONAI_VQVAE3D(cfg)
        except Exception as e:
            warnings.warn(f"MONAI VQVAE3D init failed ({e}); using fallback.")
    return LightweightAutoEncoder3D(cfg)


# ---------------------------------------------------------------------------
# HU normalisation helpers
# ---------------------------------------------------------------------------
def normalise_hu(
    ct: torch.Tensor,
    clip_min: float = -1000.0,
    clip_max: float = 2000.0,
    scale: float = 2000.0,
) -> torch.Tensor:
    """Clip and normalise HU to approx [-0.5, 1.0]."""
    return (ct.clamp(clip_min, clip_max) - clip_min) / scale - 0.5


def denormalise_hu(
    normed: torch.Tensor,
    clip_min: float = -1000.0,
    scale: float = 2000.0,
) -> torch.Tensor:
    """Reverse normalise_hu."""
    return (normed + 0.5) * scale + clip_min


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------
def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mode: str = "l1",
) -> torch.Tensor:
    if mode == "l1":
        return F.l1_loss(recon, target)
    elif mode == "l2":
        return F.mse_loss(recon, target)
    else:
        raise ValueError(f"Unknown recon loss mode: {mode}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing LightweightAutoEncoder3D (built-in fallback)...")
    cfg = AutoEncoder3DConfig(
        input_size=(32, 32, 32),
        channels=(8, 16, 32, 32),
        strides=(2, 2, 2, 1),
        num_embeddings=64,
        embedding_dim=4,
        use_vq=True,
    )
    model = LightweightAutoEncoder3D(cfg)
    x = torch.randn(1, 1, 32, 32, 32)
    recon, vq_loss = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Latent: {model.encode(x).shape}")
    print(f"  Recon:  {recon.shape}")
    print(f"  VQ loss: {vq_loss.item():.4f}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    print("\nTesting build_autoencoder factory...")
    ae = build_autoencoder(cfg)
    recon2, vq2 = ae(x)
    print(f"  Factory model: {ae.__class__.__name__}, recon shape: {recon2.shape}")
    print("Smoke test passed.")
