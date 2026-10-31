"""
code/project10/common_spatial.py
================================
Unified spatial preprocessing for Project 10 (Diffusion Synthesis, Evaluation, Turing Tests).
Guarantees training, sampling, evaluation, and figures share identical resampling and cropping transforms.
"""

import os
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import SimpleITK as sitk
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
)


def get_common_spatial_transform(
    keys=("image",),
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    modes=("bilinear",),
    cvals=(-1000.0,),
) -> Compose:
    """Build standardized MONAI transform pipeline matching MultiPhasePairedDataset."""
    t_list = [
        Spacingd(keys=keys, pixdim=spacing, mode=modes, allow_missing_keys=True),
    ]
    for k, v in zip(keys, cvals):
        t_list.append(
            ResizeWithPadOrCropd(
                keys=[k],
                spatial_size=target_shape,
                mode="constant",
                value=v,
                allow_missing_keys=True,
            )
        )
    return Compose(t_list)


def preprocess_single_volume(
    volume_path: str,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    is_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a single 3D NIfTI/NRRD volume using the unified MONAI spatial transform.
    Returns:
        (preprocessed_array, resampled_affine_4x4)
    """
    mode = "nearest" if is_mask else "bilinear"
    cval = 0.0 if is_mask else -1000.0
    t = Compose([
        LoadImaged(keys=["img"], image_only=False),
        EnsureChannelFirstd(keys=["img"]),
        Spacingd(keys=["img"], pixdim=spacing, mode=(mode,)),
        ResizeWithPadOrCropd(keys=["img"], spatial_size=target_shape, mode="constant", value=cval),
    ])
    out = t({"img": volume_path})
    arr = out["img"][0].numpy()
    affine = out["img"].meta["affine"].numpy() if "affine" in out["img"].meta else np.eye(4)
    if is_mask:
        arr = np.round(arr).astype(np.uint8)
    else:
        arr = np.clip(arr, -1000.0, 2000.0).astype(np.float32)
    return arr, affine
