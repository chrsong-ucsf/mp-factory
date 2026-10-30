"""
code/project10/datasets/multiphase_paired_dataset.py
====================================================
Dataset for Project 10 Phase 3 Stage 2 synthesis.

Reads `results/project10/manifests/registration_log.csv`, filters by status == 'OK',
and yields pairs of (ncct, target, mask).

The output dictionary contains:
    - ncct: Tensor (1, 128, 128, 128)
    - target: Tensor (1, 128, 128, 128)
    - mask: Tensor (1, 128, 128, 128) (organ mask, GI labels 0..4)
    - phase: string (e.g. 'VENOUS')
    - subject_id: string (subject_p0)

Usage:
    from code.project10.datasets.multiphase_paired_dataset import MultiphasePairedDataset
    dataset = MultiphasePairedDataset("results/project10/manifests/registration_log.csv")
    sample = dataset[0]
"""

import csv
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
)

log = logging.getLogger(__name__)


class MultiphasePairedDataset(Dataset):
    """
    Dataset for Project 10 Phase 3 Stage 2 multiphase synthesis.
    
    Reads a registration log CSV and constructs paired instances of:
      - Source CT (NCCT/P0 phase)
      - Registered target CT
      - Organ mask (from TotalSegmentator)
      
    All volumes are resampled to 1.5mm isotropic, padded or center-cropped to 128^3,
    and HU values are scaled to [-0.5, 1.0].
    """
    def __init__(
        self,
        registration_csv: str = "results/project10/manifests/registration_log.csv",
        base_dir: str = "/mnt/scratch/user/chrsong/mp-factory",
        mask_dir: str = "results/totalseg_masks",
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        spacing_mm: float = 1.5,
        totalseg_task: Optional[str] = "gi",
    ):
        """
        Args:
            registration_csv (str): Path to the registration manifest CSV.
            base_dir (str): Base directory of the repository (for resolving absolute paths).
            mask_dir (str): Directory containing TotalSegmentator masks.
            target_shape (tuple): Spatial dimensions for Padding/Cropping.
            spacing_mm (float): Isotropic voxel spacing in mm.
            totalseg_task (str, optional): 'gi', 'total', 'total_v1', or None.
        """
        super().__init__()
        self.base_dir = Path(base_dir).resolve()
        mask_p = Path(mask_dir)
        self.mask_dir = mask_p if mask_p.is_absolute() else (self.base_dir / mask_p).resolve()
        self.target_shape = target_shape
        self.spacing_mm = spacing_mm
        self.totalseg_task = totalseg_task
        
        csv_p = Path(registration_csv)
        csv_path = csv_p if csv_p.is_absolute() else (self.base_dir / csv_p).resolve()
        
        # Load and filter CSV.
        # Column order: pair_id, patient_id, subject_p0, subject_target, phase_target, status, elapsed_sec, error, out_reg_path
        _FIELDNAMES = ["pair_id", "patient_id", "subject_p0", "subject_target",
                       "phase_target", "status", "elapsed_sec", "error", "out_reg_path"]
        try:
            df = pd.read_csv(csv_path, header=None, names=_FIELDNAMES)
            if 'status' in df.columns:
                self.df = df[df['status'] == 'OK'].reset_index(drop=True)
            else:
                log.warning("Column 'status' not found in registration CSV. Using all rows.")
                self.df = df.reset_index(drop=True)
        except Exception as e:
            log.error(f"Failed to load registration CSV from {csv_path}: {e}")
            self.df = pd.DataFrame()

        # Validate rows: reject malformed OK rows missing subject_p0
        for idx, row in self.df.iterrows():
            p0 = str(row.get('subject_p0', '')).strip()
            if not p0 or p0.lower() == 'nan':
                raise ValueError(f"Malformed OK row in {csv_path}: missing subject_p0 at index {idx}")

        # MONAI spatial transform pipeline
        self.spatial_transform = Compose([
            Spacingd(
                keys=["ncct", "target", "mask"], 
                pixdim=(spacing_mm, spacing_mm, spacing_mm), 
                mode=("bilinear", "bilinear", "nearest"), 
                allow_missing_keys=True
            ),
            ResizeWithPadOrCropd(
                keys=["ncct", "target"], 
                spatial_size=target_shape,
                mode="constant",
                value=-1000.0,
                allow_missing_keys=True
            ),
            ResizeWithPadOrCropd(
                keys=["mask"], 
                spatial_size=target_shape,
                mode="constant",
                value=0.0,
                allow_missing_keys=True
            ),
            ScaleIntensityRanged(
                keys=["ncct", "target"], 
                a_min=-1000.0, 
                a_max=2000.0, 
                b_min=-0.5, 
                b_max=1.0, 
                clip=True, 
                allow_missing_keys=True
            )
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        subject_p0 = str(row['subject_p0']).strip() if 'subject_p0' in row.index else f"subject_{idx}"
        phase_target = str(row['phase_target']).strip() if 'phase_target' in row.index else 'UNKNOWN'
        
        # 1. Resolve NCCT path
        ncct_path = self.base_dir / "CancerVerse" / "CancerVerse" / subject_p0 / "ct.nii.gz"
        if not ncct_path.exists():
            ncct_path = self.base_dir / "CancerVerse_dbox" / subject_p0 / "ct.nii.gz"
        if not ncct_path.exists():
            raise FileNotFoundError(f"NCCT CT missing for subject {subject_p0} (checked CancerVerse and CancerVerse_dbox)")

        # 2. Resolve Target path
        target_path_raw = str(row['out_reg_path']).strip() if 'out_reg_path' in row.index else ""
        if not target_path_raw:
            raise FileNotFoundError(f"Target CT missing for subject {subject_p0} (empty out_reg_path)")
        target_p = Path(target_path_raw)
        target_path = target_p if target_p.is_absolute() else (self.base_dir / target_p).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Target CT missing for subject {subject_p0} at {target_path}")

        # 3. Resolve Mask path
        candidate_names = [
            f"{subject_p0}_gi_mask.nii.gz",
            f"{subject_p0}_mask.nii.gz",
            f"{subject_p0}.nii.gz"
        ]
        mask_path = None
        for c_name in candidate_names:
            p = self.mask_dir / c_name
            if p.exists():
                mask_path = p
                break
        if mask_path is None:
            raise FileNotFoundError(f"Mask file missing for subject {subject_p0} in {self.mask_dir}")

        # 4. Load CT volumes & verify finiteness
        try:
            ncct_nii = nib.load(str(ncct_path))
            ncct_arr = np.asarray(ncct_nii.dataobj, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to read NCCT CT for subject {subject_p0}: {e}")
        if not np.all(np.isfinite(ncct_arr)):
            raise RuntimeError(f"Non-finite values (NaN/Inf) detected in NCCT CT for subject {subject_p0}")

        try:
            target_nii = nib.load(str(target_path))
            target_arr = np.asarray(target_nii.dataobj, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to read target CT for subject {subject_p0}: {e}")
        if not np.all(np.isfinite(target_arr)):
            raise RuntimeError(f"Non-finite values (NaN/Inf) detected in target CT for subject {subject_p0}")

        # 5. Load Mask volume & verify integrity
        try:
            mask_nii = nib.load(str(mask_path))
            mask_arr = np.asarray(mask_nii.dataobj)
        except Exception as e:
            raise RuntimeError(f"Corrupt or unreadable mask for subject {subject_p0}: {e}")

        if (mask_arr < 0).any():
            raise ValueError(f"Negative label values found in mask for subject {subject_p0}")

        mask_fname = mask_path.name
        is_full_body = mask_fname.endswith("_mask.nii.gz") and not mask_fname.endswith("_gi_mask.nii.gz")
        if is_full_body and self.totalseg_task is None:
            raise ValueError(f"Full body mask {mask_fname} for subject {subject_p0} requires version provenance (totalseg_task cannot be None)")

        # 6. Map mask labels to 5-class GI scheme (0=bg, 1=stomach, 2=duodenum, 3=small bowel, 4=colon)
        if self.totalseg_task == "total":
            mapped_mask = np.zeros_like(mask_arr, dtype=np.uint8)
            mapped_mask[mask_arr == 6] = 1
            mapped_mask[mask_arr == 19] = 2
            mapped_mask[mask_arr == 18] = 3
            mapped_mask[mask_arr == 20] = 4
        elif self.totalseg_task == "total_v1":
            mapped_mask = np.zeros_like(mask_arr, dtype=np.uint8)
            mapped_mask[mask_arr == 6] = 1
            mapped_mask[mask_arr == 56] = 2
            mapped_mask[mask_arr == 55] = 3
            mapped_mask[mask_arr == 57] = 4
        elif self.totalseg_task == "gi" or self.totalseg_task is None:
            mapped_mask = np.zeros_like(mask_arr, dtype=np.uint8)
            for organ_id in (1, 2, 3, 4):
                mapped_mask[mask_arr == organ_id] = organ_id
        else:
            raise ValueError(f"Unknown totalseg_task '{self.totalseg_task}' for subject {subject_p0}")

        if np.count_nonzero(mapped_mask) == 0:
            raise ValueError(f"No usable GI organs found in mask for subject {subject_p0}")

        # Squeeze leading singleton if 4D
        if ncct_arr.ndim == 4 and ncct_arr.shape[0] == 1:
            ncct_arr = ncct_arr[0]
        if target_arr.ndim == 4 and target_arr.shape[0] == 1:
            target_arr = target_arr[0]
        if mapped_mask.ndim == 4 and mapped_mask.shape[0] == 1:
            mapped_mask = mapped_mask[0]

        # 7. Apply MONAI spatial transforms using MetaTensors to preserve affine
        data_dict = {
            "ncct": MetaTensor(torch.from_numpy(ncct_arr[None, ...]), affine=torch.from_numpy(ncct_nii.affine)),
            "target": MetaTensor(torch.from_numpy(target_arr[None, ...]), affine=torch.from_numpy(target_nii.affine)),
            "mask": MetaTensor(torch.from_numpy(mapped_mask[None, ...].astype(np.float32)), affine=torch.from_numpy(mask_nii.affine)),
        }

        try:
            transformed = self.spatial_transform(data_dict)
            ncct_tensor = torch.as_tensor(transformed["ncct"])
            target_tensor = torch.as_tensor(transformed["target"])
            mask_tensor = torch.as_tensor(transformed["mask"]).round().to(torch.long)

            return {
                "ncct": ncct_tensor,
                "target": target_tensor,
                "mask": mask_tensor,
                "phase": phase_target,
                "subject_id": subject_p0,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to transform sample for subject {subject_p0}: {e}")


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Smoke test for MultiphasePairedDataset")
    parser.add_argument("--csv", type=str, default="results/project10/manifests/registration_log.csv")
    parser.add_argument("--base_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory")
    parser.add_argument("--mask_dir", type=str, default="results/totalseg_masks")
    parser.add_argument("--totalseg_task", type=str, default="gi")
    args = parser.parse_args()
    
    log.info(f"Initialising MultiphasePairedDataset with CSV: {args.csv}")
    dataset = MultiphasePairedDataset(
        registration_csv=args.csv, 
        base_dir=args.base_dir,
        mask_dir=args.mask_dir,
        totalseg_task=args.totalseg_task,
    )
    log.info(f"Dataset length: {len(dataset)}")
