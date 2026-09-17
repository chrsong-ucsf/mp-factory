"""
code/project10/datasets/multiphase_paired_dataset.py
====================================================
Dataset for Project 10 Phase 3 Stage 2 synthesis.

Reads `results/project10/manifests/registration_log.csv`, filters by status == 'OK',
and yields pairs of (ncct, target, mask).

The output dictionary contains:
    - ncct: Tensor (1, 128, 128, 128)
    - target: Tensor (1, 128, 128, 128)
    - mask: Tensor (1, 128, 128, 128) (organ mask)
    - phase: string (e.g. 'VENOUS')
    - subject_id: string (subject_p0)

Usage:
    from code.project10.datasets.multiphase_paired_dataset import MultiphasePairedDataset
    dataset = MultiphasePairedDataset("results/project10/manifests/registration_log.csv")
    sample = dataset[0]
"""

import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
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
        target_shape: tuple = (128, 128, 128),
        spacing_mm: float = 1.5
    ):
        """
        Args:
            registration_csv (str): Path to the registration manifest CSV.
            base_dir (str): Base directory of the repository (for resolving absolute paths).
            mask_dir (str): Directory containing TotalSegmentator masks.
            target_shape (tuple): Spatial dimensions for Padding/Cropping.
            spacing_mm (float): Isotropic voxel spacing in mm.
        """
        super().__init__()
        self.base_dir = base_dir
        self.mask_dir = os.path.join(base_dir, mask_dir) if not os.path.isabs(mask_dir) else mask_dir
        
        csv_path = os.path.join(base_dir, registration_csv) if not os.path.isabs(registration_csv) else registration_csv
        
        # Load and filter CSV
        try:
            df = pd.read_csv(csv_path)
            if 'status' in df.columns:
                self.df = df[df['status'] == 'OK'].reset_index(drop=True)
            else:
                log.warning("Column 'status' not found in registration CSV. Using all rows.")
                self.df = df
        except Exception as e:
            log.error(f"Failed to load registration CSV from {csv_path}: {e}")
            self.df = pd.DataFrame()
            
        self.target_shape = target_shape
        self.spacing_mm = spacing_mm
        
        # Define MONAI transform pipeline for spatial alignment and normalization
        self.transform = Compose([
            LoadImaged(keys=["ncct", "target", "mask"], allow_missing_keys=True, image_only=True),
            EnsureChannelFirstd(keys=["ncct", "target", "mask"], allow_missing_keys=True),
            Spacingd(
                keys=["ncct", "target", "mask"], 
                pixdim=(spacing_mm, spacing_mm, spacing_mm), 
                mode=("bilinear", "bilinear", "nearest"), 
                allow_missing_keys=True
            ),
            ResizeWithPadOrCropd(
                keys=["ncct", "target", "mask"], 
                spatial_size=target_shape,
                mode="constant",
                value=-1000.0, # Pad with -1000 HU (air) if needed
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
        subject_p0 = str(row['subject_p0']) if 'subject_p0' in row else f"subject_{idx}"
        phase_target = str(row['phase_target']) if 'phase_target' in row else 'UNKNOWN'
        
        ncct_path = os.path.join(self.base_dir, "CancerVerse/CancerVerse", subject_p0, "ct.nii.gz")
        target_path = str(row['out_reg_path']) if 'out_reg_path' in row else ""
        
        # Resolve target_path if it's relative
        if target_path and not os.path.isabs(target_path):
            target_path = os.path.join(self.base_dir, target_path)
            
        mask_path = os.path.join(self.mask_dir, f"{subject_p0}.nii.gz")
        
        data_dict = {
            "ncct": ncct_path,
            "target": target_path
        }
        
        if os.path.exists(mask_path):
            data_dict["mask"] = mask_path
        else:
            log.debug(f"Mask not found for {subject_p0} at {mask_path}")

        try:
            transformed = self.transform(data_dict)
            
            # Extract standard torch tensors
            ncct_tensor = torch.as_tensor(transformed["ncct"])
            target_tensor = torch.as_tensor(transformed["target"])
            
            # Mask handling
            if "mask" in transformed:
                mask_tensor = torch.as_tensor(transformed["mask"])
            else:
                mask_tensor = torch.zeros((1, *self.target_shape), dtype=torch.float32)
                
            return {
                "ncct": ncct_tensor,
                "target": target_tensor,
                "mask": mask_tensor,
                "phase": phase_target,
                "subject_id": subject_p0
            }
        except Exception as e:
            log.error(f"Error processing index {idx} (subject {subject_p0}): {e}")
            # Fallback to zeros on error to prevent dataloader crashing
            return {
                "ncct": torch.zeros((1, *self.target_shape), dtype=torch.float32),
                "target": torch.zeros((1, *self.target_shape), dtype=torch.float32),
                "mask": torch.zeros((1, *self.target_shape), dtype=torch.float32),
                "phase": phase_target,
                "subject_id": subject_p0
            }

if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Smoke test for MultiphasePairedDataset")
    parser.add_argument("--csv", type=str, default="results/project10/manifests/registration_log.csv")
    parser.add_argument("--base_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory")
    parser.add_argument("--mask_dir", type=str, default="results/totalseg_masks")
    args = parser.parse_args()
    
    log.info(f"Initialising MultiphasePairedDataset with CSV: {args.csv}")
    
    dataset = MultiphasePairedDataset(
        registration_csv=args.csv, 
        base_dir=args.base_dir,
        mask_dir=args.mask_dir
    )
    
    log.info(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        log.info("Successfully loaded sample 0!")
        log.info(f"Keys: {list(sample.keys())}")
        log.info(f"ncct shape:   {sample['ncct'].shape}, min: {sample['ncct'].min():.4f}, max: {sample['ncct'].max():.4f}")
        log.info(f"target shape: {sample['target'].shape}, min: {sample['target'].min():.4f}, max: {sample['target'].max():.4f}")
        log.info(f"mask shape:   {sample['mask'].shape}, unique vals: {torch.unique(sample['mask']).tolist()}")
        log.info(f"phase:        {sample['phase']}")
        log.info(f"subject_id:   {sample['subject_id']}")
    else:
        log.warning("Dataset is empty. Ensure the CSV exists and has rows with status == 'OK'.")
