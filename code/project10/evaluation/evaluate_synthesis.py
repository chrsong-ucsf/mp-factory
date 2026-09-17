import argparse
import logging
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_nifti(filepath: str | Path) -> np.ndarray:
    """Load a NIfTI file and return its data array as a float32 numpy array."""
    img = nib.load(str(filepath))
    return img.get_fdata().astype(np.float32)

def compute_metrics_3d(real_volume: np.ndarray, fake_volume: np.ndarray) -> Dict[str, float]:
    """
    Compute SSIM, PSNR, L1, and L2 errors for a 3D volume.
    Computes SSIM and PSNR slice-by-slice across the axial (Z) dimension and averages them.
    """
    if real_volume.shape != fake_volume.shape:
        raise ValueError(f"Shape mismatch: {real_volume.shape} vs {fake_volume.shape}")
        
    l1_error = np.mean(np.abs(real_volume - fake_volume))
    l2_error = np.mean((real_volume - fake_volume) ** 2)
    
    # Calculate data range based on the real volume
    data_range = real_volume.max() - real_volume.min()
    if data_range == 0:
        data_range = 1.0  # Prevent division by zero if volume is constant

    ssim_vals = []
    psnr_vals = []
    
    # Iterate over the Z axis (assuming last dimension is Z)
    z_dim = real_volume.shape[-1]
    for z in range(z_dim):
        real_slice = real_volume[..., z]
        fake_slice = fake_volume[..., z]
        
        # Calculate SSIM per slice
        slice_ssim = ssim(
            real_slice, 
            fake_slice, 
            data_range=data_range
        )
        ssim_vals.append(slice_ssim)
        
        # Calculate PSNR per slice
        slice_psnr = psnr(
            real_slice, 
            fake_slice, 
            data_range=data_range
        )
        psnr_vals.append(slice_psnr)

    return {
        'L1_Error': float(l1_error),
        'L2_Error': float(l2_error),
        'SSIM': float(np.mean(ssim_vals)),
        'PSNR': float(np.mean(psnr_vals))
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesized multi-phase CTs against ground truth.")
    parser.add_argument("--fake_dir", type=str, required=True, help="Directory containing synthesized (fake) CT NIfTI files.")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory containing ground truth (real) CT NIfTI files.")
    parser.add_argument("--output_csv", type=str, default="results/project10/evaluation_reports/synthesis_metrics.csv", 
                        help="Path to save the output CSV report.")
    
    args = parser.parse_args()
    
    fake_dir = Path(args.fake_dir)
    real_dir = Path(args.real_dir)
    output_csv = Path(args.output_csv)
    
    if not fake_dir.is_dir():
        logger.error(f"Fake directory does not exist: {fake_dir}")
        return
        
    if not real_dir.is_dir():
        logger.error(f"Real directory does not exist: {real_dir}")
        return

    # Find all NIfTI files in the fake directory
    fake_files = list(fake_dir.rglob("*.nii.gz")) + list(fake_dir.rglob("*.nii"))
    
    if not fake_files:
        logger.warning(f"No NIfTI files found in {fake_dir}")
        return
        
    logger.info(f"Found {len(fake_files)} files in fake directory.")
    
    results = []
    
    for fake_path in fake_files:
        # Match by filename
        filename = fake_path.name
        
        # Check if the file is at the root of real_dir or keep the relative structure if needed.
        # Assuming flat matching by filename for simplicity.
        real_path = real_dir / filename
        
        if not real_path.exists():
            logger.warning(f"Missing corresponding real file for {filename} in {real_dir}. Skipping.")
            continue
            
        logger.info(f"Evaluating {filename}...")
        
        try:
            fake_vol = load_nifti(fake_path)
            real_vol = load_nifti(real_path)
            
            metrics = compute_metrics_3d(real_vol, fake_vol)
            metrics['Subject'] = filename
            results.append(metrics)
            
            logger.info(f"[{filename}] SSIM: {metrics['SSIM']:.4f}, PSNR: {metrics['PSNR']:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    if not results:
        logger.warning("No files were successfully evaluated.")
        return

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    
    # Reorder columns to have 'Subject' first
    cols = ['Subject'] + [c for c in df.columns if c != 'Subject']
    df = df[cols]
    
    df.to_csv(output_csv, index=False)
    logger.info(f"Evaluation complete. Report saved to {output_csv}")
    
    # Print summary metrics
    summary = df.drop(columns=['Subject']).mean()
    logger.info("=== Overall Summary Metrics ===")
    for metric, value in summary.items():
        logger.info(f"Mean {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
