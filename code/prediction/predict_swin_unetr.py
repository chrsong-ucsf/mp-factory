"""
predict_swin_unetr.py

Runs batch inference with a trained Swin-UNETR checkpoint on all ct.nii.gz
files in a directory, producing 5-class GI organ segmentation masks.

Organ labels:
  0 = Background
  1 = Stomach
  2 = Duodenum
  3 = Small Bowel / Intestine
  4 = Colon

Usage (single GPU):
  python predict_swin_unetr.py \
    --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
    --model_path /mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_models/fold_0/best_swin_unetr_gi.pt \
    --out_dir /mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions

SLURM array usage (4 workers each handle a shard):
  Called via run_predict_swin_unetr.sbatch with SLURM_ARRAY_TASK_ID
"""

import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete

ORGAN_NAMES = {0: 'background', 1: 'stomach', 2: 'duodenum', 3: 'small_bowel', 4: 'colon'}


def discover_ct_files(data_dir, array_id=0, total_workers=1):
    """Find all ct.nii.gz files and shard them across parallel workers."""
    all_ct_files = sorted(glob.glob(os.path.join(data_dir, "*", "ct.nii.gz")))
    print(f"Total ct.nii.gz files discovered: {len(all_ct_files)}")

    # Shard the file list across workers
    chunk_size = (len(all_ct_files) + total_workers - 1) // total_workers
    start = array_id * chunk_size
    end = min(start + chunk_size, len(all_ct_files))
    my_files = all_ct_files[start:end]
    print(f"[Worker {array_id}/{total_workers}] Processing {len(my_files)} scans (indices {start}..{end-1})")
    return my_files


def get_preprocessing():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])


def main():
    parser = argparse.ArgumentParser(description="Swin-UNETR GI Organ Segmentation Inference")
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox",
                        help="Root directory containing BDMAP_* subdirs with ct.nii.gz")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_models/fold_0/best_swin_unetr_gi.pt",
                        help="Path to trained Swin-UNETR .pt checkpoint")
    parser.add_argument("--out_dir", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions",
                        help="Output directory for predicted segmentation masks")
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96],
                        help="Sliding window inference patch size (default: 96 96 96)")
    parser.add_argument("--sw_batch_size", type=int, default=4,
                        help="Number of patches per sliding window forward pass")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap ratio (0.0-1.0, default: 0.5)")
    parser.add_argument("--array_id", type=int, default=0,
                        help="SLURM array task ID (worker index)")
    parser.add_argument("--total_workers", type=int, default=1,
                        help="Total number of parallel workers / SLURM array size")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run inference even if output mask already exists")
    parser.add_argument("--save_probs", action="store_true",
                        help="Save softmax probability maps (.npz) alongside segmentation masks")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate checkpoint path BEFORE allocating GPU memory (avoids wasted CUDA alloc on missing file)
    if not os.path.exists(args.model_path):
        print(f"ERROR: Checkpoint not found at {args.model_path}")
        print("Available checkpoints:")
        for pt in glob.glob("/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_models/**/best_*.pt", recursive=True):
            print(f"  {pt}")
        sys.exit(1)

    # Load model
    print(f"Loading Swin-UNETR checkpoint from: {args.model_path}")
    model = SwinUNETR(
        in_channels=1,
        out_channels=5,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3,
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")

    # Discover and shard CT files
    ct_files = discover_ct_files(args.data_dir, args.array_id, args.total_workers)
    if not ct_files:
        print("No ct.nii.gz files found for this worker. Exiting.")
        return

    transform = get_preprocessing()
    post_pred = AsDiscrete(argmax=True)

    skipped = 0
    processed = 0
    failed = 0

    for ct_path in ct_files:
        # Build output path: preserve subject folder name
        subject_id = os.path.basename(os.path.dirname(ct_path))
        out_mask_path = os.path.join(args.out_dir, f"{subject_id}_gi_seg.nii.gz")

        if os.path.exists(out_mask_path) and not args.overwrite:
            print(f"  [SKIP] {subject_id} — output already exists")
            skipped += 1
            continue

        print(f"  [RUN ] {subject_id}")

        try:
            # Preprocess
            data_dict = transform({"image": ct_path})
            img_tensor = data_dict["image"].unsqueeze(0).to(device)   # (1, 1, H, W, D)

            roi_size = tuple(args.roi_size)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda'):
                    pred_logits = sliding_window_inference(
                        inputs=img_tensor,
                        roi_size=roi_size,
                        sw_batch_size=args.sw_batch_size,
                        predictor=model,
                        overlap=args.overlap,
                    )

            # Argmax over class dimension → (H, W, D)
            # Extract softmax probabilities and check finiteness
            probs_t = torch.softmax(pred_logits[0], dim=0).cpu().numpy().astype(np.float32)
            if not np.all(np.isfinite(probs_t)):
                raise ValueError(f"Corrupted non-finite probabilities detected for {subject_id}")

            # Get true updated affine of the resampled image
            if hasattr(data_dict["image"], "affine"):
                resampled_affine = data_dict["image"].affine.cpu().numpy()
            elif "affine" in data_dict["image"].meta:
                resampled_affine = data_dict["image"].meta["affine"]
                if hasattr(resampled_affine, "numpy"):
                    resampled_affine = resampled_affine.numpy()
            else:
                resampled_affine = np.eye(4)

            # Rigorously resample probabilities back onto native CT scanner coordinate grid
            orig_nii = nib.load(ct_path)
            import nibabel.processing as nip

            native_probs = np.zeros((5,) + orig_nii.shape, dtype=np.float32)
            for c in range(5):
                ch_nii = nib.Nifti1Image(probs_t[c], affine=resampled_affine)
                ch_native = nip.resample_from_to(ch_nii, orig_nii, order=1).get_fdata(dtype=np.float32)
                native_probs[c] = np.maximum(0.0, ch_native)

            # Simplex normalization & out-of-support definition
            prob_sum = np.sum(native_probs, axis=0)
            support_mask = (prob_sum > 1e-4).astype(np.uint8)

            prob_sum_safe = np.where(prob_sum > 1e-4, prob_sum, 1.0)
            native_probs = native_probs / prob_sum_safe
            # For out-of-support background: set class 0 to 1.0 as storage convention, record support_mask=0
            native_probs[0] = np.where(support_mask == 1, native_probs[0], 1.0)
            for c in range(1, 5):
                native_probs[c] = np.where(support_mask == 1, native_probs[c], 0.0)

            # Derive native hard mask directly from native normalized probabilities (complete consistency)
            native_pred_np = np.argmax(native_probs, axis=0).astype(np.uint8)
            native_pred_nii = nib.Nifti1Image(native_pred_np, affine=orig_nii.affine, header=orig_nii.header)
            nib.save(native_pred_nii, out_mask_path)

            if args.save_probs:
                # Round-trip verification: test whether float16 precision preserves argmax exactly
                probs_f16 = native_probs.astype(np.float16)
                disk_pred = np.argmax(probs_f16.astype(np.float32), axis=0).astype(np.uint8)
                if not np.array_equal(disk_pred, native_pred_np):
                    probs_to_save = native_probs
                    precision_stored = "float32"
                else:
                    probs_to_save = probs_f16
                    precision_stored = "float16"

                prob_path = os.path.join(args.out_dir, f"{subject_id}_gi_probs.npz")
                np.savez_compressed(
                    prob_path,
                    probs=probs_to_save,
                    affine=orig_nii.affine,
                    shape=np.array(orig_nii.shape),
                    spacing=np.array(orig_nii.header.get_zooms()[:3]),
                    channel_order=np.array([0, 1, 2, 3, 4]),
                    support_mask=support_mask,
                    precision=precision_stored,
                    coordinate_space="native_ct_scanner"
                )

            print(f"    Saved: {out_mask_path}")
            unique_labels = np.unique(native_pred_np)
            label_names = [ORGAN_NAMES.get(int(l), str(l)) for l in unique_labels]
            print(f"    Predicted organs: {label_names}")
            processed += 1

        except Exception as e:
            print(f"  [FAIL] {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n========================================")
    print(f"Worker {args.array_id} Summary")
    print("========================================")
    print(f"  Processed : {processed}")
    print(f"  Skipped   : {skipped}")
    print(f"  Failed    : {failed}")


if __name__ == "__main__":
    main()
