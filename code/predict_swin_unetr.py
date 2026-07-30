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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading Swin-UNETR checkpoint from: {args.model_path}")
    model = SwinUNETR(
        in_channels=1,
        out_channels=5,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3,
    ).to(device)

    if not os.path.exists(args.model_path):
        print(f"ERROR: Checkpoint not found at {args.model_path}")
        print("Available checkpoints:")
        for pt in glob.glob("/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_models/**/best_*.pt", recursive=True):
            print(f"  {pt}")
        sys.exit(1)

    state_dict = torch.load(args.model_path, map_location=device)
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
                with torch.cuda.amp.autocast():
                    pred_logits = sliding_window_inference(
                        inputs=img_tensor,
                        roi_size=roi_size,
                        sw_batch_size=args.sw_batch_size,
                        predictor=model,
                        overlap=args.overlap,
                    )

            # Argmax over class dimension → (1, H, W, D)
            pred_label = post_pred(pred_logits[0])   # (H, W, D)
            pred_np = pred_label.cpu().numpy().astype(np.uint8)

            # Load original ct.nii.gz to recover affine/header for output
            orig_nii = nib.load(ct_path)

            # The prediction is in resampled/reoriented RAS space;
            # save in the preprocessed space (1.5x1.5x2.0 mm RAS)
            out_nii = nib.Nifti1Image(pred_np, affine=orig_nii.affine)
            nib.save(out_nii, out_mask_path)

            print(f"    Saved: {out_mask_path}")
            unique_labels = np.unique(pred_np)
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
