"""
predict_mednext.py

Batch inference with a trained MedNeXt checkpoint across all ct.nii.gz files,
producing 5-class GI organ segmentation masks.

Organ labels:
  0 = Background
  1 = Stomach
  2 = Duodenum
  3 = Small Bowel / Intestine
  4 = Colon
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

from nnunet_mednext import create_mednext_v1
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, EnsureTyped,
)
from monai.transforms import AsDiscrete

ORGAN_NAMES = {0: 'background', 1: 'stomach', 2: 'duodenum', 3: 'small_bowel', 4: 'colon'}


def discover_ct_files(data_dir, array_id=0, total_workers=1):
    all_ct_files = sorted(glob.glob(os.path.join(data_dir, "*", "ct.nii.gz")))
    print(f"Total ct.nii.gz files discovered: {len(all_ct_files)}", flush=True)
    chunk_size = (len(all_ct_files) + total_workers - 1) // total_workers
    start = array_id * chunk_size
    end = min(start + chunk_size, len(all_ct_files))
    my_files = all_ct_files[start:end]
    print(f"[Worker {array_id}/{total_workers}] Processing {len(my_files)} scans (indices {start}..{end-1})", flush=True)
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
    parser = argparse.ArgumentParser(description="MedNeXt GI Organ Segmentation Inference")
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument("--model_path", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/mednext_models/fold_0/best_mednext_gi.pt")
    parser.add_argument("--out_dir", type=str,
                        default="/mnt/scratch/user/chrsong/mp-factory/results/mednext_predictions")
    parser.add_argument("--model_id", type=str, default="B", help="MedNeXt variant: S, B, M, L")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--array_id", type=int, default=0)
    parser.add_argument("--total_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_probs", action="store_true", help="Save softmax probability maps (.npz) alongside segmentation masks")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    if not os.path.exists(args.model_path):
        print(f"ERROR: Checkpoint not found at {args.model_path}")
        print("Available checkpoints:")
        for pt in glob.glob("/mnt/scratch/user/chrsong/mp-factory/results/mednext_models/**/best_*.pt", recursive=True):
            print(f"  {pt}")
        sys.exit(1)

    print(f"Loading MedNeXt-{args.model_id} checkpoint from: {args.model_path}", flush=True)
    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=5,
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        deep_supervision=False,
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!", flush=True)

    ct_files = discover_ct_files(args.data_dir, args.array_id, args.total_workers)
    if not ct_files:
        print("No ct.nii.gz files found for this worker. Exiting.")
        return

    transform = get_preprocessing()
    post_pred = AsDiscrete(argmax=True)

    skipped = processed = failed = 0

    for ct_path in ct_files:
        subject_id = os.path.basename(os.path.dirname(ct_path))
        out_mask_path = os.path.join(args.out_dir, f"{subject_id}_gi_seg.nii.gz")

        if os.path.exists(out_mask_path) and not args.overwrite:
            print(f"  [SKIP] {subject_id}", flush=True)
            skipped += 1
            continue

        print(f"  [RUN ] {subject_id}", flush=True)

        try:
            data_dict = transform({"image": ct_path})
            img_tensor = data_dict["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    pred_logits = sliding_window_inference(
                        inputs=img_tensor,
                        roi_size=tuple(args.roi_size),
                        sw_batch_size=args.sw_batch_size,
                        predictor=model,
                        overlap=args.overlap,
                    )

            pred_label = post_pred(pred_logits[0])
            pred_np = pred_label.cpu().numpy().astype(np.uint8)

            orig_nii = nib.load(ct_path)
            out_nii = nib.Nifti1Image(pred_np, affine=orig_nii.affine)
            nib.save(out_nii, out_mask_path)

            if args.save_probs:
                probs = torch.softmax(pred_logits[0], dim=0).cpu().numpy().astype(np.float16)
                prob_path = os.path.join(args.out_dir, f"{subject_id}_gi_probs.npz")
                np.savez_compressed(prob_path, probs=probs)

            unique_labels = np.unique(pred_np)
            label_names = [ORGAN_NAMES.get(int(l), str(l)) for l in unique_labels]
            print(f"    Saved: {out_mask_path} | Organs: {label_names}", flush=True)
            processed += 1

        except Exception as e:
            import traceback
            print(f"  [FAIL] {subject_id}: {e}", flush=True)
            traceback.print_exc()
            failed += 1

    print(f"\n[Worker {args.array_id}] Done — Processed: {processed} | Skipped: {skipped} | Failed: {failed}", flush=True)


if __name__ == "__main__":
    main()
