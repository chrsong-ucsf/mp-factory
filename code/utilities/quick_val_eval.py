#!/usr/bin/env python3
"""
quick_val_eval.py

Fast standalone evaluation of the Phase 2 validation pipeline on real data.
Runs sliding window inference on 1-5 validation cases using a pretrained checkpoint,
verifying that the ResampleToMatchd spatial alignment works and printing the exact
Dice score immediately (no need to wait for 5 training epochs).

Usage on CHPC (interactive node or srun):
    python code/utilities/quick_val_eval.py --num_cases 3
"""

import os
import sys
import glob
import torch
import argparse
import numpy as np

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

# Project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
from train_mednext_phase2 import (
    get_transforms,
    NUM_CLASSES,
    IGNORE_INDEX,
    create_mednext_v1,
)

DEFAULT_SCRATCH = "/mnt/scratch/user/chrsong/mp-factory"


def main():
    parser = argparse.ArgumentParser(description="Quick standalone validation evaluation.")
    parser.add_argument("--data_dir",         type=str, default=f"{DEFAULT_SCRATCH}/CancerVerse_dbox")
    parser.add_argument("--audit_csv",        type=str, default=f"{DEFAULT_SCRATCH}/results/ensemble_audit_summary.csv")
    parser.add_argument("--ensemble_out_dir", type=str, default=f"{DEFAULT_SCRATCH}/results/ensemble_out")
    parser.add_argument("--autolabel_dir",    type=str, default=f"{DEFAULT_SCRATCH}/results/autolabel_out")
    parser.add_argument("--gold_standard_dir",type=str, default=f"{DEFAULT_SCRATCH}/JHU_data_radiologist_corrected")
    parser.add_argument("--ckpt",             type=str, default=f"{DEFAULT_SCRATCH}/results/mednext_models/fold_0/best_mednext_gi.pt")
    parser.add_argument("--num_cases",        type=int, default=3, help="Number of validation cases to test")
    parser.add_argument("--model_id",         type=str, default="B")
    parser.add_argument("--kernel_size",      type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  Quick Validation Evaluation")
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Cases:      {args.num_cases}")
    print("=" * 65)

    # 1. Gather validation scans (same logic as Phase 2 train script)
    from train_mednext_phase2 import load_dataset_pairs
    data_pairs = load_dataset_pairs(
        data_dir=args.data_dir,
        audit_csv=args.audit_csv,
        ensemble_out_dir=args.ensemble_out_dir,
        autolabel_dir=args.autolabel_dir,
        gold_standard_dir=args.gold_standard_dir,
        use_weak=True,
    )

    if not data_pairs:
        print("ERROR: No data pairs found. Check paths.")
        sys.exit(1)

    # Pick validation samples
    np.random.seed(42)
    val_pairs = data_pairs[-args.num_cases:]
    print(f"Evaluating {len(val_pairs)} cases:")
    for p in val_pairs:
        print(f"  - CT:    {os.path.basename(p['image'])}\n    Label: {os.path.basename(p['label'])}")

    # 2. Setup transforms and dataset
    _, val_tf = get_transforms(roi_size=(96, 96, 96))
    val_ds = Dataset(data=val_pairs, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

    # 3. Build Model & load checkpoint
    if create_mednext_v1 is None:
        print("ERROR: MedNeXt is not installed. Run: pip install git+https://github.com/MIC-DKFZ/MedNeXt.git")
        sys.exit(1)

    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=NUM_CLASSES,
        model_id=args.model_id,
        kernel_size=args.kernel_size,
        deep_supervision=False,
    ).to(device)

    if os.path.exists(args.ckpt):
        print(f"\nLoading weights from: {args.ckpt}")
        ckpt_data = torch.load(args.ckpt, map_location=device)
        state_dict = ckpt_data.get("state_dict", ckpt_data)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded! (missing: {len(missing)}, unexpected: {len(unexpected)})")
    else:
        print(f"WARNING: Checkpoint {args.ckpt} not found! Testing with randomly initialized model.")

    model.eval()
    post_pred  = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    print("\nRunning sliding window inference with ResampleToMatchd alignment...")
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            vi = batch["image"].to(device)
            vl = batch["label"].to(device)

            print(f"\n[Case {idx+1}/{len(val_loader)}] Input CT shape: {tuple(vi.shape)}, Label shape: {tuple(vl.shape)}")
            with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
                vo = sliding_window_inference(vi, (96, 96, 96), 4, model)

            print(f"  Prediction shape: {tuple(vo.shape)}")
            if vo.shape[-3:] != vl.shape[-3:]:
                print(f"  Shape mismatch! {vo.shape[-3:]} vs {vl.shape[-3:]} - Falling back to interpolation")
                vo = torch.nn.functional.interpolate(
                    vo.float(), size=vl.shape[-3:], mode="trilinear", align_corners=False
                )
            else:
                print("  Shapes perfectly match! ResampleToMatchd aligned grid correctly.")

            vl_clean = torch.where(vl == IGNORE_INDEX, torch.zeros_like(vl), vl)
            vo_post = [post_pred(i) for i in decollate_batch(vo)]
            vl_post = [post_label(i) for i in decollate_batch(vl_clean)]
            dice_metric(y_pred=vo_post, y=vl_post)

    mean_dice = dice_metric.aggregate().item()
    print("\n" + "=" * 65)
    print(f"  --> Final Mean Validation Dice: {mean_dice:.4f}")
    print("=" * 65)
    if mean_dice > 0.05:
        print("  SUCCESS: Validation Dice is positive and correctly computed!")
    else:
        print("  CAUTION: Dice is near 0. Check label alignment or checkpoint compatibility.")


if __name__ == "__main__":
    main()
