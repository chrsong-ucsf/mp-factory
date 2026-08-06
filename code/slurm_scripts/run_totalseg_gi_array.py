import os
import sys

# Force CUDA device initialization & optimize memory allocation
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import gc
import torch
from totalsegmentator.python_api import totalsegmentator

def main():
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    DATA_DIR = "/mnt/scratch/user/chrsong/mp-factory/CancerVerse/CancerVerse"
    OUTPUT_DIR = "/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    # Fetch all candidate CT scans
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "CV_*", "ct.nii.gz")))

    # Filter down to ONLY missing files
    unprocessed_files = []
    for ct_path in all_files:
        case_id = os.path.basename(os.path.dirname(ct_path))
        out_mask_path = os.path.join(OUTPUT_DIR, f"{case_id}_gi_mask.nii.gz")
        if not os.path.exists(out_mask_path):
            unprocessed_files.append(ct_path)

    print(f"Total scans in dataset: {len(all_files)}")
    print(f"Remaining unprocessed scans: {len(unprocessed_files)}")

    if not unprocessed_files:
        print("All scans have already been processed! Exiting.")
        return

    # Divide missing files across array workers
    chunk_size = len(unprocessed_files) // num_tasks
    start_idx = task_id * chunk_size
    end_idx = len(unprocessed_files) if task_id == num_tasks - 1 else (task_id + 1) * chunk_size

    ct_files = unprocessed_files[start_idx:end_idx]
    print(f"[Worker {task_id+1}/{num_tasks}] Assigned {len(ct_files)} missing files...")

    for idx, ct_path in enumerate(ct_files):
        case_id = os.path.basename(os.path.dirname(ct_path))
        out_mask_path = os.path.join(OUTPUT_DIR, f"{case_id}_gi_mask.nii.gz")

        if os.path.exists(out_mask_path):
            continue

        try:
            totalsegmentator(
                input=ct_path,
                output=out_mask_path,
                fast=True,
                ml=True,
                device="gpu"
            )
            print(f"[{task_id+1}] Finished {case_id}")
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
        finally:
            # Explicitly release PyTorch cache & force Garbage Collection after every scan
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
