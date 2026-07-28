import os
import glob
import subprocess

DATA_DIR = "/mnt/scratch/user/chrsong/mp-factory/CancerVerse/CancerVerse"
OUTPUT_DIR = "/mnt/scratch/user/chrsong/mp-factory/results/totalseg_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all CT scan files
ct_files = sorted(glob.glob(os.path.join(DATA_DIR, "CV_*", "ct.nii.gz")))
print(f"Found {len(ct_files)} CT scans to process.")

for idx, ct_path in enumerate(ct_files):
    # Extract ID (e.g., CV_00023731)
    case_id = os.path.basename(os.path.dirname(ct_path))
    out_mask_path = os.path.join(OUTPUT_DIR, f"{case_id}_mask.nii.gz")

    # Skip if already processed
    if os.path.exists(out_mask_path):
        continue

    print(f"[{idx+1}/{len(ct_files)}] Processing {case_id}...")

    # Run TotalSegmentator with multi-label output (--ml)
    cmd = [
        "TotalSegmentator",
        "-i", ct_path,
        "-o", out_mask_path,
        "--ml",
        "--device", "gpu"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {case_id}: {e}")
