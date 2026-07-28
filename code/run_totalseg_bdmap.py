import os
import glob
import sys
import subprocess
import nibabel as nib

def main():
    array_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    input_base = "/mnt/scratch/user/chrsong/CancerVerse_data"
    output_base = "/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap"
    os.makedirs(output_base, exist_ok=True)

    all_cases = sorted(glob.glob(os.path.join(input_base, "BDMAP_*", "ct.nii.gz")))
    print(f"Total BDMAP cases found: {len(all_cases)}")

    if len(all_cases) == 0:
        print(f"No BDMAP cases found in {input_base}! Exiting.")
        sys.exit(1)

    chunk_size = (len(all_cases) + total_chunks - 1) // total_chunks
    start_idx = array_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_cases))
    my_cases = all_cases[start_idx:end_idx]

    print(f"Task ID {array_id}/{total_chunks}: processing cases [{start_idx} to {end_idx - 1}] ({len(my_cases)} cases)...")

    for ct_path in my_cases:
        case_id = os.path.basename(os.path.dirname(ct_path))
        out_mask_path = os.path.join(output_base, f"{case_id}_gi_mask.nii.gz")

        if os.path.exists(out_mask_path):
            print(f"[{case_id}] Already exists. Skipping.")
            continue

        try:
            nii = nib.load(ct_path)
            if len(nii.shape) != 3:
                print(f"[{case_id}] Skipping non-3D volume: {nii.shape}")
                continue
        except Exception as e:
            print(f"[{case_id}] Skipping corrupt file: {e}")
            continue

        print(f"[{case_id}] Running TotalSegmentator...")
        cmd = [
            "TotalSegmentator",
            "-i", ct_path,
            "-o", out_mask_path,
            "--task", "total",
            "--fast",
            "--quiet"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"[{case_id}] Mask saved: {out_mask_path}")
        except subprocess.CalledProcessError as e:
            print(f"[{case_id}] TotalSegmentator error: {e}")

if __name__ == "__main__":
    main()
