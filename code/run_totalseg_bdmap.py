import os
import glob
import sys
import subprocess
import nibabel as nib

def main():
    array_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    input_base = "/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox"
    output_base = "/mnt/scratch/user/chrsong/mp-factory/results/totalseg_gi_masks_bdmap"
    os.makedirs(output_base, exist_ok=True)

    import time

    print(f"[Worker {array_id}] Continuous monitoring initialized on {input_base}...")

    # Run continuous processing loop while files are being transferred
    while True:
        # Find primary CT scan per subfolder
        all_cases = sorted(glob.glob(os.path.join(input_base, "*", "ct.nii.gz")))
        if not all_cases:
            all_cases = sorted(glob.glob(os.path.join(input_base, "**", "*.nii.gz"), recursive=True))

        if len(all_cases) == 0:
            print(f"[Worker {array_id}] No scan cases found in {input_base} yet. Sleeping for 30s...")
            time.sleep(30)
            continue

        chunk_size = (len(all_cases) + total_chunks - 1) // total_chunks
        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_cases))
        my_cases = all_cases[start_idx:end_idx]

        processed_something = False

        for ct_path in my_cases:
            rel_path = os.path.relpath(ct_path, input_base)
            parts = rel_path.split(os.sep)
            case_id = parts[0] if len(parts) > 1 else os.path.splitext(os.path.basename(ct_path))[0]

            out_mask_path = os.path.join(output_base, f"{case_id}_gi_mask.nii.gz")

            if os.path.exists(out_mask_path):
                continue

            try:
                nii = nib.load(ct_path)
                if len(nii.shape) != 3:
                    print(f"[{case_id}] Skipping non-3D volume: {nii.shape}")
                    continue
            except Exception as e:
                # File might still be actively downloading via rsync; wait and skip for now
                print(f"[{case_id}] File partial/locked, skipping for next iteration: {e}")
                continue

            print(f"[{case_id}] Running TotalSegmentator...")
            cmd = [
                "TotalSegmentator",
                "-i", ct_path,
                "-o", out_mask_path,
                "--task", "total",
                "--fast",
                "--ml",
                "--quiet"
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"[{case_id}] Mask saved: {out_mask_path}")
                processed_something = True
            except subprocess.CalledProcessError as e:
                print(f"[{case_id}] TotalSegmentator error: {e}")

        # Sleep briefly before rescanning for newly downloaded batches
        if not processed_something:
            time.sleep(20)

if __name__ == "__main__":
    main()
