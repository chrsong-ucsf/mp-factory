#!/usr/bin/env python3
"""
run_totalsegmentator_batch.py

Runs TotalSegmentator on all BDMAP directories containing `ct.nii.gz`
under the CancerVerse dataset directory on CHPC.

Usage:
    python3 run_totalsegmentator_batch.py [--fast] [--gpu]

Output:
    Saves segmentation outputs into `BDMAP_XXXX/totalsegmentator_output/`
    Skips directories that have already been processed.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Base data directory on CHPC
DATA_DIR = Path("/mnt/scratch/user/chrsong/CancerVerse_data")

def main():
    parser = argparse.ArgumentParser(description="Run TotalSegmentator across all BDMAP folders.")
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator --fast mode (faster, slightly lower resolution)")
    parser.add_argument("--body-only", action="store_true", help="Segment body/organs only")
    parser.add_argument("--task", type=str, default=None, help="Specific task for TotalSegmentator (e.g. total, organs, bones, etc.)")
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} does not exist!")
        sys.exit(1)

    # Find all subdirectories matching BDMAP_*
    bdmap_folders = sorted([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("BDMAP_")])
    total_folders = len(bdmap_folders)

    print("==========================================================")
    print("           TotalSegmentator Batch Processor               ")
    print("==========================================================")
    print(f"Data Path       : {DATA_DIR}")
    print(f"Found BDMAPs    : {total_folders}")
    print(f"Fast Mode       : {args.fast}")
    print("==========================================================\n")

    processed = 0
    skipped = 0
    failed = 0

    for idx, folder in enumerate(bdmap_folders, start=1):
        ct_file = folder / "ct.nii.gz"
        out_dir = folder / "totalsegmentator_output"

        # Check if input CT scan exists
        if not ct_file.exists():
            # Check for lower/upper case variations if needed
            alt_ct = folder / "ct.nii"
            if alt_ct.exists():
                ct_file = alt_ct
            else:
                skipped += 1
                continue

        # Skip if already processed (output folder exists and is non-empty)
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[{idx}/{total_folders}] ⏩ Skipping {folder.name} (already processed)")
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{total_folders}] ⚙️  Processing {folder.name} ...")

        # Build TotalSegmentator command
        cmd = ["TotalSegmentator", "-i", str(ct_file), "-o", str(out_dir)]
        if args.fast:
            cmd.append("--fast")
        if args.task:
            cmd.extend(["--task", args.task])

        try:
            res = subprocess.run(cmd, check=True)
            print(f"  ✓ Finished {folder.name}")
            processed += 1
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed processing {folder.name}: {e}")
            failed += 1
        except FileNotFoundError:
            print("Error: 'TotalSegmentator' CLI tool not found in PATH.")
            print("Make sure TotalSegmentator is installed: pip install TotalSegmentator")
            sys.exit(1)

    print("\n==========================================================")
    print("                    Batch Complete                        ")
    print("==========================================================")
    print(f"Total Folders Scanned : {total_folders}")
    print(f"Successfully Processed: {processed}")
    print(f"Skipped (Done/No CT)  : {skipped}")
    print(f"Failed                : {failed}")

if __name__ == "__main__":
    main()
