#!/usr/bin/env python3
"""
monitor_sweep_val.py

Monitors Phase 2 scaling sweep logs and extracts validation Dice scores,
specifically highlighting Epoch 5 (the first validation checkpoint) and subsequent epochs.

Usage on CHPC:
    # 1. Print current status of all sweep jobs (losses & validation dice)
    python code/utilities/monitor_sweep_val.py

    # 2. Continuously watch in real-time (updates every 10s)
    python code/utilities/monitor_sweep_val.py --watch 10

    # 3. Filter specifically for Epoch 5 validation results
    python code/utilities/monitor_sweep_val.py --epoch 5
"""

import os
import re
import sys
import glob
import time
import argparse
from pathlib import Path

DEFAULT_LOG_DIR = "/mnt/scratch/user/chrsong/mp-factory/logs/scaling_sweep"


def parse_job_log(log_path: str):
    """
    Parses a single SLURM .out log file from train_mednext_phase2.py.
    Extracts:
      - Cohort size N and Seed from filename or log header
      - Latest epoch reached & loss
      - Validation results by epoch (e.g., epoch 5, 10, ...)
      - Best validation Dice
    """
    fname = os.path.basename(log_path)
    
    # Extract N and Seed from filename: sweep_N0004_s42_12345.out
    m = re.search(r"sweep_N(\d+)_s(\d+)", fname)
    n_cases = int(m.group(1)) if m else None
    seed = int(m.group(2)) if m else None

    latest_epoch = 0
    latest_loss = None
    latest_lr = None
    val_dices = {}  # epoch -> dice
    best_dice = 0.0

    epoch_re = re.compile(r"Epoch\s+\[(\d+)/\d+\]\s+Loss:\s+([\d\.]+)\s+LR:\s+([\deE\.\-\+]+)")
    val_re = re.compile(r"-->\s+Val Mean Dice:\s+([\d\.]+)")
    best_re = re.compile(r"\[\+\]\s+New Best!\s+Dice:\s+([\d\.]+)")

    current_parsing_epoch = None

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ep_match = epoch_re.search(line)
                if ep_match:
                    current_parsing_epoch = int(ep_match.group(1))
                    latest_epoch = current_parsing_epoch
                    latest_loss = float(ep_match.group(2))
                    latest_lr = ep_match.group(3)
                    continue

                val_match = val_re.search(line)
                if val_match:
                    dice = float(val_match.group(1))
                    if current_parsing_epoch is not None:
                        val_dices[current_parsing_epoch] = dice
                    continue

                best_match = best_re.search(line)
                if best_match:
                    best_dice = max(best_dice, float(best_match.group(1)))
    except Exception as e:
        return {"error": str(e), "file": fname}

    return {
        "file": fname,
        "n": n_cases,
        "seed": seed,
        "latest_epoch": latest_epoch,
        "latest_loss": latest_loss,
        "latest_lr": latest_lr,
        "val_dices": val_dices,
        "best_dice": best_dice,
    }


def display_status(log_dir: str, target_epoch: int = None):
    log_files = sorted(glob.glob(os.path.join(log_dir, "sweep_N*.out")))
    if not log_files:
        print(f"No sweep log files found in {log_dir}")
        return

    results = [parse_job_log(f) for f in log_files]
    # Filter out error logs
    valid_results = [r for r in results if "error" not in r]

    # Sort by cohort size N, then seed
    valid_results.sort(key=lambda r: (r["n"] if r["n"] is not None else 9999, r["seed"] if r["seed"] is not None else 0))

    header = f"{'Job Log':<30} {'N':>4} {'Seed':>5} {'Current Ep':>11} {'Loss':>8} "
    if target_epoch is not None:
        header += f"{f'Epoch {target_epoch} Dice':>16} {'Best Dice':>10}"
    else:
        header += f"{'Epoch 5 Dice':>14} {'Latest Val Dice':>16} {'Best Dice':>10}"

    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for r in valid_results:
        ep_str = f"[{r['latest_epoch']:03d}/150]"
        loss_str = f"{r['latest_loss']:.4f}" if r["latest_loss"] is not None else "N/A"
        n_str = f"{r['n']}" if r["n"] is not None else "?"
        seed_str = f"{r['seed']}" if r["seed"] is not None else "?"
        best_str = f"{r['best_dice']:.4f}" if r["best_dice"] > 0 else "---"

        if target_epoch is not None:
            target_dice = r["val_dices"].get(target_epoch)
            target_str = f"{target_dice:.4f}" if target_dice is not None else "pending..."
            print(f"{r['file']:<30} {n_str:>4} {seed_str:>5} {ep_str:>11} {loss_str:>8} {target_str:>16} {best_str:>10}")
        else:
            ep5_dice = r["val_dices"].get(5)
            ep5_str = f"{ep5_dice:.4f}" if ep5_dice is not None else ("pending..." if r["latest_epoch"] < 5 else "N/A")
            
            # Find latest val dice
            if r["val_dices"]:
                last_val_ep = max(r["val_dices"].keys())
                last_val_dice = r["val_dices"][last_val_ep]
                latest_val_str = f"{last_val_dice:.4f} (ep {last_val_ep})"
            else:
                latest_val_str = "pending..." if r["latest_epoch"] < 5 else "---"

            print(f"{r['file']:<30} {n_str:>4} {seed_str:>5} {ep_str:>11} {loss_str:>8} {ep5_str:>14} {latest_val_str:>16} {best_str:>10}")

    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Monitor Phase 2 validation Dice scores across sweep jobs.")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="Path to scaling_sweep log directory")
    parser.add_argument("--epoch",   type=int, default=None, help="Target specific epoch validation to inspect (e.g. 5)")
    parser.add_argument("--watch",   type=int, default=0, help="Interval in seconds to watch continuously (0 to run once)")
    args = parser.parse_args()

    if args.watch > 0:
        try:
            while True:
                # Clear terminal screen
                os.system("clear" if os.name == "posix" else "cls")
                print(f"[monitor_sweep_val] Watching {args.log_dir} (refreshing every {args.watch}s)... Press Ctrl+C to stop.\n")
                display_status(args.log_dir, target_epoch=args.epoch)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        display_status(args.log_dir, target_epoch=args.epoch)


if __name__ == "__main__":
    main()
