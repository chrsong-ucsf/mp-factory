#!/usr/bin/env python3
"""
evaluate_jhu_radiologist_set.py

Standardized CPU/GPU evaluator against JHU radiologist-corrected ground truth masks.
Uses the unified GT_ORGAN_MAP, MODEL_LABEL_REMAP, and physical coordinate space resampling
to guarantee identical evaluations across CPU and GPU execution paths.
"""

import argparse
import os
import sys

# Ensure repo root is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from code.evaluation.evaluate_jhu_radiologist_gpu import run_gpu_evaluation, find_jhu_dir


def evaluate_jhu_set(jhu_dir: str, pred_dir: str, out_csv: str, model_name: str = "MedNeXt"):
    model_names = [model_name]
    pred_dirs = [pred_dir]
    run_gpu_evaluation(jhu_dir, pred_dirs, model_names, out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against JHU Radiologist Corrected NRRD set.")
    parser.add_argument("--jhu_dir", default="/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected", help="Path to JHU radiologist corrected NRRD directory")
    parser.add_argument("--pred_dir", required=True, help="Path to directory containing model prediction NIfTI files")
    parser.add_argument("--model_name", default="MedNeXt", help="Name of model being evaluated (for label remapping)")
    parser.add_argument("--out_csv", default="/mnt/scratch/user/chrsong/mp-factory/results/jhu_radiologist_audit_results.csv", help="Path to save evaluation summary CSV")
    args = parser.parse_args()

    evaluate_jhu_set(args.jhu_dir, args.pred_dir, args.out_csv, model_name=args.model_name)
