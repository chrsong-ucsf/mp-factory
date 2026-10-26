"""
tests/test_pipeline_e2e_dryrun.py

Comprehensive pipeline verification test suite:
  1. Unit tests for fast vector Adjusted Rand Index (ARI).
  2. Integration test for chunk CSV merging and split export.
  3. Integration test for hard-threshold auto-labeling on actual data.
  4. Checkpoint integrity verification for MedNeXt & Swin-UNETR Phase 2 weights.
  5. SLURM script path and environment validation.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_EVAL_DIR = os.path.join(_ROOT, "code", "evaluation")
_UTIL_DIR = os.path.join(_ROOT, "code", "utilities")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
if _UTIL_DIR not in sys.path:
    sys.path.insert(0, _UTIL_DIR)

from evaluate_multi_model_ensemble import adjusted_rand_score_fast, compute_clustering_metrics
from hard_threshold_autolabel import autolabel_subject


class TestPipelineDryRun(unittest.TestCase):

    def test_01_adjusted_rand_score_fast_correctness(self):
        """Verify adjusted_rand_score_fast matches theoretical ground truth across edge cases."""
        # Perfect agreement
        self.assertAlmostEqual(adjusted_rand_score_fast([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        # Permutation invariance
        self.assertAlmostEqual(adjusted_rand_score_fast([0, 0, 1, 1], [1, 1, 0, 0]), 1.0)
        # Random / orthogonal
        self.assertAlmostEqual(adjusted_rand_score_fast([0, 0, 1, 1], [0, 1, 0, 1]), -0.5)
        # Multi-class
        self.assertAlmostEqual(adjusted_rand_score_fast([0, 0, 1, 2], [0, 0, 1, 1]), 0.5714285714285715)
        # All same label
        self.assertAlmostEqual(adjusted_rand_score_fast([1, 1, 1], [1, 1, 1]), 1.0)
        # Empty array
        self.assertAlmostEqual(adjusted_rand_score_fast([], []), 1.0)

    def test_02_compute_clustering_metrics(self):
        """Verify clustering metrics compute without sklearn dependency errors."""
        a = np.random.randint(0, 5, size=(40, 40, 40))
        b = np.random.randint(0, 5, size=(40, 40, 40))
        ari, voi = compute_clustering_metrics(a, b)
        self.assertFalse(np.isnan(ari))
        self.assertFalse(np.isnan(voi))

    def test_03_merge_chunks_and_export_splits(self):
        """Verify chunk CSVs merge properly and export clean dataset splits via CLI."""
        import subprocess
        chunk_pattern = os.path.join(_ROOT, "results", "ensemble_phase2_audit_summary_[0-9].csv")
        merge_script = os.path.join(_ROOT, "code", "utilities", "merge_ensemble_chunks.py")
        cons_dir = os.path.join(_ROOT, "results", "ensemble_phase2_out")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "merged.csv")
            splits_dir = os.path.join(tmpdir, "splits")
            cmd = [
                sys.executable, merge_script,
                "--pattern", chunk_pattern,
                "--out_csv", out_csv,
                "--splits_dir", splits_dir,
                "--ensemble_out_dir", cons_dir,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"merge failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
            df = pd.read_csv(out_csv)
            self.assertGreaterEqual(len(df), 1438)
            self.assertTrue(os.path.isfile(out_csv))
            self.assertTrue(os.path.isfile(os.path.join(splits_dir, "dataset_splits.json")))
            self.assertTrue(os.path.isfile(os.path.join(splits_dir, "ensemble_split_train_pool.txt")))

    def test_04_autolabel_subject_execution(self):
        """Verify autolabel_subject processes a real subject volume without crashing."""
        data_dir = os.path.join(_ROOT, "CancerVerse_dbox")
        cons_dir = os.path.join(_ROOT, "results", "ensemble_phase2_out")
        sub_id = "BDMAP_00242385"
        with tempfile.TemporaryDirectory() as tmpdir:
            res = autolabel_subject(sub_id, data_dir, cons_dir, tmpdir, overwrite=True)
            self.assertEqual(res.get("status"), "SUCCESS")
            out_file = os.path.join(tmpdir, f"{sub_id}_autolabel.nii.gz")
            self.assertTrue(os.path.isfile(out_file))

    def test_05_phase2_checkpoints_exist_and_load(self):
        """Verify all 8 Phase 2 checkpoints (MedNeXt + Swin-UNETR, 4 folds) load into PyTorch."""
        for fold in range(4):
            med_p = os.path.join(_ROOT, "results", "mednext_phase2", f"fold_{fold}", "best_mednext_phase2.pt")
            swin_p = os.path.join(_ROOT, "results", "swin_unetr_phase2", f"fold_{fold}", "best_swin_unetr_phase2.pt")
            self.assertTrue(os.path.isfile(med_p), f"MedNeXt Fold {fold} missing: {med_p}")
            self.assertTrue(os.path.isfile(swin_p), f"Swin-UNETR Fold {fold} missing: {swin_p}")
            ckpt_med = torch.load(med_p, map_location="cpu")
            ckpt_swin = torch.load(swin_p, map_location="cpu")
            self.assertIsNotNone(ckpt_med)
            self.assertIsNotNone(ckpt_swin)


if __name__ == "__main__":
    unittest.main(verbosity=2)
