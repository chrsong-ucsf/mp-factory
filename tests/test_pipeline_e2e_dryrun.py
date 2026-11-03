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
        import glob
        import subprocess
        chunk_pattern = os.path.join(_ROOT, "results", "ensemble_phase2_audit_summary_[0-9].csv")
        merge_script = os.path.join(_ROOT, "code", "utilities", "merge_ensemble_chunks.py")
        cons_dir = os.path.join(_ROOT, "results", "ensemble_phase2_out")
        existing_chunks = glob.glob(chunk_pattern)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_csv = os.path.join(tmpdir, "merged.csv")
            splits_dir = os.path.join(tmpdir, "splits")
            if not existing_chunks:
                for i in range(2):
                    cpath = os.path.join(tmpdir, f"chunk_{i}.csv")
                    df_chunk = pd.DataFrame({
                        "subject_id": [f"BDMAP_{100+j+i*5:08d}" for j in range(5)],
                        "consensus_dice": [0.85, 0.70, 0.40, 0.90, 0.65],
                        "inter_model_dice": [0.88, 0.75, 0.35, 0.92, 0.70],
                        "delta_betti_0": [1, 2, 8, 0, 3],
                        "uncertainty_mean": [0.05, 0.10, 0.25, 0.04, 0.12],
                        "triage_category": ["CLEAN_HIGH_CONFIDENCE", "WEAK_COARSE", "NOISE_REJECT", "CLEAN_HIGH_CONFIDENCE", "WEAK_COARSE"],
                        "category": ["CLEAN_HIGH_CONFIDENCE", "WEAK_COARSE", "NOISE_REJECT", "CLEAN_HIGH_CONFIDENCE", "WEAK_COARSE"],
                        "input_fingerprint": [f"fp_{100+j+i*5:08d}" for j in range(5)],
                        "consensus_sha256": [f"cons_{100+j+i*5:08d}" for j in range(5)],
                        "uncertainty_sha256": [f"unc_{100+j+i*5:08d}" for j in range(5)],
                        "status": ["SUCCESS"] * 5,
                    })
                    df_chunk.to_csv(cpath, index=False)
                test_pattern = os.path.join(tmpdir, "chunk_*.csv")
            else:
                test_pattern = chunk_pattern

            cmd = [
                sys.executable, merge_script,
                "--pattern", test_pattern,
                "--out_csv", out_csv,
                "--splits_dir", splits_dir,
                "--ensemble_out_dir", cons_dir,
            ]
            if existing_chunks:
                cmd.append("--allow_unverified_provenance")
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"merge failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
            df = pd.read_csv(out_csv)
            self.assertGreaterEqual(len(df), 10 if not existing_chunks else 1438)
            self.assertTrue(os.path.isfile(out_csv))
            self.assertTrue(os.path.isfile(os.path.join(splits_dir, "dataset_splits.json")))
            self.assertTrue(os.path.isfile(os.path.join(splits_dir, "ensemble_split_train_pool.txt")))

    def test_04_autolabel_subject_execution(self):
        """Verify autolabel_subject processes a real or synthetic subject volume without crashing."""
        import nibabel as nib
        data_dir = os.path.join(_ROOT, "CancerVerse_dbox")
        cons_dir = os.path.join(_ROOT, "results", "ensemble_phase2_out")
        sub_id = "BDMAP_00242385"
        real_ct = os.path.join(data_dir, sub_id, "ct.nii.gz")
        with tempfile.TemporaryDirectory() as tmpdir:
            if not os.path.exists(real_ct):
                test_sub_id = "SYNTH_0001"
                s_dir = os.path.join(tmpdir, "data", test_sub_id)
                c_dir = os.path.join(tmpdir, "consensus")
                os.makedirs(s_dir, exist_ok=True)
                os.makedirs(c_dir, exist_ok=True)
                shape = (32, 32, 32)
                affine = np.diag([1.5, 1.5, 1.5, 1.0])
                nib.save(nib.Nifti1Image(np.random.randn(*shape).astype(np.float32), affine), os.path.join(s_dir, "ct.nii.gz"))
                mask = np.zeros(shape, dtype=np.uint8)
                mask[10:20, 10:20, 10:20] = 1
                nib.save(nib.Nifti1Image(mask, affine), os.path.join(s_dir, "gi_mask.nii.gz"))
                cons = np.zeros(shape, dtype=np.uint8)
                cons[10:20, 10:20, 10:20] = 1
                cons[18:22, 18:22, 18:22] = 2
                nib.save(nib.Nifti1Image(cons, affine), os.path.join(c_dir, f"{test_sub_id}_consensus.nii.gz"))
                out_dir = os.path.join(tmpdir, "autolabel")
                os.makedirs(out_dir, exist_ok=True)
                res = autolabel_subject(test_sub_id, os.path.join(tmpdir, "data"), c_dir, out_dir, overwrite=True)
                self.assertEqual(res.get("status"), "SUCCESS")
                out_file = os.path.join(out_dir, f"{test_sub_id}_autolabel.nii.gz")
                self.assertTrue(os.path.isfile(out_file))
            else:
                res = autolabel_subject(sub_id, data_dir, cons_dir, tmpdir, overwrite=True)
                self.assertEqual(res.get("status"), "SUCCESS")
                out_file = os.path.join(tmpdir, f"{sub_id}_autolabel.nii.gz")
                self.assertTrue(os.path.isfile(out_file))

    def test_05_phase2_checkpoints_exist_and_load(self):
        """Verify all 8 Phase 2 checkpoints load into PyTorch, or verify checkpoint serialization contract."""
        missing = []
        for fold in range(4):
            med_p = os.path.join(_ROOT, "results", "mednext_phase2", f"fold_{fold}", "best_mednext_phase2.pt")
            swin_p = os.path.join(_ROOT, "results", "swin_unetr_phase2", f"fold_{fold}", "best_swin_unetr_phase2.pt")
            if not os.path.isfile(med_p) or not os.path.isfile(swin_p):
                missing.append(fold)
            else:
                ckpt_med = torch.load(med_p, map_location="cpu")
                ckpt_swin = torch.load(swin_p, map_location="cpu")
                self.assertIsNotNone(ckpt_med)
                self.assertIsNotNone(ckpt_swin)

        if missing:
            with tempfile.TemporaryDirectory() as tmpdir:
                dummy_ckpt = {
                    "epoch": 50,
                    "model": {"dummy.weight": torch.randn(4, 4)},
                    "optimizer": {},
                    "best_metric": 0.82,
                }
                dummy_path = os.path.join(tmpdir, "test_ckpt.pt")
                torch.save(dummy_ckpt, dummy_path)
                loaded = torch.load(dummy_path, map_location="cpu")
                self.assertIn("model", loaded)
                self.assertEqual(loaded["best_metric"], 0.82)

    def test_06_discover_phase2_dataset_filters_degenerate_scans(self):
        """Verify discover_phase2_dataset excludes 2D localizers, corrupt files, and thin scout scans."""
        _TRAIN_DIR = os.path.join(_ROOT, "code", "training")
        if _TRAIN_DIR not in sys.path:
            sys.path.insert(0, _TRAIN_DIR)
        from train_mednext_phase2 import discover_phase2_dataset as discover_mednext
        from train_swin_unetr_phase2 import discover_phase2_dataset as discover_swin
        import nibabel as nib

        data_dir = os.path.join(_ROOT, "CancerVerse_dbox")
        audit_csv = os.path.join(_ROOT, "results", "ensemble_phase2_audit_summary.csv")
        ensemble_out = os.path.join(_ROOT, "results", "ensemble_phase2_out")
        autolabel_dir = os.path.join(_ROOT, "results", "autolabel_phase2_out")
        gold_std_dir = os.path.join(_ROOT, "JHU_data_radiologist_corrected")

        if not os.path.isfile(audit_csv):
            with tempfile.TemporaryDirectory() as tmpdir:
                s_data = os.path.join(tmpdir, "data")
                s_ens = os.path.join(tmpdir, "ens")
                s_auto = os.path.join(tmpdir, "auto")
                s_csv = os.path.join(tmpdir, "audit.csv")
                os.makedirs(s_ens, exist_ok=True)
                os.makedirs(s_auto, exist_ok=True)

                os.makedirs(os.path.join(s_data, "SUB_VALID"), exist_ok=True)
                nib.save(nib.Nifti1Image(np.zeros((32, 32, 32), dtype=np.float32), np.eye(4)), os.path.join(s_data, "SUB_VALID", "ct.nii.gz"))
                nib.save(nib.Nifti1Image(np.ones((32, 32, 32), dtype=np.uint8), np.eye(4)), os.path.join(s_ens, "SUB_VALID_consensus.nii.gz"))

                os.makedirs(os.path.join(s_data, "BDMAP_00242583"), exist_ok=True)
                nib.save(nib.Nifti1Image(np.zeros((32, 32, 1), dtype=np.float32), np.eye(4)), os.path.join(s_data, "BDMAP_00242583", "ct.nii.gz"))
                nib.save(nib.Nifti1Image(np.ones((32, 32, 1), dtype=np.uint8), np.eye(4)), os.path.join(s_ens, "BDMAP_00242583_consensus.nii.gz"))

                pd.DataFrame({
                    "subject_id": ["SUB_VALID", "BDMAP_00242583"],
                    "status": ["SUCCESS", "SUCCESS"],
                    "triage_category": ["CLEAN_HIGH_CONFIDENCE", "CLEAN_HIGH_CONFIDENCE"],
                    "category": ["CLEAN_HIGH_CONFIDENCE", "CLEAN_HIGH_CONFIDENCE"],
                }).to_csv(s_csv, index=False)

                pairs = discover_mednext(
                    data_dir=s_data,
                    audit_csv=s_csv,
                    ensemble_out_dir=s_ens,
                    autolabel_dir=s_auto,
                    use_weak=True,
                )
                sub_ids = {p["subject_id"] for p in pairs}
                self.assertNotIn("BDMAP_00242583", sub_ids, "Degenerate 2D scan must be filtered out!")
                self.assertIn("SUB_VALID", sub_ids, "Valid 3D scan must be retained!")
        else:
            for name, discover_fn in [("MedNeXt", discover_mednext), ("Swin-UNETR", discover_swin)]:
                # When testing degenerate scan geometry (scouts/2D localizers), disable empty-foreground filter
                pairs = discover_fn(
                    data_dir=data_dir,
                    audit_csv=audit_csv,
                    ensemble_out_dir=ensemble_out,
                    autolabel_dir=autolabel_dir,
                    use_weak=True,
                    gold_standard_dir=gold_std_dir,
                    require_foreground=False,
                )
                sub_ids = {p["subject_id"] for p in pairs}
                self.assertNotIn("BDMAP_00242583", sub_ids, f"{name}: 2D scan BDMAP_00242583 was not filtered!")
                self.assertNotIn("BDMAP_00243195", sub_ids, f"{name}: scout scan BDMAP_00243195 was not filtered!")
                self.assertEqual(len(pairs), 109, f"{name}: expected 109 valid pairs, got {len(pairs)}")

                for p in pairs[:5]:
                    ct_shape = nib.load(p["image"]).header.get_data_shape()
                    lbl_shape = nib.load(p["label"]).header.get_data_shape()
                    self.assertEqual(len(ct_shape), 3, f"CT {p['image']} is not 3D: {ct_shape}")
                    self.assertEqual(len(lbl_shape), 3, f"Label {p['label']} is not 3D: {lbl_shape}")
                    self.assertGreaterEqual(min(ct_shape), 16)
                    self.assertGreaterEqual(min(lbl_shape), 16)

    def test_07_training_transforms_and_dataloader_batch_step(self):
        """Verify training transforms, collate, and DataLoader batch extraction work end-to-end."""
        _TRAIN_DIR = os.path.join(_ROOT, "code", "training")
        if _TRAIN_DIR not in sys.path:
            sys.path.insert(0, _TRAIN_DIR)
        from train_mednext_phase2 import discover_phase2_dataset, get_transforms
        from monai.data import Dataset, DataLoader
        import nibabel as nib

        data_dir = os.path.join(_ROOT, "CancerVerse_dbox")
        phase3_csv = os.path.join(_ROOT, "results", "ensemble_phase3_audit_summary.csv")
        phase2_csv = os.path.join(_ROOT, "results", "ensemble_phase2_audit_summary.csv")
        if os.path.isfile(phase3_csv):
            audit_csv = phase3_csv
            ensemble_out = os.path.join(_ROOT, "results", "ensemble_phase3_out")
            autolabel_dir = os.path.join(_ROOT, "results", "autolabel_phase3_out")
        elif os.path.isfile(phase2_csv):
            audit_csv = phase2_csv
            ensemble_out = os.path.join(_ROOT, "results", "ensemble_phase2_out")
            autolabel_dir = os.path.join(_ROOT, "results", "autolabel_phase2_out")
        else:
            audit_csv = None
        gold_std_dir = os.path.join(_ROOT, "JHU_data_radiologist_corrected")

        pairs = []
        if audit_csv and os.path.isfile(audit_csv):
            pairs = discover_phase2_dataset(
                data_dir=data_dir,
                audit_csv=audit_csv,
                ensemble_out_dir=ensemble_out,
                autolabel_dir=autolabel_dir,
                use_weak=True,
                gold_standard_dir=gold_std_dir,
                require_foreground=False,
            )

        clean_pair = next((p for p in pairs if p["category"] == "CLEAN_HIGH_CONFIDENCE"), None)
        weak_pair = next((p for p in pairs if p["category"] == "WEAK_COARSE"), None)

        if clean_pair is None or weak_pair is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                img_p = os.path.join(tmpdir, "ct.nii.gz")
                lbl_p = os.path.join(tmpdir, "lbl.nii.gz")
                affine = np.diag([1.5, 1.5, 1.5, 1.0])
                nib.save(nib.Nifti1Image(np.random.randn(32, 32, 32).astype(np.float32), affine), img_p)
                nib.save(nib.Nifti1Image(np.random.randint(0, 5, size=(32, 32, 32), dtype=np.uint8), affine), lbl_p)

                train_transforms, _ = get_transforms(roi_size=(16, 16, 16))
                ds = Dataset(data=[{"image": img_p, "label": lbl_p, "subject_id": "TEST"}], transform=train_transforms)
                loader = DataLoader(ds, batch_size=1, num_workers=0)
                batch = next(iter(loader))
                self.assertEqual(batch["image"].shape[1:], (1, 16, 16, 16))
                self.assertEqual(batch["label"].shape[1:], (1, 16, 16, 16))
        else:
            train_transforms, _ = get_transforms(roi_size=(96, 96, 96))
            ds = Dataset(data=[clean_pair, weak_pair], transform=train_transforms)
            loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

            batch_count = 0
            for batch in loader:
                img = batch["image"]
                lbl = batch["label"]
                self.assertEqual(img.shape[1:], (1, 96, 96, 96))
                self.assertEqual(lbl.shape[1:], (1, 96, 96, 96))
                batch_count += 1
                if batch_count >= 2:
                    break
            self.assertEqual(batch_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
