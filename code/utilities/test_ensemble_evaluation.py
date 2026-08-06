"""
test_ensemble_evaluation.py

Unit test suite for evaluate_multi_model_ensemble.py.
Tests diversity weighting, spatial uncertainty heatmaps, consensus mask assembly,
strict automated-cleansing triage, dataset-split export, and topological triage.
"""

import os
import shutil
import sys
import tempfile
import json
import numpy as np
import pandas as pd

# Ensure the evaluation package dir is importable regardless of CWD.
_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

# triage_case / export_dataset_splits are dependency-light (numpy/pandas only)
# and must be importable even when heavy deps (nibabel/skimage) are absent.
from evaluate_multi_model_ensemble import (
    triage_case,
    export_dataset_splits,
    NOISE_BETTI_DIFF_MAX,
    NOISE_DICE_MIN,
    NOISE_UNCERTAINTY_MAX,
    CLEAN_DICE_MIN,
    CLEAN_INTER_MODEL_MIN,
    CLEAN_BETTI_DIFF_MAX,
    TRAIN_POOL_CATEGORIES,
)


def test_triage_case_rules():
    # --- NOISE_REJECT gates (any one trips it) ---
    # Topological blow-up: |Δβ0| strictly greater than the max.
    cat, act = triage_case(0.95, 0.95, NOISE_BETTI_DIFF_MAX + 1, 0.0)
    assert cat == 'NOISE_REJECT', cat
    assert 'Discard' in act
    # Low mean Dice (< 0.50) even with otherwise perfect agreement.
    cat, _ = triage_case(NOISE_DICE_MIN - 0.01, 0.99, 0, 0.0)
    assert cat == 'NOISE_REJECT', cat
    # High predictive entropy.
    cat, _ = triage_case(0.95, 0.95, 0, NOISE_UNCERTAINTY_MAX + 0.01)
    assert cat == 'NOISE_REJECT', cat

    # --- CLEAN_HIGH_CONFIDENCE (all three conditions satisfied) ---
    cat, act = triage_case(CLEAN_DICE_MIN, CLEAN_INTER_MODEL_MIN,
                           CLEAN_BETTI_DIFF_MAX, 0.05)
    assert cat == 'CLEAN_HIGH_CONFIDENCE', cat
    assert 'GKD' in act or 'Approve' in act

    # --- WEAK_COARSE (survives reject gate but not clean enough) ---
    # Good Dice but inter-model agreement below the clean bar.
    cat, act = triage_case(0.83, CLEAN_INTER_MODEL_MIN - 0.05, 1, 0.05)
    assert cat == 'WEAK_COARSE', cat
    assert 'Ignore' in act or 'Threshold' in act
    # Dice in the coarse band (>=0.50 but <0.82).
    cat, _ = triage_case(0.70, 0.90, 0, 0.05)
    assert cat == 'WEAK_COARSE', cat
    print("\u2713 test_triage_case_rules passed!")


def test_triage_boundary_conditions():
    # Exactly at the CLEAN thresholds -> CLEAN (inclusive >=, <=).
    assert triage_case(CLEAN_DICE_MIN, CLEAN_INTER_MODEL_MIN,
                       CLEAN_BETTI_DIFF_MAX, NOISE_UNCERTAINTY_MAX)[0] == 'CLEAN_HIGH_CONFIDENCE'
    # Exactly at NOISE_DICE_MIN is NOT a reject (strict <), and with high
    # agreement + low betti it lands in WEAK (Dice < CLEAN_DICE_MIN).
    assert triage_case(NOISE_DICE_MIN, 0.99, 0, 0.0)[0] == 'WEAK_COARSE'
    # Betti exactly at NOISE max is not rejected, but exceeds CLEAN_BETTI_DIFF_MAX
    # (2), so despite strong Dice/agreement it is WEAK_COARSE, not CLEAN.
    assert triage_case(0.9, 0.9, NOISE_BETTI_DIFF_MAX, 0.0)[0] == 'WEAK_COARSE'
    assert triage_case(0.9, 0.9, NOISE_BETTI_DIFF_MAX + 1, 0.0)[0] == 'NOISE_REJECT'
    # At the CLEAN betti bound it stays CLEAN; one above drops to WEAK.
    assert triage_case(0.9, 0.9, CLEAN_BETTI_DIFF_MAX, 0.0)[0] == 'CLEAN_HIGH_CONFIDENCE'
    assert triage_case(0.9, 0.9, CLEAN_BETTI_DIFF_MAX + 1, 0.0)[0] == 'WEAK_COARSE'
    # Reject gate takes precedence over clean-looking Dice/inter-model.
    assert triage_case(0.99, 0.99, 99, 0.0)[0] == 'NOISE_REJECT'
    print("\u2713 test_triage_boundary_conditions passed!")


def test_export_dataset_splits():
    temp_dir = tempfile.mkdtemp()
    try:
        rows = [
            {'subject_id': 'CLEAN_A', 'status': 'SUCCESS', 'triage_category': 'CLEAN_HIGH_CONFIDENCE',
             'mean_consensus_dice': 0.90, 'mean_inter_model_dice': 0.88, 'max_betti_diff': 1,
             'mean_uncertainty': 0.05, 'action': 'Auto-Approve for GKD Distillation & VAE'},
            {'subject_id': 'CLEAN_B', 'status': 'SUCCESS', 'triage_category': 'CLEAN_HIGH_CONFIDENCE',
             'mean_consensus_dice': 0.85, 'mean_inter_model_dice': 0.86, 'max_betti_diff': 0,
             'mean_uncertainty': 0.03, 'action': 'Auto-Approve for GKD Distillation & VAE'},
            {'subject_id': 'WEAK_A', 'status': 'SUCCESS', 'triage_category': 'WEAK_COARSE',
             'mean_consensus_dice': 0.70, 'mean_inter_model_dice': 0.80, 'max_betti_diff': 2,
             'mean_uncertainty': 0.08, 'action': 'Apply Hard Thresholding (Set Conflicting Pixels to Ignore Class)'},
            {'subject_id': 'REJECT_A', 'status': 'SUCCESS', 'triage_category': 'NOISE_REJECT',
             'mean_consensus_dice': 0.40, 'mean_inter_model_dice': 0.30, 'max_betti_diff': 9,
             'mean_uncertainty': 0.20, 'action': 'Auto-Exclude (Discard from Training Pool)'},
            # A failed row that must be ignored entirely.
            {'subject_id': 'BAD', 'status': 'ERROR: boom', 'triage_category': np.nan},
        ]
        df = pd.DataFrame(rows)
        splits_dir = os.path.join(temp_dir, "dataset_splits")
        manifest = export_dataset_splits(df, splits_dir, ensemble_out_dir="/fake/ensemble_out")

        # Manifest counts
        assert manifest['total_success'] == 4, manifest['total_success']
        assert manifest['splits']['CLEAN_HIGH_CONFIDENCE']['count'] == 2
        assert manifest['splits']['WEAK_COARSE']['count'] == 1
        assert manifest['splits']['NOISE_REJECT']['count'] == 1
        # Train pool = CLEAN + WEAK, REJECT excluded
        assert manifest['train_pool']['count'] == 3
        assert manifest['train_pool']['categories'] == TRAIN_POOL_CATEGORIES

        # Files exist
        for stem in ("clean_high_confidence", "weak_coarse", "noise_reject"):
            assert os.path.exists(os.path.join(splits_dir, f"ensemble_split_{stem}.txt"))
            assert os.path.exists(os.path.join(splits_dir, f"ensemble_split_{stem}.csv"))
        assert os.path.exists(os.path.join(splits_dir, "ensemble_split_train_pool.txt"))
        assert os.path.exists(os.path.join(splits_dir, "dataset_splits.json"))

        # Clean split content is sorted and correct
        with open(os.path.join(splits_dir, "ensemble_split_clean_high_confidence.txt")) as fh:
            clean_ids = [ln for ln in fh.read().splitlines() if ln]
        assert clean_ids == ['CLEAN_A', 'CLEAN_B'], clean_ids

        # Train pool excludes the rejected subject and the errored row
        with open(os.path.join(splits_dir, "ensemble_split_train_pool.txt")) as fh:
            pool_ids = [ln for ln in fh.read().splitlines() if ln]
        assert 'REJECT_A' not in pool_ids
        assert 'BAD' not in pool_ids
        assert set(pool_ids) == {'CLEAN_A', 'CLEAN_B', 'WEAK_A'}, pool_ids

        # CSV carries consensus_path + metrics
        clean_csv = pd.read_csv(os.path.join(splits_dir, "ensemble_split_clean_high_confidence.csv"))
        assert 'consensus_path' in clean_csv.columns
        assert clean_csv['consensus_path'].iloc[0].endswith('CLEAN_A_consensus.nii.gz')
        assert 'mean_consensus_dice' in clean_csv.columns

        # JSON round-trips
        with open(os.path.join(splits_dir, "dataset_splits.json")) as fh:
            loaded = json.load(fh)
        assert loaded['thresholds']['CLEAN_DICE_MIN'] == CLEAN_DICE_MIN
        print("\u2713 test_export_dataset_splits passed!")
    finally:
        shutil.rmtree(temp_dir)


def _require_nibabel():
    try:
        import nibabel  # noqa: F401
        return True
    except ImportError:
        return False


def test_diversity_weights():
    from evaluate_multi_model_ensemble import compute_diversity_weights
    # Model 1 and Model 2 are identical; Model 3 is different
    m1 = np.ones((30, 30, 30), dtype=np.uint8)
    m2 = np.ones((30, 30, 30), dtype=np.uint8)
    m3 = np.zeros((30, 30, 30), dtype=np.uint8)
    m3[10:20, 10:20, 10:20] = 1

    masks = np.stack([m1, m2, m3], axis=0)
    weights = compute_diversity_weights(masks)

    assert len(weights) == 3
    assert np.isclose(np.sum(weights), 1.0)
    # Model 3 should get a higher weight than correlated models 1 and 2
    assert weights[2] > weights[0]
    print("\u2713 test_diversity_weights passed!")


def test_spatial_uncertainty():
    from evaluate_multi_model_ensemble import compute_spatial_uncertainty
    # 2 classes (BG, FG)
    # Voxel 1: High agreement (all models predict 1.0 for class 1)
    # Voxel 2: High uncertainty (half models 1.0, half models 0.0)
    probs = np.array([
        [[1.0, 0.5]],
        [[1.0, 0.5]],
        [[1.0, 0.0]],
        [[1.0, 0.0]],
    ])  # shape (4, 1, 2)

    ent = compute_spatial_uncertainty(probs)
    # High agreement voxel should have 0 entropy
    assert np.isclose(ent[0, 0], 0.0, atol=1e-3)
    # High uncertainty voxel should have max entropy
    assert ent[0, 1] > 0.5
    print("\u2713 test_spatial_uncertainty passed!")


def test_full_pipeline_mock():
    if not _require_nibabel():
        print("\u26a0 test_full_pipeline_mock skipped (nibabel not installed)")
        return
    import nibabel as nib
    from evaluate_multi_model_ensemble import evaluate_subject
    temp_dir = tempfile.mkdtemp()
    try:
        dir1 = os.path.join(temp_dir, "model1")
        dir2 = os.path.join(temp_dir, "model2")
        out_dir = os.path.join(temp_dir, "out")
        os.makedirs(dir1, exist_ok=True)
        os.makedirs(dir2, exist_ok=True)

        shape = (20, 20, 20)
        affine = np.eye(4)

        mask1 = np.zeros(shape, dtype=np.uint8)
        mask1[5:15, 5:15, 5:15] = 1  # stomach

        mask2 = np.zeros(shape, dtype=np.uint8)
        mask2[6:14, 6:14, 6:14] = 1  # stomach

        f1 = os.path.join(dir1, "SUBJ001_gi_seg.nii.gz")
        f2 = os.path.join(dir2, "SUBJ001_gi_seg.nii.gz")

        nib.save(nib.Nifti1Image(mask1, affine), f1)
        nib.save(nib.Nifti1Image(mask2, affine), f2)

        res = evaluate_subject("SUBJ001", [f1, f2], ["m1", "m2"], out_dir)

        assert res['status'] == 'SUCCESS'
        assert os.path.exists(os.path.join(out_dir, "SUBJ001_consensus.nii.gz"))
        assert os.path.exists(os.path.join(out_dir, "SUBJ001_uncertainty.nii.gz"))
        assert 'mean_consensus_dice' in res
        assert 'triage_category' in res

        # Wiring guard: the metrics evaluate_subject actually reports must feed
        # triage_case consistently, i.e. re-deriving the category from the stored
        # columns reproduces the stored triage_category/action. This locks the
        # exact integration point between metric computation and triage binning.
        for key in ('mean_consensus_dice', 'mean_inter_model_dice',
                    'max_betti_diff', 'mean_uncertainty', 'action'):
            assert key in res, f"evaluate_subject result missing '{key}'"
        recomputed_cat, recomputed_act = triage_case(
            res['mean_consensus_dice'], res['mean_inter_model_dice'],
            res['max_betti_diff'], res['mean_uncertainty'])
        assert recomputed_cat == res['triage_category'], (recomputed_cat, res['triage_category'])
        assert recomputed_act == res['action']
        print("\u2713 test_full_pipeline_mock passed!")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running Ensemble Evaluation Unit Tests...")
    test_triage_case_rules()
    test_triage_boundary_conditions()
    test_export_dataset_splits()
    test_diversity_weights()
    test_spatial_uncertainty()
    test_full_pipeline_mock()
    print("\nAll Ensemble Evaluation Unit Tests Passed Successfully!")
