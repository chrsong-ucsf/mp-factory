"""
test_ensemble_evaluation.py

Unit test suite for evaluate_multi_model_ensemble.py.
Tests diversity weighting, spatial uncertainty heatmaps, consensus mask assembly, and topological triage.
"""

import os
import shutil
import tempfile
import numpy as np
import nibabel as nib
from evaluate_multi_model_ensemble import (
    compute_diversity_weights,
    compute_spatial_uncertainty,
    evaluate_subject
)


def test_diversity_weights():
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
    print("✓ test_diversity_weights passed!")


def test_spatial_uncertainty():
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
    print("✓ test_spatial_uncertainty passed!")


def test_full_pipeline_mock():
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
        print("✓ test_full_pipeline_mock passed!")

    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running Ensemble Evaluation Unit Tests...")
    test_diversity_weights()
    test_spatial_uncertainty()
    test_full_pipeline_mock()
    print("\nAll Ensemble Evaluation Unit Tests Passed Successfully!")
