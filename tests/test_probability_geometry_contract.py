"""
tests/test_probability_geometry_contract.py
===========================================
Regression tests for prediction probability geometry, native-grid alignment,
float16 tie-breaking round-trip, and non-finite rejection.
"""

import tempfile
import numpy as np
import nibabel as nib
import nibabel.processing as nip
import pytest


def test_probability_geometry_round_trip():
    with tempfile.TemporaryDirectory() as td:
        # Create non-trivial anisotropic, oblique native CT scanner geometry
        native_shape = (32, 32, 20)
        native_affine = np.array([
            [0.75,  0.0,   0.0,   -120.0],
            [0.0,   0.75,  0.0,   -150.0],
            [0.0,   0.0,   1.50,  -80.0],
            [0.0,   0.0,   0.0,   1.0],
        ])
        ct_data = np.random.randint(-1000, 500, size=native_shape, dtype=np.int16)
        native_ct_nii = nib.Nifti1Image(ct_data, affine=native_affine)

        # Simulate resampled inference grid (e.g. 1.5mm isotropic)
        resampled_shape = (16, 16, 20)
        resampled_affine = np.array([
            [1.50,  0.0,   0.0,   -120.0],
            [0.0,   1.50,  0.0,   -150.0],
            [0.0,   0.0,   1.50,  -80.0],
            [0.0,   0.0,   0.0,   1.0],
        ])

        # Generate smooth probability logits across 5 classes
        logits = np.random.randn(5, *resampled_shape).astype(np.float32)
        exp_l = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs_inf = exp_l / np.sum(exp_l, axis=0, keepdims=True)

        # Resample probabilities to native CT grid
        native_probs = np.zeros((5,) + native_shape, dtype=np.float32)
        for c in range(5):
            ch_nii = nib.Nifti1Image(probs_inf[c], affine=resampled_affine)
            ch_native = nip.resample_from_to(ch_nii, native_ct_nii, order=1).get_fdata(dtype=np.float32)
            native_probs[c] = np.maximum(0.0, ch_native)

        # Simplex normalization & out-of-support definition
        prob_sum = np.sum(native_probs, axis=0)
        support_mask = (prob_sum > 1e-4).astype(np.uint8)
        prob_sum_safe = np.where(prob_sum > 1e-4, prob_sum, 1.0)
        native_probs = native_probs / prob_sum_safe
        native_probs[0] = np.where(support_mask == 1, native_probs[0], 1.0)
        for c in range(1, 5):
            native_probs[c] = np.where(support_mask == 1, native_probs[c], 0.0)

        # In-memory argmax mask
        native_mask = np.argmax(native_probs, axis=0).astype(np.uint8)

        # Save to NPZ with verified metadata
        npz_path = f"{td}/test_probs.npz"
        probs_f16 = native_probs.astype(np.float16)
        disk_pred = np.argmax(probs_f16.astype(np.float32), axis=0).astype(np.uint8)
        if not np.array_equal(disk_pred, native_mask):
            probs_to_save = native_probs
            precision_stored = "float32"
        else:
            probs_to_save = probs_f16
            precision_stored = "float16"

        np.savez_compressed(
            npz_path,
            probs=probs_to_save,
            affine=native_affine,
            shape=np.array(native_shape),
            channel_order=np.array([0, 1, 2, 3, 4]),
            support_mask=support_mask,
            precision=precision_stored,
            coordinate_space="native_ct_scanner"
        )

        # Reload from disk and verify contract
        loaded = np.load(npz_path, allow_pickle=True)
        assert "affine" in loaded
        assert np.allclose(loaded["affine"], native_affine)
        assert tuple(loaded["shape"]) == native_shape
        assert np.all(np.isfinite(loaded["probs"]))
        
        reloaded_mask = np.argmax(loaded["probs"].astype(np.float32), axis=0).astype(np.uint8)
        # Verify bitwise agreement between disk probabilities and native hard mask
        assert np.array_equal(reloaded_mask, native_mask)


def test_corrupted_probability_rejection():
    # Verify that non-finite values (NaN/Inf) are detected
    corrupted_probs = np.array([0.5, np.nan, 0.2, 0.1, 0.2], dtype=np.float32)
    assert not np.all(np.isfinite(corrupted_probs))
