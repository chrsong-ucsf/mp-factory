"""
test_hard_threshold_autolabel.py

Unit tests for the dependency-light (numpy/pandas-only) parts of
hard_threshold_autolabel.py: resolve_human_label, IGNORE_INDEX constant,
and the full autolabel_subject end-to-end (with nibabel/nilearn when available).
"""

import os
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from hard_threshold_autolabel import (
    resolve_human_label,
    autolabel_subject,
    IGNORE_INDEX,
    ORGAN_MAP,
)


def test_resolve_human_label_none():
    kind, path = resolve_human_label("/tmp/fake_dataset", "SUBJ_UNKNOWN")
    assert kind is None
    assert path is None


def test_resolve_human_label_combined():
    with tempfile.TemporaryDirectory() as td:
        sub5 = os.path.join(td, "SUBJ_005")
        os.makedirs(sub5)
        gt = os.path.join(sub5, "gi_mask.nii.gz")
        open(gt, "w").close()  # dummy token file; resolve just checks existaence
        kind, path = resolve_human_label(td, "SUBJ_005")
        assert kind == "combined"
        assert path == gt


def test_resolve_human_label_segmentations():
    with tempfile.TemporaryDirectory() as td:
        sub6 = os.path.join(td, "SUBJ_006", "segmentations")
        os.makedirs(sub6)
        open(os.path.join(sub6, "stomach.nii.gz"), "w").close()
        # no gi_mask.nii.gz -> falls back to segmentations/
        kind, path = resolve_human_label(td, "SUBJ_006")
        assert kind == "segmentations"
        assert path.endswith("segmentations")


def test_ignore_index_consistency():
    """v must match train_mednext_phase2.py IGNORE_INDEX."""
    assert IGNORE_INDEX == 255, "IGNORE_INDEX must be 255 to match train_mednext_phase2.py"


def test_organ_classes():
    """The 4 GI classes must remain stable."""
    assert ORGAN_MAP == {1: 'stomach', 2: 'duodenum', 3: 'small_bowel', 4: 'colon'}


def _require_nibabel():
    try:
        import nibabel  # noqa: F401
        return True
    except ImportError:
        return False


def test_autolabel_e2e_synthetic():
    """End-to-end synthetic autolabel pipeline: create simulated masks and verify.
    Requires nibabel + nilearn for full end-to-end validation before cluster deploy."""
    if not _require_nibabel():
        print("\u26a0 test_autolabel_e2e_synthetic skipped (nibabel not installed)")
        return
    import nibabel as nib

    with tempfile.TemporaryDirectory() as td:
        data8 = os.path.join(td, "CancerVerse_dbox")
        cons8 = os.path.join(td, "ensemble_out")
        out8  = os.path.join(td, "autolabel_out")
        os.makedirs(os.path.join(data8, "SUBJ_001"), exist_ok=True)
        os.makedirs(cons8, exist_ok=True)
        os.makedirs(out8, exist_ok=True)

        shape = (20, 20, 20)
        aff = np.eye(4)

        # Human coarse: stomach (1) + colon (4) slightly different positions
        human = np.zeros(shape, dtype=np.uint8)
        human[5:15, 5:15, 5:10] = 1   # stomach part
        human[5:10, 5:10, 10:15] = 4   # colon part

        # Consensus: similar but slightly shifted stomach, colon extra region
        cons = np.zeros(shape, dtype=np.uint8)
        cons[6:15, 6:15, 5:10] = 1
        cons[5:12, 5:12, 10:15] = 4

        nib.save(nib.Nifti1Image(human, aff), os.path.join(data8, "SUBJ_001", "gi_mask.nii.gz"))
        nib.save(nib.Nifti1Image(cons, aff), os.path.join(cons8, "SUBJ_001_consensus.nii.gz"))

        res = autolabel_subject("SUBJ_001", data8, cons8, out8)
        assert res['status'] == 'SUCCESS', res['status']

        # Load and inspect output
        out_nii = nib.load(res['output_path'])
        out_arr = np.asanyarray(out_nii.dataobj).astype(np.int32)

        # Basic expectations: output shape equals original, conflicts get 255
        assert out_arr.shape == shape
        fg = human > 0
        conflict_idx = fg & (human != cons)
        assert np.any(conflict_idx), "synthetic data should produce at least one conflict"

        # Check that all conflicting voxels are set to IGNORE_INDEX = 255
        assert (out_arr[conflict_idx] == IGNORE_INDEX).all()
        # All non-conflicting voxels should remain the human label
        agreements = fg & (human == cons)
        assert (out_arr[agreements] == human[agreements]).all()

        # Check that ignore statistics match
        assert res['n_ignore_voxels'] == conflict_idx.sum()
        assert res['pct_ignore'] > 0

        print("\u2713 test_autolabel_e2e_synthetic passed!")


if __name__ == "__main__":
    print("Running Hard-Threshold Autolabel Unit Tests...")
    test_resolve_human_label_none()
    test_resolve_human_label_combined()
    test_resolve_human_label_segmentations()
    test_ignore_index_consistency()
    test_organ_classes()
    test_autolabel_e2e_synthetic()
    print("\nAll Hard-Threshold Autolabel Unit Tests Passed Successfully!")