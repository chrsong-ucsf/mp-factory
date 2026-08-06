"""
test_unified_gastro_dataset.py

Unit tests for the dependency-light (numpy/pandas/torch) parts of
unified_gastro_dataset.py: build_manifest, verify_dataset_integrity,
and the PyTorch Dataset/DataLoader round-trip.
"""

import os
import shutil
import sys
import tempfile
import json
import numpy as np
import pandas as pd

_EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from unified_gastro_dataset import (
    build_manifest,
    verify_dataset_integrity,
    CATEGORY_CLEAN,
    CATEGORY_WEAK,
    IGNORE_INDEX,
)


def _sample_audit():
    return pd.DataFrame([
        {"subject_id": "CLEAN_001", "status": "SUCCESS", "triage_category": "CLEAN_HIGH_CONFIDENCE",
         "mean_consensus_dice": 0.92, "mean_inter_model_dice": 0.89, "max_betti_diff": 0, "mean_uncertainty": 0.04},
        {"subject_id": "WEAK_001", "status": "SUCCESS", "triage_category": "WEAK_COARSE",
         "mean_consensus_dice": 0.73, "mean_inter_model_dice": 0.80, "max_betti_diff": 2, "mean_uncertainty": 0.09},
        {"subject_id": "REJ_001", "status": "SUCCESS", "triage_category": "NOISE_REJECT",
         "mean_consensus_dice": 0.30, "mean_inter_model_dice": 0.33, "max_betti_diff": 9, "mean_uncertainty": 0.18},
    ])


def test_build_manifest_clean():
    with tempfile.TemporaryDirectory() as td:
        csv = pd.DataFrame([{"subject_id":"S001","status":"SUCCESS","triage_category":"CLEAN_HIGH_CONFIDENCE"}])
        p = os.path.join(td, "audit.csv")
        csv.to_csv(p, index=False)
        manifest = build_manifest(p, "/fake/ct", "/fake/cons")
        assert len(manifest) == 1
        r0 = manifest[0]
        assert r0['subject_id'] == 'S001'
        assert r0['ct_path'].endswith("S001/ct.nii.gz")
        assert r0['label_path'].endswith("S001_consensus.nii.gz")
        assert r0['category'] == CATEGORY_CLEAN


def test_build_manifest_with_weak():
    df = _sample_audit()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "audit.csv")
        df.to_csv(p, index=False)
        m = build_manifest(p, "/a", "/c", "/aut", use_weak=True)
        cats = {r['category'] for r in m}
        assert CATEGORY_CLEAN in cats
        assert CATEGORY_WEAK in cats
        # REJECT should not appear at all
        assert all(r['category'] != 'NOISE_REJECT' for r in m)
        assert len(m) == 2


def test_manifest_no_use_weak():
    df = _sample_audit()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "audit.csv")
        df.to_csv(p, index=False)
        m = build_manifest(p, "/x", "/z", "/a", use_weak=False)
        cats = [c['category'] for c in m]
        assert cats == [CATEGORY_CLEAN] * 1
        assert len(m) == 1


def test_manifest_ignore_index_match():
    """must match train_mednext_phase2.py and hard_threshold_autolabel.py"""
    assert IGNORE_INDEX == 255


def test_verify_report_nonexistent():
    """verification correctly identifies missing files"""
    man = [{"subject_id": "s1", "ct_path": "/nope/ct.nii.gz", "label_path": "/nope/lbl.nii.gz",
            "category": "CLEAN_HIGH_CONFIDENCE"}]
    errors = verify_dataset_integrity(man)  # no nilearn needed; just existence checks
    assert len(errors) == 1
    assert "ct missing" in errors[0]["errors"] or "label missing" in (errors[0]["errors"] or [])


def test_roundtrip_dataloader():
    from unified_gastro_dataset import UnifiedGI3Dataset
    from torch.utils.data import DataLoader
    manifest = [
        {"subject_id": "A","ct_path": "/none/ct.nii.gz","label_path": "/none/lbl.nii.gz",
         "category": CATEGORY_CLEAN},
    ]
    ds = UnifiedGI3Dataset(manifest)
    dl = DataLoader(ds, batch_size=1)
    for batch in dl:
        assert "image" in batch and "label" in batch
        break


if __name__ == "__main__":
    print("Running Unified Gastro Dataset Tests...")
    for test_name, fn in {
        "build_manifest base":       test_build_manifest_clean,
        "build_manifest with weak":  test_build_manifest_with_weak,
        "build_manifest without weak": test_manifest_no_use_weak,
        "IGNORE_INDEX = 255":        test_manifest_ignore_index_match,
        "verify missing files":      test_verify_report_nonexistent,
        "DataLoader roundtrip":      test_roundtrip_dataloader,
    }.items():
        try:
            fn()
            print(f"  {test_name}: PASS")
        except Exception as e:
            print(f"  {test_name}: FAIL - {e}")
    print("done")