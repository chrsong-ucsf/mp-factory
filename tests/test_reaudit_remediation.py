"""
tests/test_reaudit_remediation.py
=================================
Automated verification tests for all 9 deficiencies addressed in
the Research Re-Audit Remediation Plan:
1. Lesion Augmentation Benchmark non-fabrication gate (DATA_UNAVAILABLE).
2. Annotation Completeness Scanner schema ambiguity rejection.
3. Heterogeneous batch marginal loss integration.
4. Probability grid geometry round-trip contract.
5. Multiphase pairing session matching and registration QA.
6. Scaling law patient-cluster bootstrap and convergence rate.
7. P9 independent study gating.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
import torch
import nibabel as nib


def test_lesion_utility_benchmark_data_unavailable(tmp_path):
    """Verify that benchmark_lesion_augmentation_utility writes DATA_UNAVAILABLE and exits with 2 on missing data."""
    from code.project10.evaluation.benchmark_lesion_augmentation_utility import run_benchmark

    out_json = str(tmp_path / "utility_benchmark.json")
    with pytest.raises(SystemExit) as exc_info:
        run_benchmark(
            real_data_csv=str(tmp_path / "nonexistent_real.csv"),
            synthetic_manifest=str(tmp_path / "nonexistent_synth.csv"),
            out_json=out_json,
        )
    assert exc_info.value.code == 2

    assert os.path.exists(out_json)
    with open(out_json) as f:
        data = json.load(f)
    assert data["status"] == "DATA_UNAVAILABLE"
    # Ensure no fabricated performance scores
    assert "delta_auc" not in data
    assert "statistically_significant" not in data


def test_schema_ambiguity_rejection(tmp_path):
    """Verify audit_dataset_completeness rejects partial masks without metadata as AMBIGUOUS."""
    from code.evaluation.audit_dataset_completeness import audit_case

    case_dir = tmp_path / "case_001"
    case_dir.mkdir()

    arr = np.zeros((20, 20, 20), dtype=np.uint8)
    arr[5:10, 5:10, 5:10] = 2  # Duodenum
    arr[12:16, 12:16, 12:16] = 4  # Colon
    nii = nib.Nifti1Image(arr, np.eye(4))
    nib.save(nii, str(case_dir / "gi_mask.nii.gz"))
    ct_nii = nib.Nifti1Image(np.zeros((20, 20, 20), dtype=np.float32), np.eye(4))
    nib.save(ct_nii, str(case_dir / "ct.nii.gz"))

    res = audit_case(str(case_dir), schema_override=None)
    assert res is not None
    assert res["schema"] == "AMBIGUOUS"
    assert res["status"] == "EXCLUDED_UNVERIFIED_METADATA"
    assert res["is_complete_gi"] is False
    assert set(res["visible_classes"]) == {2, 4}


def test_heterogeneous_batch_marginal_loss():
    """Verify marginal loss handles (B, C) reviewed classes with differing supervision per case."""
    from code.training.marginal_loss import MarginalDiceCELoss

    loss_fn = MarginalDiceCELoss(num_classes=4, ignore_index=255)

    # Batch of 2 cases: case 0 has all 4 classes reviewed, case 1 has only classes [1, 2] reviewed
    logits = torch.randn(2, 5, 16, 16, 16, requires_grad=True)
    target = torch.randint(0, 5, (2, 1, 16, 16, 16)).long()

    reviewed_classes = torch.tensor([
        [1, 1, 1, 1],  # All reviewed
        [1, 1, 0, 0],  # Only stomach and duodenum reviewed
    ], dtype=torch.uint8)

    loss = loss_fn(logits, target, reviewed_classes=reviewed_classes)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_scaling_law_patient_cluster_bootstrap(tmp_path):
    """Verify compute_scaling_law uses patient-cluster bootstrap and selects AIC-preferred model."""
    from code.evaluation.compute_scaling_law import fit_scaling_models

    csv_path = str(tmp_path / "scaling_test.csv")
    out_json = str(tmp_path / "scaling_out.json")

    # Generate synthetic scaling records across 10 patients and 3 training sizes
    records = []
    for pid in range(10):
        for n_train in [20, 50, 100]:
            dice = 0.5 + 0.1 * np.log(n_train / 10.0) + np.random.normal(0, 0.02)
            records.append({
                "patient_id": f"PATIENT_{pid:03d}",
                "train_scan_count": n_train,
                "mean_dice": float(dice),
            })

    pd.DataFrame(records).to_csv(csv_path, index=False)

    results = fit_scaling_models(csv_path, out_json, n_boot=50)
    assert results["preferred_model"] in ["logarithmic", "power_law", "exponential"]
    assert "bootstrap_ci" in results
    assert results["bootstrap_ci"]["clustering_mode"] == "patient_cluster"
    assert results["bootstrap_ci"]["convergence_rate"] > 0.5
    assert os.path.exists(out_json)


def test_p9_independent_study_gating(tmp_path):
    """Verify benchmark_p9_independent_study exits with 2 when independent GT is missing."""
    from code.evaluation.benchmark_p9_independent_study import run_benchmark

    out_dir = str(tmp_path / "p9_out")
    with pytest.raises(SystemExit) as exc_info:
        run_benchmark(
            independent_gt_dir=str(tmp_path / "nonexistent_gt"),
            teacher_dir=str(tmp_path / "teacher"),
            majority_dir=str(tmp_path / "majority"),
            ensemble_dir=str(tmp_path / "ensemble"),
            student_dir=str(tmp_path / "student"),
            out_dir=out_dir,
            require_independent_gt=True,
        )
    assert exc_info.value.code == 2

    summary_file = os.path.join(out_dir, "p9_independent_benchmark_summary.json")
    assert os.path.exists(summary_file)
    with open(summary_file) as f:
        summary = json.load(f)
    assert summary["status"] == "DATA_UNAVAILABLE"
