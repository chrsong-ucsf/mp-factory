"""
test_active_learning.py

Comprehensive unit test suite for active learning module:
  - UncertaintyEstimator (entropy calculation)
  - ActiveLearningPool (pool management, ranking, annotation)
  - ActiveLearningOrchestrator (cycle execution, error handling)
"""

import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from active_learning import (
    UncertaintyEstimator,
    ActiveLearningPool,
    ActiveLearningOrchestrator
)


def test_uncertainty_estimator():
    estimator = UncertaintyEstimator(mode="entropy")
    probs = np.ones((5, 10, 10, 10), dtype=np.float32) / 5.0  # Max uncertainty
    voxel_ent, scan_unc, max_unc = estimator.calculate_entropy(probs)

    assert voxel_ent.shape == (10, 10, 10)
    assert np.isclose(scan_unc, np.log2(5.0), atol=1e-3)
    print("✓ test_uncertainty_estimator passed!")


def test_active_learning_pool():
    subjects = ["SUBJ_01", "SUBJ_02", "SUBJ_03", "SUBJ_04"]
    pool = ActiveLearningPool(subjects)

    pool.update_uncertainty("SUBJ_01", 0.15, "CLEAN_HIGH_CONFIDENCE")
    pool.update_uncertainty("SUBJ_02", 0.85, "REJECT_OR_TRIAGE")
    pool.update_uncertainty("SUBJ_03", 0.45, "REJECT_OR_TRIAGE")
    pool.update_uncertainty("SUBJ_04", 0.05, "CLEAN_HIGH_CONFIDENCE")

    top_cases = pool.query_top_k(k=2)
    assert len(top_cases) == 2
    assert top_cases[0] == "SUBJ_02"  # Highest uncertainty in triage queue
    assert top_cases[1] == "SUBJ_03"

    pool.mark_annotated(["SUBJ_02"])
    assert "SUBJ_02" in pool.labeled_pool
    assert "SUBJ_02" not in pool.triage_queue
    print("✓ test_active_learning_pool passed!")


def test_active_learning_orchestrator():
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "test_audit.csv")
        df = pd.DataFrame([
            {'subject_id': 'S001', 'mean_uncertainty': 0.1, 'triage_category': 'CLEAN_HIGH_CONFIDENCE'},
            {'subject_id': 'S002', 'mean_uncertainty': 0.9, 'triage_category': 'REJECT_OR_TRIAGE'},
            {'subject_id': 'S003', 'mean_uncertainty': 0.6, 'triage_category': 'REJECT_OR_TRIAGE'},
        ])
        df.to_csv(csv_path, index=False)

        pool = ActiveLearningPool(['S001', 'S002', 'S003'])
        estimator = UncertaintyEstimator()
        orchestrator = ActiveLearningOrchestrator(pool, estimator)

        cycle = orchestrator.run_cycle(csv_path, top_k=2)
        assert cycle['cycle'] == 1
        assert cycle['queried_cases'] == ['S002', 'S003']
        print("✓ test_active_learning_orchestrator passed!")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Running Active Learning Unit Test Suite...")
    test_uncertainty_estimator()
    test_active_learning_pool()
    test_active_learning_orchestrator()
    print("\nAll Active Learning Unit Tests Passed Successfully!")
