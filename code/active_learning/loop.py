"""
active_learning/loop.py

Core Active Learning Module for Triage, Uncertainty Estimation, Pool Ranking, and Iterative Model Retraining.

Classes:
  - UncertaintyEstimator: Calculates spatial entropy, MC dropout variance, and multi-model disagreement.
  - ActiveLearningPool: Manages labeled, unlabeled, and radiologist-review pools; ranks scans by uncertainty.
  - ActiveLearningOrchestrator: Executes active learning cycles, querying top K most valuable cases.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn


class UncertaintyEstimator:
    """Estimates voxel-level and scan-level uncertainty metrics."""
    def __init__(self, mode="entropy", mc_samples=5):
        self.mode = mode
        self.mc_samples = mc_samples

    def calculate_entropy(self, probs):
        """
        Computes spatial predictive entropy H(p) = - sum_c p_c log2(p_c)
        probs: (C, H, W, D) or (N_models, C, H, W, D)
        """
        if probs.ndim == 5:
            probs = np.mean(probs, axis=0)
        epsilon = 1e-7
        clamped = np.clip(probs, epsilon, 1.0 - epsilon)
        voxel_entropy = -np.sum(clamped * np.log2(clamped), axis=0)  # (H, W, D)
        scan_uncertainty = float(np.mean(voxel_entropy))
        max_uncertainty = float(np.max(voxel_entropy))
        return voxel_entropy, scan_uncertainty, max_uncertainty

    def calculate_mc_dropout(self, model, inputs, device):
        """Computes Monte Carlo Dropout variance across multiple forward passes."""
        model.train()  # Enable dropout at test time
        preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                out = model(inputs)
                prob = torch.softmax(out, dim=1).cpu().numpy()
                preds.append(prob[0])
        preds_stack = np.stack(preds, axis=0)  # (mc_samples, C, H, W, D)
        var_map = np.var(preds_stack, axis=0).mean(axis=0)  # (H, W, D)
        scan_var = float(np.mean(var_map))
        return var_map, scan_var


class ActiveLearningPool:
    """Manages dataset partitioning, uncertainty ranking, and targeted radiologist query selection."""
    def __init__(self, all_subject_ids):
        self.unlabeled_pool = list(all_subject_ids)
        self.labeled_pool = []
        self.triage_queue = []  # Scans flagged for radiologist review
        self.uncertainty_scores = {}

    def update_uncertainty(self, subject_id, score, triage_category=None):
        """Records scan-level uncertainty score and category."""
        self.uncertainty_scores[subject_id] = score
        if triage_category == 'REJECT_OR_TRIAGE':
            if subject_id not in self.triage_queue:
                self.triage_queue.append(subject_id)

    def query_top_k(self, k=10):
        """
        Ranks unlabeled/triage pool by uncertainty and returns top K most valuable cases
        for targeted radiologist review.
        """
        # Prioritize scans in the triage queue first, sorted by uncertainty score
        candidates = self.triage_queue if self.triage_queue else self.unlabeled_pool
        sorted_candidates = sorted(
            candidates,
            key=lambda sid: self.uncertainty_scores.get(sid, 0.0),
            reverse=True
        )
        selected = sorted_candidates[:k]
        return selected

    def mark_annotated(self, subject_ids):
        """Moves annotated scans from unlabeled/triage pool into labeled pool."""
        for sid in subject_ids:
            if sid in self.unlabeled_pool:
                self.unlabeled_pool.remove(sid)
            if sid in self.triage_queue:
                self.triage_queue.remove(sid)
            if sid not in self.labeled_pool:
                self.labeled_pool.append(sid)


class ActiveLearningOrchestrator:
    """Coordinates the Active Learning loop: evaluation, ranking, querying, and updating pools."""
    def __init__(self, pool, estimator):
        self.pool = pool
        self.estimator = estimator
        self.history = []

    def run_cycle(self, audit_csv_path, top_k=10):
        """
        Executes one Active Learning iteration:
          1. Parses model comparison / ensemble audit CSV.
          2. Ranks dataset by predictive uncertainty & topological disagreement.
          3. Selects top K most valuable edge cases for radiologist annotation.
        """
        if not os.path.exists(audit_csv_path):
            raise FileNotFoundError(f"Audit CSV not found: {audit_csv_path}")

        import pandas as pd
        df = pd.read_csv(audit_csv_path)

        for _, row in df.iterrows():
            sid = str(row['subject_id'])
            unc = float(row.get('mean_uncertainty', 0.0))
            cat = str(row.get('triage_category', 'UNKNOWN'))
            self.pool.update_uncertainty(sid, unc, cat)

        # Query top K most valuable cases for radiologist review
        queried = self.pool.query_top_k(k=top_k)

        cycle_info = {
            'cycle': len(self.history) + 1,
            'queried_cases': queried,
            'num_triage_queue': len(self.pool.triage_queue),
            'num_labeled': len(self.pool.labeled_pool),
            'num_unlabeled': len(self.pool.unlabeled_pool)
        }
        self.history.append(cycle_info)

        print(f"\n[Active Learning Cycle {cycle_info['cycle']}]")
        print(f"  - Queried Top {len(queried)} Edge Cases for Radiologist Review: {queried}")
        print(f"  - Remaining Triage Queue: {len(self.pool.triage_queue)} | Labeled: {len(self.pool.labeled_pool)}")

        return cycle_info
