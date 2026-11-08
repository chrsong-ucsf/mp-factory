"""
tests/test_marginal_loss.py
===========================
Unit tests for MarginalDiceCELoss (Shi et al., 2020 formulation).
"""

import torch
import pytest
import numpy as np
from code.training.marginal_loss import MarginalDiceCELoss


def test_marginal_loss_fully_supervised():
    loss_fn = MarginalDiceCELoss(num_classes=5)
    logits = torch.randn(2, 5, 16, 16, 16, requires_grad=True)
    targets = torch.randint(0, 5, (2, 16, 16, 16))

    loss = loss_fn(logits, targets, present_classes=[0, 1, 2, 3, 4])
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None


def test_marginal_loss_missing_class_no_false_negative_penalty():
    """
    If class 3 (small bowel) is missing from present_classes, predicting high logits for class 3
    in an unlabeled region (target=0) should NOT increase the marginal loss because class 3
    is included in P_marginal = P(0) + P(3).
    """
    loss_fn = MarginalDiceCELoss(num_classes=5)
    
    # Batch 1: predict background for target 0
    # Batch 2: predict class 3 for target 0
    B, C, D, H, W = 1, 5, 8, 8, 8
    
    targets = torch.zeros(B, D, H, W, dtype=torch.long)
    
    # Logits favoring background
    logits_bg = torch.zeros(B, C, D, H, W)
    logits_bg[:, 0] = 10.0  # high prob for class 0
    
    # Logits favoring missing class 3
    logits_missing = torch.zeros(B, C, D, H, W)
    logits_missing[:, 3] = 10.0  # high prob for missing class 3
    
    loss_bg = loss_fn(logits_bg, targets, present_classes=[0, 1, 2, 4])
    loss_missing = loss_fn(logits_missing, targets, present_classes=[0, 1, 2, 4])
    
    # Under marginal loss, predicting missing class 3 on unlabeled 0 should yield ~0 loss,
    # identical to predicting true background 0!
    assert abs(loss_bg.item() - loss_missing.item()) < 1e-3


def test_marginal_loss_ignore_index():
    loss_fn = MarginalDiceCELoss(num_classes=5, ignore_index=255)
    logits = torch.randn(1, 5, 8, 8, 8, requires_grad=True)
    targets = torch.full((1, 8, 8, 8), 255, dtype=torch.long)
    
    loss = loss_fn(logits, targets)
    assert loss.item() == 0.0 or abs(loss.item()) < 1e-5
