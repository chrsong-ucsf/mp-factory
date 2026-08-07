"""
test_asymmetric_loss.py

Unit tests for AsymmetricPDCELoss:
- probability clamping (no NaNs)
- ignore_index masking (ignored voxels yield zero gradient)
- false negative vs false positive asymmetric penalties
- gradient flow to backbone parameters
- 3D spatial dimensions
"""

import torch
import torch.nn as nn
import sys
import os

# Ensure src is in python path to load the loss
_this_dir = os.path.dirname(os.path.abspath(__file__))
# _this_dir = utilities
# _this_dir/.. = code
# _this_dir/../.. = mp-factory
# _this_dir/../../.. = 02_Projects
# _this_dir/../../../.. = research_su26
_vault_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_dir))))
if _vault_root not in sys.path:
    sys.path.insert(0, _vault_root)

from src.segmentation.losses.asymmetric_loss import AsymmetricPDCELoss


def test_probability_clamping_no_nan():
    print("Running test_probability_clamping_no_nan...")
    # Extreme logits to test the 1e-7 clamping (would cause log(0) -> NaN if unclamped)
    loss_fn = AsymmetricPDCELoss(apply_softmax=True, alpha=1.0, beta=1.0)
    
    logits = torch.tensor([[[[-1e4, 1e4], [-1e4, 1e4]]]], dtype=torch.float32, requires_grad=True)  # (1, 1, 2, 2)
    target = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.float32)  # (1, 2, 2)
    
    loss = loss_fn(logits, target)
    assert not torch.isnan(loss), "Loss should not be NaN with extreme logits"
    assert not torch.isinf(loss), "Loss should not be Inf with extreme logits"
    
    loss.backward()
    assert not torch.isnan(logits.grad).any(), "Gradients should not be NaN"
    print("✓ test_probability_clamping_no_nan passed")


def test_ignore_index_masks_voxels():
    print("Running test_ignore_index_masks_voxels...")
    loss_fn = AsymmetricPDCELoss(apply_softmax=True, ignore_index=255)
    
    logits = torch.randn(2, 3, 4, 4, 4, requires_grad=True)  # (B, C, D, H, W)
    
    # Target with some ignored voxels
    target = torch.randint(0, 3, (2, 4, 4, 4))
    target[:, 1:3, 1:3, 1:3] = 255  # Set center block to ignore
    
    loss = loss_fn(logits, target)
    loss.backward()
    
    # Gradients for ignored voxels should be exactly 0
    ignored_grads = logits.grad[:, :, 1:3, 1:3, 1:3]
    assert torch.all(ignored_grads == 0), "Ignored voxels should have zero gradients"
    
    # Gradients for non-ignored voxels should be non-zero (most of them anyway)
    valid_grads = logits.grad[:, :, 0, 0, 0]
    assert torch.any(valid_grads != 0), "Valid voxels should have non-zero gradients"
    print("✓ test_ignore_index_masks_voxels passed")


def test_asymmetric_penalty_fn_gt_fp():
    print("Running test_asymmetric_penalty_fn_gt_fp...")
    loss_fn = AsymmetricPDCELoss(apply_softmax=False, alpha=2.0, beta=1.0, ce_weight=1.0, dice_weight=0.0)
    
    # Binary case, assume probs directly (apply_softmax=False)
    # False Negative case: true label=1, predicted prob=0.1
    logits_fn = torch.full((1, 1, 2, 2, 2), 0.1)
    target_fn = torch.ones((1, 1, 2, 2, 2))
    
    # False Positive case: true label=0, predicted prob=0.9 (same error magnitude)
    logits_fp = torch.full((1, 1, 2, 2, 2), 0.9)
    target_fp = torch.zeros((1, 1, 2, 2, 2))
    
    loss_fn_val = loss_fn(logits_fn, target_fn)
    loss_fp_val = loss_fn(logits_fp, target_fp)
    
    # Since alpha (2.0) > beta (1.0), False Negatives should incur higher penalty
    assert loss_fn_val.item() > loss_fp_val.item(), "FN penalty should be greater than FP penalty when alpha > beta"
    
    # Reverse test
    loss_fn_rev = AsymmetricPDCELoss(apply_softmax=False, alpha=1.0, beta=2.0, ce_weight=1.0, dice_weight=0.0)
    assert loss_fn_rev(logits_fp, target_fp).item() > loss_fn_rev(logits_fn, target_fn).item(), "FP penalty should be greater when beta > alpha"
    print("✓ test_asymmetric_penalty_fn_gt_fp passed")


def test_gradient_flows_to_backbone():
    print("Running test_gradient_flows_to_backbone...")
    # Simulate a tiny backbone
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv3d(1, 4, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(4, 5, kernel_size=3, padding=1)  # 5 classes
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            return self.conv2(x)
            
    model = TinyBackbone()
    loss_fn = AsymmetricPDCELoss(apply_softmax=True, ignore_index=-1)
    
    inputs = torch.randn(2, 1, 8, 8, 8)
    target = torch.randint(0, 5, (2, 8, 8, 8))
    
    outputs = model(inputs)
    loss = loss_fn(outputs, target)
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.all(param.grad == 0), f"Parameter {name} has zero gradient"
        
    print("✓ test_gradient_flows_to_backbone passed")


def test_multiclass_3d_shape():
    print("Running test_multiclass_3d_shape...")
    loss_fn = AsymmetricPDCELoss(apply_softmax=True)
    
    # (B, C, D, H, W) for logits, (B, D, H, W) for targets
    logits = torch.randn(2, 5, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 5, (2, 8, 8, 8))
    
    loss = loss_fn(logits, target)
    assert loss.ndim == 0, "Loss should be a scalar"
    loss.backward()
    assert logits.grad.shape == logits.shape, "Gradient shape should match logits shape"
    print("✓ test_multiclass_3d_shape passed")


def test_ignore_index_minus_one_noop():
    print("Running test_ignore_index_minus_one_noop...")
    loss_fn_noop = AsymmetricPDCELoss(apply_softmax=True, ignore_index=-1)
    loss_fn_explicit = AsymmetricPDCELoss(apply_softmax=True, ignore_index=255)
    
    logits = torch.randn(2, 5, 4, 4, 4)
    target = torch.randint(0, 5, (2, 4, 4, 4))  # No 255 values, so ignore_index=255 is a noop here too
    
    loss1 = loss_fn_noop(logits, target)
    loss2 = loss_fn_explicit(logits, target)
    
    assert torch.isclose(loss1, loss2), "ignore_index=-1 should behave identically to an absent ignore_index"
    print("✓ test_ignore_index_minus_one_noop passed")


if __name__ == "__main__":
    print("Running AsymmetricPDCELoss Unit Tests...")
    test_probability_clamping_no_nan()
    test_ignore_index_masks_voxels()
    test_asymmetric_penalty_fn_gt_fp()
    test_gradient_flows_to_backbone()
    test_multiclass_3d_shape()
    test_ignore_index_minus_one_noop()
    print("\nAll AsymmetricPDCELoss tests passed successfully!")
