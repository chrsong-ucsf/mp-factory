"""
code/training/marginal_loss.py
==============================
Marginal Loss and Masked Background Loss for partially supervised multi-organ segmentation.
Follows Shi et al. ("Marginal loss and exclusion loss for partially supervised multi-organ segmentation",
arXiv:2007.03868).

Addresses Research Re-Audit Item 3:
- Prevents false-negative cross-entropy and Dice suppression when unannotated organs are present
  in a training image but labeled as background 0 in partial annotations (such as missing small bowel).
- Supports heterogeneous batches via a (B, C) binary reviewed_classes tensor.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginalDiceCELoss(nn.Module):
    """
    Combined Marginal Cross-Entropy and Masked Dice Loss.

    When some classes are missing/unannotated in a case:
    1. Unlabeled voxels (target == 0) represent P(true_background) U P(missing_organs).
       The target probability for unlabeled voxels is the marginal sum of background and missing classes:
           P_marginal = P(c=0) + sum_{m in unreviewed} P(c=m)
       The cross-entropy loss against 0 is replaced by -log(P_marginal), eliminating false-negative penalties.
    2. Dice loss is computed ONLY over confirmed reviewed classes, skipping unreviewed classes.
    3. Heterogeneous batches are supported via a (B, C) binary tensor where reviewed_classes[b, c] == 1
       indicates organ c was reviewed in case b.
    """
    def __init__(
        self,
        num_classes: int = 5,
        ignore_index: int = 255,
        alpha: float = 1.0,
        beta: float = 1.0,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        present_classes: Optional[Union[List[int], torch.Tensor]] = None,
        reviewed_classes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) or (B, C, H, W) raw prediction logits.
            targets: (B, 1, ...) or (B, ...) integer ground-truth mask.
            present_classes: optional 1D list or tensor of class integer IDs present across the batch.
            reviewed_classes: optional (B, C) binary tensor indicating reviewed classes per case.
        Returns:
            Scalar combined loss tensor.
        """
        if targets.ndim == logits.ndim:
            targets = targets.squeeze(1)

        targets = targets.long()
        probs = F.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index)
        B, C = logits.shape[0], logits.shape[1]

        # Standardize reviewed_classes to a (B, C) boolean/byte mask
        if reviewed_classes is not None:
            if reviewed_classes.ndim == 1:
                reviewed_mask = reviewed_classes.unsqueeze(0).expand(B, -1).bool().to(device=logits.device)
            else:
                reviewed_mask = reviewed_classes.bool().to(device=logits.device)

            # If shape is (B, C - 1) representing foreground classes only, prepend True for background
            if reviewed_mask.shape[1] == C - 1:
                bg_col = torch.ones((B, 1), dtype=torch.bool, device=logits.device)
                reviewed_mask = torch.cat([bg_col, reviewed_mask], dim=1)
        elif present_classes is not None:
            if isinstance(present_classes, list):
                present_t = torch.tensor(present_classes, device=logits.device, dtype=torch.long)
            else:
                present_t = present_classes.to(device=logits.device, dtype=torch.long)
            reviewed_mask = torch.zeros((B, C), dtype=torch.bool, device=logits.device)
            reviewed_mask[:, present_t] = True
            reviewed_mask[:, 0] = True  # Background is always in the evaluated space
        else:
            # Full supervision across all classes
            reviewed_mask = torch.ones((B, C), dtype=torch.bool, device=logits.device)

        # -------------------------------------------------------------
        # 1. Marginal Cross-Entropy Loss (per-batch element)
        # -------------------------------------------------------------
        loss_map = torch.zeros_like(targets, dtype=torch.float32)

        for b in range(B):
            b_valid = valid_mask[b]
            if not b_valid.any():
                continue

            rev_b = reviewed_mask[b]
            unreviewed_indices = torch.where(~rev_b)[0].tolist()

            # Background voxels (targets[b] == 0): marginalize unreviewed classes
            bg_mask = (targets[b] == 0) & b_valid
            if bg_mask.any():
                p_marginal = probs[b, 0].clone()
                for m in unreviewed_indices:
                    p_marginal = p_marginal + probs[b, m]
                log_p_marginal = torch.log(p_marginal.clamp(min=1e-7))
                loss_map[b, bg_mask] = -log_p_marginal[bg_mask]

            # Reviewed foreground voxels
            reviewed_fg = torch.where(rev_b)[0].tolist()
            for c in reviewed_fg:
                if c == 0:
                    continue
                c_mask = (targets[b] == c) & b_valid
                if c_mask.any():
                    log_p_c = torch.log(probs[b, c].clamp(min=1e-7))
                    loss_map[b, c_mask] = -log_p_c[c_mask]

        valid_count = valid_mask.sum().clamp(min=1.0)
        ce_loss = loss_map.sum() / valid_count

        # -------------------------------------------------------------
        # 2. Masked Dice Loss (over confirmed reviewed foreground classes)
        # -------------------------------------------------------------
        dice_losses = []
        dims = tuple(range(1, logits.ndim - 1))  # spatial dimensions

        for c in range(1, C):
            # Evaluate Dice only for batch elements where class c was reviewed
            active_b = reviewed_mask[:, c]
            if not active_b.any():
                continue

            pred_c = probs[active_b, c] * valid_mask[active_b].float()
            gt_c = (targets[active_b] == c).float() * valid_mask[active_b].float()

            intersect = (pred_c * gt_c).sum(dim=dims)
            union = (pred_c * pred_c).sum(dim=dims) + (gt_c * gt_c).sum(dim=dims)
            dice_c = (2.0 * intersect + self.smooth) / (union + self.smooth)
            dice_losses.append(1.0 - dice_c.mean())

        if dice_losses:
            dice_loss = torch.stack(dice_losses).mean()
        else:
            dice_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        total_loss = self.alpha * ce_loss + self.beta * dice_loss
        return total_loss
