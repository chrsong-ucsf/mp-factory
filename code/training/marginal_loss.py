"""
code/training/marginal_loss.py
==============================
Marginal Loss and Masked Background Loss for partially supervised multi-organ segmentation.
Follows Shi et al. ("Marginal loss and exclusion loss for partially supervised multi-organ segmentation",
arXiv:2007.03868).

Addresses Audit Finding 3:
Prevents false-negative cross-entropy and Dice suppression when unannotated organs are present
in a training image but labeled as background 0 in partial annotations (such as missing small bowel).
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginalDiceCELoss(nn.Module):
    """
    Combined Marginal Cross-Entropy and Masked Dice Loss.

    When some classes are missing/unannotated in a case:
    1. Unlabeled voxels (target == 0) represent P(true_background) U P(missing_organs).
       The target probability for unlabeled voxels is the marginal sum of background and missing classes:
           P_marginal = P(c=0) + sum_{m in missing} P(c=m)
       The cross-entropy loss against 0 is replaced by -log(P_marginal), eliminating false-negative penalties.
    2. Dice loss is computed ONLY over present classes, completely skipping missing classes.
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
        present_classes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) or (B, C, H, W) raw prediction logits.
            targets: (B, 1, ...) or (B, ...) integer ground-truth mask.
            present_classes: list of class integer IDs present in the dataset/mask.
                             If None or contains all classes, standard DiceCE is used.
        Returns:
            Scalar combined loss tensor.
        """
        if targets.ndim == logits.ndim:
            targets = targets.squeeze(1)

        targets = targets.long()
        probs = F.softmax(logits, dim=1)
        valid_mask = (targets != self.ignore_index)

        # Determine missing classes
        all_classes = set(range(self.num_classes))
        if present_classes is not None and len(present_classes) > 0:
            present_set = set(present_classes)
            if 0 not in present_set:
                present_set.add(0)
            missing_classes = sorted(list(all_classes - present_set))
        else:
            present_set = all_classes
            missing_classes = []

        # -------------------------------------------------------------
        # 1. Marginal Cross-Entropy Loss
        # -------------------------------------------------------------
        if not missing_classes:
            # Fully supervised standard CE
            clamped_targets = targets.clamp(0, self.num_classes - 1)
            ce_loss = F.cross_entropy(
                logits, clamped_targets, reduction="none", ignore_index=self.ignore_index
            )
            ce_loss = (ce_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
        else:
            # Partial supervision: marginalize missing classes into background
            # P_marginal = P(0) + sum_{m in missing} P(m)
            p_marginal = probs[:, 0].clone()
            for m in missing_classes:
                p_marginal = p_marginal + probs[:, m]

            log_p_marginal = torch.log(p_marginal.clamp(min=1e-7))

            loss_map = torch.zeros_like(targets, dtype=torch.float32)

            # Unlabeled background voxels: target likelihood is P_marginal
            bg_mask = (targets == 0) & valid_mask
            loss_map[bg_mask] = -log_p_marginal[bg_mask]

            # Confirmed present foreground voxels
            for c in sorted(list(present_set)):
                if c == 0:
                    continue
                c_mask = (targets == c) & valid_mask
                if c_mask.any():
                    log_p_c = torch.log(probs[:, c].clamp(min=1e-7))
                    loss_map[c_mask] = -log_p_c[c_mask]

            ce_loss = loss_map.sum() / valid_mask.sum().clamp(min=1.0)

        # -------------------------------------------------------------
        # 2. Masked Dice Loss (over confirmed present foreground classes)
        # -------------------------------------------------------------
        fg_classes = [c for c in sorted(list(present_set)) if c > 0]
        if not fg_classes:
            dice_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        else:
            dice_losses = []
            for c in fg_classes:
                pred_c = probs[:, c] * valid_mask.float()
                gt_c = (targets == c).float() * valid_mask.float()

                # Sum over all spatial dimensions
                dims = tuple(range(1, pred_c.ndim))
                intersect = (pred_c * gt_c).sum(dim=dims)
                union = (pred_c * pred_c).sum(dim=dims) + (gt_c * gt_c).sum(dim=dims)

                dice_c = 1.0 - (2.0 * intersect + self.smooth) / (union + self.smooth)
                dice_losses.append(dice_c.mean())

            dice_loss = torch.mean(torch.stack(dice_losses))

        return self.alpha * ce_loss + self.beta * dice_loss
