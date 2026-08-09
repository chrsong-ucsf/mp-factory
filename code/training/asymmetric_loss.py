import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricPDCELoss(nn.Module):
    """
    Asymmetric Partial Cross-Entropy + Dice Loss (Asymmetric PDCE Loss).
    """

    EPS = 1e-7

    def __init__(self,
                 apply_softmax: bool = True,
                 smooth_nr: float = 1e-5,
                 smooth_dr: float = 1e-5,
                 ce_weight: float = 0.5,
                 dice_weight: float = 1.0,
                 alpha: float = 2.0,
                 beta: float = 1.0,
                 ignore_index=-1):
        super(AsymmetricPDCELoss, self).__init__()
        self.apply_softmax = apply_softmax
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if (
            target.ndim == logits.ndim
            and target.shape[1] == 1
            and logits.shape[1] != 1
        ):
            target = target.squeeze(1)

        is_integer_target = (target.ndim + 1 == logits.ndim)

        if is_integer_target and self.ignore_index is not None:
            valid_mask = (target.long() != self.ignore_index)
        else:
            valid_mask = None

        if is_integer_target:
            num_classes = 2 if logits.shape[1] == 1 else logits.shape[1]
            if valid_mask is not None:
                target_safe = torch.where(valid_mask, target.long(), torch.zeros_like(target.long()))
            else:
                target_safe = target.long()
            target_one_hot = F.one_hot(target_safe, num_classes=num_classes)
            dims = [0, logits.ndim - 1] + list(range(1, logits.ndim - 1))
            target_oh = target_one_hot.permute(dims).to(logits.dtype)
        else:
            target_oh = target.to(logits.dtype)

        expanded_binary = logits.shape[1] == 1

        if self.apply_softmax:
            if expanded_binary:
                probs = torch.sigmoid(logits)
                probs = torch.cat([1 - probs, probs], dim=1)
                if target_oh.shape[1] == 1:
                    target_oh = torch.cat([1 - target_oh, target_oh], dim=1)
            else:
                probs = F.softmax(logits, dim=1)
        else:
            probs = logits
            if expanded_binary:
                probs = torch.cat([1 - probs, probs], dim=1)
                if target_oh.shape[1] == 1:
                    target_oh = torch.cat([1 - target_oh, target_oh], dim=1)

        probs = probs.float()
        target_oh = target_oh.float()
        eps = self.EPS
        probs = torch.clamp(probs, min=eps, max=1.0 - eps)

        if valid_mask is not None:
            valid_mask_expanded = valid_mask.unsqueeze(1).to(probs.dtype)
        else:
            valid_mask_expanded = None

        if expanded_binary:
            ce_probs = probs[:, 1:2]
            ce_target = target_oh[:, 1:2]
        else:
            ce_probs = probs
            ce_target = target_oh

        ce_loss = - (self.alpha * ce_target * torch.log(ce_probs) +
                     self.beta * (1.0 - ce_target) * torch.log(1.0 - ce_probs))

        if valid_mask_expanded is not None:
            ce_loss = ce_loss * valid_mask_expanded
            n_valid_elements = valid_mask_expanded.sum() * ce_probs.shape[1]
            n_valid_elements = torch.clamp(n_valid_elements, min=1.0)
            ce_loss = ce_loss.sum() / n_valid_elements
        else:
            ce_loss = ce_loss.mean()

        if valid_mask_expanded is not None:
            probs_masked = probs * valid_mask_expanded
            target_masked = target_oh * valid_mask_expanded
        else:
            probs_masked = probs
            target_masked = target_oh

        reduce_dims = [0, 2, 3] + ([4] if probs.ndim == 5 else [])
        intersection = (probs_masked * target_masked).sum(dim=reduce_dims)
        cardinality = (probs_masked + target_masked).sum(dim=reduce_dims)

        dice_score = (2.0 * intersection + self.smooth_nr) / (cardinality + self.smooth_dr)
        dice_loss = 1.0 - dice_score.mean()

        total_loss = (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)
        return total_loss
