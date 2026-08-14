import torch
from monai.metrics import compute_hausdorff_distance

def compute_dice_hd95(pred_one_hot, label_one_hot, num_classes):
    dice_vals, hd95_vals = [], []
    for c in range(num_classes):
        p = pred_one_hot[:, c:c+1, ...]
        g = label_one_hot[:, c:c+1, ...]
        inter = (p * g).sum()
        union = p.sum() + g.sum()

        if g.sum() == 0 and p.sum() == 0:
            dice = 1.0 # Both empty -> perfect match
        elif g.sum() == 0 or p.sum() == 0:
            dice = 0.0 # One empty -> complete mismatch
        else:
            dice = (2.0 * inter / (union + 1e-6)).item()
        dice_vals.append(dice)

        hd95 = float("nan")
        if g.sum() > 0 and p.sum() > 0:
            try:
                hd_t = compute_hausdorff_distance(p, g, percentile=95)
                if not (torch.isnan(hd_t).all() or torch.isinf(hd_t).all()):
                    hd95 = hd_t.item()
            except Exception as e:
                pass
        hd95_vals.append(hd95)

    return dice_vals, hd95_vals

pred = torch.zeros(1, 4, 10, 10, 10)
label = torch.zeros(1, 4, 10, 10, 10)
label[:, 0, 1:5, 1:5, 1:5] = 1 # class 0 present
print(compute_dice_hd95(pred, label, 4))
