import torch

def compute_dice_hd95(pred_one_hot, label_one_hot, num_classes):
    dice_vals = []
    for c in range(num_classes):
        p = pred_one_hot[:, c:c+1, ...]
        g = label_one_hot[:, c:c+1, ...]
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        dice = (2.0 * inter / (union + 1e-6)).item()
        dice_vals.append(dice)
    return dice_vals

# Simulating model output
num_classes = 4
B, C, H, W, D = 1, num_classes + 1, 96, 96, 96
v_out = torch.randn(B, C, H, W, D)
probs = torch.softmax(v_out, dim=1)
preds = torch.argmax(probs, dim=1, keepdim=True)
v_lbls = torch.randint(0, num_classes + 1, (B, 1, H, W, D))

pred_oh = torch.zeros(B, num_classes, H, W, D)
lbl_oh = torch.zeros_like(pred_oh)

for c in range(num_classes):
    pred_oh[:, c, ...] = (preds[:, 0, ...] == (c + 1)).float()
    lbl_oh[:, c, ...]  = (v_lbls[:, 0, ...] == (c + 1)).float()

print(compute_dice_hd95(pred_oh, lbl_oh, num_classes))
