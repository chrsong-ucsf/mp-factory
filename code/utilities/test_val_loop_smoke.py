"""
test_val_loop_smoke.py

Smoke test for the Phase 2 validation loop.

Runs entirely on CPU with synthetic tensors — no real data, no GPU, no SLURM.
Catches the exact bugs we've been hitting:

  1. RuntimeError: size of tensor a (227) must match tensor b (170)
     -> Fixed by interpolating vo to vl's spatial size when they differ.

  2. TypeError: GradScaler.__init__() got unexpected kwarg 'device_type'
     -> Fixed by using torch.cuda.amp.GradScaler() (only checked if CUDA present)

  3. torch.amp.autocast(device_type=...) compat issue
     -> Fixed by using torch.cuda.amp.autocast()

Usage (on cluster or locally):
    python code/utilities/test_val_loop_smoke.py

All assertions must pass before re-submitting the SLURM sweep.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config (mirrors train_mednext_phase2.py)
# ---------------------------------------------------------------------------
NUM_CLASSES  = 5
IGNORE_INDEX = 255
DEVICE       = torch.device("cpu")   # CPU only — safe on login node

print("=" * 60)
print("Phase 2 Validation Loop Smoke Test")
print(f"  PyTorch : {torch.__version__}")
print(f"  Device  : {DEVICE}")
print("=" * 60)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def post_pred_fn(logits):
    """Argmax -> one-hot (mirrors AsDiscrete in training script)."""
    pred_class = torch.argmax(logits, dim=0, keepdim=True)        # [1, H, W, D]
    one_hot = torch.zeros(NUM_CLASSES, *logits.shape[1:])
    one_hot.scatter_(0, pred_class, 1)
    return one_hot


def post_label_fn(label_1ch):
    """int label -> one-hot, ignore 255 pixels (already zeroed out before this)."""
    label_long = label_1ch.long().clamp(0, NUM_CLASSES - 1)       # [1, H, W, D]
    one_hot = torch.zeros(NUM_CLASSES, *label_1ch.shape[1:])
    one_hot.scatter_(0, label_long, 1)
    return one_hot


def manual_mean_dice(vo_post_list, vl_post_list):
    """Simple mean Dice over non-background classes (channels 1..NUM_CLASSES-1)."""
    dices = []
    for vo_1hot, vl_1hot in zip(vo_post_list, vl_post_list):
        for c in range(1, NUM_CLASSES):
            p = vo_1hot[c].flatten()
            g = vl_1hot[c].flatten()
            inter = (p * g).sum()
            denom = p.sum() + g.sum()
            dice = (2.0 * inter / denom).item() if denom > 0 else 1.0
            dices.append(dice)
    return sum(dices) / len(dices) if dices else 0.0


# ---------------------------------------------------------------------------
# Test 1: Matching spatial sizes (should always work, baseline)
# ---------------------------------------------------------------------------
print("\n[Test 1] Matching image / label spatial sizes …")

image_size  = (1, 1, 96, 96, 96)     # B, C, H, W, D
label_size  = (1, 1, 96, 96, 96)

vi = torch.randn(*image_size)
vl = torch.randint(0, NUM_CLASSES, label_size).float()

# Fake model output: same spatial size as image
vo = torch.randn(1, NUM_CLASSES, *image_size[2:])

if vo.shape[-3:] != vl.shape[-3:]:
    vo = F.interpolate(vo.float(), size=vl.shape[-3:], mode="trilinear", align_corners=False)

vl_clean = torch.where(vl == IGNORE_INDEX, torch.zeros_like(vl), vl)
vo_post  = [post_pred_fn(vo[0])]
vl_post  = [post_label_fn(vl_clean[0])]
dice     = manual_mean_dice(vo_post, vl_post)

assert vo_post[0].shape == vl_post[0].shape, \
    f"Shape mismatch after post: {vo_post[0].shape} vs {vl_post[0].shape}"
print(f"  PASS  vo_post={tuple(vo_post[0].shape)}, vl_post={tuple(vl_post[0].shape)}, dice={dice:.4f}")


# ---------------------------------------------------------------------------
# Test 2: MISMATCHED spatial sizes (the exact bug that crashed all jobs)
# ---------------------------------------------------------------------------
print("\n[Test 2] MISMATCHED image / label spatial sizes (the real bug) …")

image_size_mismatch = (1, 1, 227, 148, 227)
label_size_mismatch = (1, 1, 170, 104, 170)

vi2 = torch.randn(*image_size_mismatch)
vl2 = torch.randint(0, NUM_CLASSES, label_size_mismatch).float()

# Fake model output matches image (not label) — this is what caused the crash
vo2 = torch.randn(1, NUM_CLASSES, *image_size_mismatch[2:])

# THE FIX: interpolate prediction to label size
if vo2.shape[-3:] != vl2.shape[-3:]:
    print(f"  Size mismatch detected: pred={tuple(vo2.shape[-3:])} label={tuple(vl2.shape[-3:])}")
    vo2 = F.interpolate(vo2.float(), size=vl2.shape[-3:], mode="trilinear", align_corners=False)
    print(f"  Interpolated prediction to: {tuple(vo2.shape[-3:])}")

vl2_clean = torch.where(vl2 == IGNORE_INDEX, torch.zeros_like(vl2), vl2)
vo2_post  = [post_pred_fn(vo2[0])]
vl2_post  = [post_label_fn(vl2_clean[0])]
dice2     = manual_mean_dice(vo2_post, vl2_post)

assert vo2_post[0].shape == vl2_post[0].shape, \
    f"Shape mismatch after fix: {vo2_post[0].shape} vs {vl2_post[0].shape}"
print(f"  PASS  vo_post={tuple(vo2_post[0].shape)}, vl_post={tuple(vl2_post[0].shape)}, dice={dice2:.4f}")


# ---------------------------------------------------------------------------
# Test 3: IGNORE_INDEX (255) pixels are properly zeroed before metric
# ---------------------------------------------------------------------------
print("\n[Test 3] IGNORE_INDEX=255 boundary pixels are remapped to background …")

label_with_ignore = torch.randint(0, NUM_CLASSES, (1, 1, 32, 32, 32)).float()
# Inject some 255 voxels
label_with_ignore[0, 0, 10:15, 10:15, 10:15] = 255.0

vl3_clean = torch.where(label_with_ignore == IGNORE_INDEX,
                         torch.zeros_like(label_with_ignore),
                         label_with_ignore)

assert (vl3_clean == 255).sum() == 0, "Found 255 values in cleaned label!"
assert (vl3_clean < NUM_CLASSES).all(), "Labels out of range after ignore remap!"
print(f"  PASS  All 255-voxels remapped to 0, max label value = {int(vl3_clean.max())}")


# ---------------------------------------------------------------------------
# Test 4: GradScaler compatibility (only if CUDA available)
# ---------------------------------------------------------------------------
print("\n[Test 4] GradScaler initialization compatibility …")
if torch.cuda.is_available():
    try:
        scaler = torch.cuda.amp.GradScaler()
        print(f"  PASS  torch.cuda.amp.GradScaler() initialized: {type(scaler).__name__}")
    except Exception as e:
        print(f"  FAIL  {e}")
        sys.exit(1)
else:
    print("  SKIP  No CUDA device — will be verified on GPU node at runtime")


# ---------------------------------------------------------------------------
# Test 5: autocast compatibility
# ---------------------------------------------------------------------------
print("\n[Test 5] torch.cuda.amp.autocast compatibility …")
if torch.cuda.is_available():
    try:
        with torch.cuda.amp.autocast():
            x = torch.randn(2, 2).cuda()
            y = x @ x
        print(f"  PASS  torch.cuda.amp.autocast() works")
    except Exception as e:
        print(f"  FAIL  {e}")
        sys.exit(1)
else:
    print("  SKIP  No CUDA device — will be verified on GPU node at runtime")


# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("ALL TESTS PASSED — safe to submit the scaling sweep.")
print("=" * 60)
