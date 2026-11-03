#!/usr/bin/env python3
"""
run_scaling_sweep.py

Dataset scaling law analysis for GI organ segmentation in mp-factory.

Trains MedNeXt-B across sub-cohort sizes N = [4, 10, 15, 20, 30, 50] with
Asymmetric Partial-CE/Dice loss (AsymmetricPDCELoss), logging per-epoch Dice, HD95,
and radiologist true-Dice metrics to structured JSON + CSV files.

Usage:
    python code/training/run_scaling_sweep.py \
        --data_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
        --mask_dir /mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox \
        --cohort_sizes 4 10 15 20 30 50 \
        --loss asymmetric \
        --epochs 30 \
        --radiologist_gt_dir /mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected \
        --output_dir /mnt/scratch/user/chrsong/mp-factory/results/scaling_sweep

    # Smoke-test with synthetic data:
    python code/training/run_scaling_sweep.py \
        --generate_dummy --cohort_sizes 4 10 --epochs 2 --output_dir /tmp/sweep_test
"""

import os
import sys
import gc
import glob
import json
import hashlib
import csv
import time
import random
import shutil
import tempfile
import argparse
import numpy as np
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from scipy import ndimage
from scipy.spatial import KDTree

# Memory-safe CUDA allocator — belt-and-suspenders (SLURM script also exports this)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import monai
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    EnsureTyped,
    Lambdad,
    MapTransform,
    ResampleToMatchd,
)
from monai.data import Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance
from monai.losses import DiceCELoss

# Robust local imports for MedNeXt-B and AsymmetricPDCELoss within mp-factory
try:
    from code.training.mednext import MedNeXtB
except ImportError:
    try:
        from mednext import MedNeXtB
    except ImportError:
        try:
            from src.segmentation.models.mednext import MedNeXtB
        except ImportError:
            MedNeXtB = None

try:
    from nnunet_mednext import create_mednext_v1
except ImportError:
    try:
        from mednext.create_mednext_v1 import create_mednext_v1
    except ImportError:
        create_mednext_v1 = None

try:
    from code.training.asymmetric_loss import AsymmetricPDCELoss
except ImportError:
    try:
        from asymmetric_loss import AsymmetricPDCELoss
    except ImportError:
        try:
            from src.segmentation.losses.asymmetric_loss import AsymmetricPDCELoss
        except ImportError:
            AsymmetricPDCELoss = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_COHORT_SIZES = [4, 10, 15, 20, 30, 50]
DEFAULT_EPOCHS = 30
DEFAULT_ROI_SIZE = (96, 96, 96)
NUM_CLASSES = 4  # stomach, duodenum, small bowel, colon (label map 1-4)
ORGAN_NAMES = ["stomach", "duodenum", "small_bowel", "colon"]

JHU_SCAN_TO_PATIENT_MAP = {
    "BDMAP_00242113": "PATIENT_BDMAP_P1", "BDMAP_00242114": "PATIENT_BDMAP_P1",
    "BDMAP_00242115": "PATIENT_BDMAP_P1", "BDMAP_00242116": "PATIENT_BDMAP_P1",
    "BDMAP_00242131": "PATIENT_BDMAP_P2", "BDMAP_00242132": "PATIENT_BDMAP_P2", "BDMAP_00242133": "PATIENT_BDMAP_P2",
    "BDMAP_00242134": "PATIENT_BDMAP_P3", "BDMAP_00242135": "PATIENT_BDMAP_P4", "BDMAP_00242136": "PATIENT_BDMAP_P4",
    "BDMAP_00394224": "PATIENT_BDMAP_P5",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def determine_mask_schema(mask_path, data_dir, mask_dir=None):
    """Explicitly determines label schema without erroneously mapping 1 (stomach) to 0 (esophagus) (F01/F14)."""
    p_lower = mask_path.lower()
    d_lower = (data_dir or "").lower()
    m_lower = (mask_dir or "").lower()

    if "jhu" in d_lower or "jhu" in m_lower or "jhu" in p_lower:
        return "jhu"

    # Canonical 5-class GI masks or single organ masks (where 1=stomach, 2=duo, 3=sb, 4=colon)
    base_name = os.path.basename(mask_path).lower()
    if base_name in ("gi_mask.nii.gz", "gi_mask_merged.nii.gz", "stomach.nii.gz", "duodenum.nii.gz", "small_bowel.nii.gz", "colon.nii.gz"):
        return "gi_5class"

    if any(k in p_lower for k in ("autolabel", "consensus", "canonical", "gi_5class")):
        return "gi_5class"

    # Raw TotalSegmentator multi-organ masks (including TotalSeg run on BDMAP cohorts, e.g. totalseg_gi_masks_bdmap)
    # Check totalseg before bdmap because totalseg_gi_masks_bdmap was produced by TotalSegmentator (ml=True)
    if "totalseg" in p_lower or "totalseg" in m_lower or "totalseg" in d_lower:
        if "v1" in p_lower or "v1" in m_lower or "total_v1" in p_lower:
            return "totalseg_v1"
        return "totalseg_v2"

    # Raw BDMAP multi-organ annotations (where 1=esophagus, 2=stomach, 3=duo, 4=sb, 5=sb, 6=colon)
    if ("bdmap" in m_lower or "bdmap" in d_lower) and "gi_mask" not in p_lower:
        return "bdmap"

    return "gi_5class"


def compute_file_sha256(path: str) -> str:
    """Streaming SHA-256 computation (1MB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def extract_patient_id(subject_id: str) -> str:
    """Authoritative patient ID extraction ensuring CV_XXXX and BDMAP_XXXX are not collapsed into 'CV' (Finding 3)."""
    s = str(subject_id).strip()
    if s in JHU_SCAN_TO_PATIENT_MAP:
        return JHU_SCAN_TO_PATIENT_MAP[s]
    parts = s.split("_")
    if len(parts) >= 2 and parts[0] in ("CV", "CancerVerse", "BDMAP"):
        return f"{parts[0]}_{parts[1]}"
    return s


def assemble_or_get_mask(mask_dir, patient_id, ct_path):
    """Returns path to combined multi-organ mask, merging individual organ masks from segmentations/ if needed (Finding 2/6)."""
    mask_path = os.path.join(mask_dir, patient_id, "gi_mask.nii.gz")
    if os.path.exists(mask_path):
        return mask_path

    seg_dir = os.path.join(mask_dir, patient_id, "segmentations")
    if os.path.isdir(seg_dir):
        organ_map = {
            1: ["stomach.nii.gz", "stomach.nii"],
            2: ["duodenum.nii.gz", "duodenum.nii"],
            3: ["small_bowel.nii.gz", "small_intestine.nii.gz", "small_bowel.nii"],
            4: ["colon.nii.gz", "colon.nii", "large_bowel.nii.gz"]
        }
        found_organs = {}
        for organ_idx, organ_files in organ_map.items():
            for of in organ_files:
                op = os.path.join(seg_dir, of)
                if os.path.exists(op):
                    found_organs[organ_idx] = op
                    break

        if found_organs:
            # If only stomach exists and no other organs are present
            if len(found_organs) == 1 and 1 in found_organs and len(os.listdir(seg_dir)) == 1:
                return found_organs[1]

            merged_cached = os.path.join(mask_dir, patient_id, "gi_mask_merged.nii.gz")
            meta_cached = merged_cached + ".meta.json"

            # Compute current fingerprints of source inputs
            source_hashes = {}
            for o_idx, op in sorted(found_organs.items()):
                source_hashes[str(o_idx)] = compute_file_sha256(op)
            if os.path.exists(ct_path):
                source_hashes["ct"] = compute_file_sha256(ct_path)

            # Check if cache is strictly valid (Finding 4)
            cache_valid = False
            if os.path.exists(merged_cached) and os.path.exists(meta_cached):
                try:
                    with open(meta_cached, "r") as mf:
                        cached_meta = json.load(mf)
                    if (
                        cached_meta.get("source_hashes") == source_hashes
                        and cached_meta.get("schema_version") == "v2_merged"
                        and "merged_sha256" in cached_meta
                    ):
                        # Verify the cached file has not been replaced or tampered with
                        if compute_file_sha256(merged_cached) == cached_meta["merged_sha256"]:
                            # Verify readability, discrete canonical labels, and geometry agreement with CT
                            m_nii = nib.load(merged_cached)
                            ct_nii = nib.load(ct_path)
                            if (
                                m_nii.shape[:3] == ct_nii.shape[:3]
                                and np.allclose(m_nii.affine, ct_nii.affine, atol=1e-3)
                            ):
                                m_arr = np.asanyarray(m_nii.dataobj)
                                uniq = np.unique(m_arr)
                                if np.all(np.equal(np.mod(uniq, 1), 0)):
                                    uniq_set = set(int(v) for v in uniq)
                                    if uniq_set.issubset({0, 1, 2, 3, 4}):
                                        present_organs = uniq_set - {0}
                                        if present_organs.issubset(set(found_organs.keys())):
                                            cache_valid = True
                except Exception:
                    cache_valid = False

            if cache_valid:
                return merged_cached

            # Regenerate merged mask from source organs onto CT grid
            ct_nii = nib.load(ct_path)
            shape = ct_nii.shape[:3]
            merged_data = np.zeros(shape, dtype=np.uint8)

            for organ_idx, op in sorted(found_organs.items()):
                # Fail closed on corrupt/unreadable organ masks (F06)
                o_nii = nib.load(op)
                o_arr = np.asanyarray(o_nii.dataobj)
                if o_arr.shape[:3] != shape or not np.allclose(o_nii.affine, ct_nii.affine, atol=1e-3):
                    import nibabel.processing as nip
                    o_nii_res = nip.resample_from_to(o_nii, ct_nii, order=0)
                    o_arr = np.asanyarray(o_nii_res.dataobj)
                mask_vox = (o_arr > 0)
                merged_data[mask_vox] = organ_idx

            merged_nii = nib.Nifti1Image(merged_data, ct_nii.affine, ct_nii.header)
            save_path = merged_cached
            meta_path = meta_cached
            try:
                nib.save(merged_nii, save_path)
                merged_sha = compute_file_sha256(save_path)
                with open(meta_path, "w") as mf:
                    json.dump({
                        "source_hashes": source_hashes,
                        "merged_sha256": merged_sha,
                        "schema_version": "v2_merged",
                        "organs": list(found_organs.keys()),
                        "shape": list(shape),
                    }, mf)
                return save_path
            except Exception:
                import tempfile
                td = tempfile.gettempdir()
                save_path = os.path.join(td, f"{patient_id}_gi_mask_merged.nii.gz")
                meta_path = save_path + ".meta.json"
                nib.save(merged_nii, save_path)
                merged_sha = compute_file_sha256(save_path)
                with open(meta_path, "w") as mf:
                    json.dump({
                        "source_hashes": source_hashes,
                        "merged_sha256": merged_sha,
                        "schema_version": "v2_merged",
                        "organs": list(found_organs.keys()),
                        "shape": list(shape),
                    }, mf)
                return save_path

    stomach_p = os.path.join(mask_dir, patient_id, "segmentations", "stomach.nii.gz")
    if os.path.exists(stomach_p):
        return stomach_p

    return None


def find_cases(data_dir, mask_dir):
    """Return list of {image, label, name, schema} dicts for paired NIfTI cases with verified schema."""
    cases = []

    # Layout 1: data_dir/BDMAP_*/ct.nii.gz + mask_dir/BDMAP_*/gi_mask.nii.gz
    for ct_path in sorted(glob.glob(os.path.join(data_dir, "BDMAP_*", "ct.nii.gz"))):
        patient_id = os.path.basename(os.path.dirname(ct_path))
        mask_path = assemble_or_get_mask(mask_dir, patient_id, ct_path)
        if mask_path and os.path.exists(mask_path):
            schema = determine_mask_schema(mask_path, data_dir, mask_dir)
            cases.append({"image": ct_path, "label": mask_path, "name": patient_id, "schema": schema})

    # Layout 2: CancerVerse / mp-factory subject directories (sub/ct.nii.gz + sub/gi_mask.nii.gz)
    if not cases:
        for ct_path in sorted(glob.glob(os.path.join(data_dir, "*", "ct.nii.gz"))):
            patient_id = os.path.basename(os.path.dirname(ct_path))
            mask_path = os.path.join(data_dir, patient_id, "gi_mask.nii.gz")
            if os.path.exists(mask_path):
                schema = determine_mask_schema(mask_path, data_dir, mask_dir)
                cases.append({"image": ct_path, "label": mask_path, "name": patient_id, "schema": schema})

    # Layout 3: Flat layout data_dir/*.nii.gz + mask_dir/*.nii.gz
    if not cases:
        for img_path in sorted(glob.glob(os.path.join(data_dir, "*.nii.gz"))):
            fname = os.path.basename(img_path)
            lbl_path = os.path.join(mask_dir, fname)
            if os.path.exists(lbl_path):
                schema = determine_mask_schema(lbl_path, data_dir, mask_dir)
                cases.append({"image": img_path, "label": lbl_path, "name": fname.split(".")[0], "schema": schema})

    return cases


def create_dummy_cohort(tmp_dir, n_cases=50):
    """Generate tiny dummy NIfTI volumes for smoke-testing without real data."""
    img_dir = os.path.join(tmp_dir, "images")
    lbl_dir = os.path.join(tmp_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cases = []
    for i in range(n_cases):
        name = f"dummy_{i:04d}"
        vol = np.random.randn(32, 32, 32).astype(np.float32)
        lbl = np.zeros((32, 32, 32), dtype=np.int32)
        lbl[4:12, 4:12, 4:12] = 1   # stomach
        lbl[12:18, 4:12, 4:12] = 2  # duodenum
        lbl[18:24, 4:12, 4:12] = 3  # small bowel
        lbl[24:30, 4:12, 4:12] = 4  # colon

        img_path = os.path.join(img_dir, f"{name}.nii.gz")
        lbl_path = os.path.join(lbl_dir, f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(vol, np.eye(4)), img_path)
        nib.save(nib.Nifti1Image(lbl, np.eye(4)), lbl_path)
        cases.append({"image": img_path, "label": lbl_path, "name": name})

    return cases


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

_eval_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)
from schema_mapping import normalize_to_gi_classes

class RemapGILabelTargets(MapTransform):
    """
    Remaps loaded labels to 4-class GI targets using canonical schema_mapping.
    Carries explicit schema provenance per record (F11/F14).
    """
    def __init__(self, keys=["label"], default_schema=None):
        super().__init__(keys)
        self.default_schema = default_schema

    def __call__(self, data):
        d = dict(data)
        schema = d.get("schema", self.default_schema)
        for key in self.keys:
            if key in d:
                lbl = d[key]
                is_torch = torch.is_tensor(lbl)
                arr = lbl.cpu().numpy() if is_torch else np.asanyarray(lbl)
                remapped = normalize_to_gi_classes(arr, schema=schema, preserve_ignore_index=True)
                from monai.data import MetaTensor
                if isinstance(lbl, MetaTensor):
                    d[key] = MetaTensor(
                        torch.from_numpy(remapped).to(lbl.device) if is_torch else remapped,
                        affine=lbl.affine,
                        meta=getattr(lbl, "meta", None),
                    )
                elif is_torch:
                    d[key] = torch.from_numpy(remapped).to(lbl.device)
                else:
                    d[key] = remapped
        return d


def build_transforms(roi_size=DEFAULT_ROI_SIZE, is_train=True):
    """Build MONAI transform pipeline with standard Spacing/Intensity before final ROI cropping (F02/F14)."""
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        RemapGILabelTargets(keys=["label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ResampleToMatchd(keys=["label"], key_dst="image", mode="nearest", padding_mode="zeros"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    ]
    if is_train:
        transforms.extend([
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=2, neg=1, num_samples=2,
                image_key="image", image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ])
    transforms.append(EnsureTyped(keys=["image", "label"]))
    return Compose(transforms)


def build_model(num_classes, device):
    """Build MedNeXt-B; fall back to create_mednext_v1 or UNet if needed."""
    model = None
    if MedNeXtB is not None:
        try:
            model = MedNeXtB(in_channels=1, n_channels=32, n_classes=num_classes + 1)
        except Exception:
            pass

    if model is None and create_mednext_v1 is not None:
        try:
            model = create_mednext_v1(
                num_classes=num_classes + 1,
                model_id="B",
                kernel_size=3,
                deep_supervision=False,
            )
        except Exception:
            pass

    if model is None:
        print("Warning: MedNeXt import failed — falling back to MONAI UNet.")
        model = UNet(
            spatial_dims=3, in_channels=1, out_channels=num_classes + 1,
            channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
        )

    model = model.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Enabling torch.nn.DataParallel across {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    return model



# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

# Shared 3-D face-connectivity structuring element for surface extraction
_STRUCT3D = ndimage.generate_binary_structure(3, 1)


def _extract_3d_boundaries(mask: np.ndarray) -> np.ndarray | None:
    """Single-voxel surface via morphological erosion XOR. None if mask is empty."""
    if mask.sum() == 0:
        return None
    return mask ^ ndimage.binary_erosion(mask, structure=_STRUCT3D)


def _compute_hd95_numpy(pred_np: np.ndarray, gt_np: np.ndarray,
                        spacing: tuple = (1.5, 1.5, 2.0)) -> float:
    """KDTree-based HD95 on surface voxels, physical mm. Returns nan if either mask empty."""
    b_pred = _extract_3d_boundaries(pred_np)
    b_gt   = _extract_3d_boundaries(gt_np)
    if b_pred is None or b_gt is None:
        return float("nan")
    sp = np.asarray(spacing, dtype=np.float64)
    pts_pred = np.argwhere(b_pred).astype(np.float64) * sp
    pts_gt   = np.argwhere(b_gt).astype(np.float64)   * sp
    d_p2g, _ = KDTree(pts_gt).query(pts_pred)
    d_g2p, _ = KDTree(pts_pred).query(pts_gt)
    return max(float(np.percentile(d_p2g, 95)), float(np.percentile(d_g2p, 95)))


def compute_dice_hd95(pred_one_hot, label_one_hot, num_classes, voxel_spacing=(1.5, 1.5, 2.0)):
    """Returns per-class Dice and HD95 lists.

    Dice:
      both-empty -> NaN (absent organ excluded to prevent score inflation, F15)
      one-empty  -> 0.0
      standard   -> 2|intersection| / (|pred| + |gt|)

    HD95 uses morphological surface extraction + KDTree (O(N log N), spacing-aware).
    """
    dice_vals, hd95_vals = [], []
    spacing = voxel_spacing

    for c in range(num_classes):
        p = pred_one_hot[:, c:c+1, ...]
        g = label_one_hot[:, c:c+1, ...]
        p_sum = p.sum()
        g_sum = g.sum()

        if g_sum == 0 and p_sum == 0:
            dice_vals.append(float("nan"))
        elif g_sum == 0 or p_sum == 0:
            dice_vals.append(0.0)
        else:
            inter = (p * g).sum()
            dice = (2.0 * inter / (p_sum + g_sum + 1e-6)).item()
            dice_vals.append(dice)

        hd95 = float("nan")
        if g_sum > 0 and p_sum > 0:
            try:
                # Move to CPU numpy for KDTree computation
                pred_np = p[0, 0].cpu().numpy().astype(bool)
                gt_np   = g[0, 0].cpu().numpy().astype(bool)
                hd95 = _compute_hd95_numpy(pred_np, gt_np, spacing)
            except Exception:
                pass
        hd95_vals.append(hd95)

    return dice_vals, hd95_vals


def evaluate_radiologist_true_dice(model, rad_cases, roi_size, device):
    """
    Evaluates model predictions against expert radiologist GT cases.
    Returns mean radiologist true-Dice score across available radiologist GT cases.
    """
    if not rad_cases:
        return None
    try:
        model.eval()
        val_ds = Dataset(data=rad_cases, transform=build_transforms(roi_size, is_train=False))
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        dices = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                lbls = batch["label"].to(device)
                out = sliding_window_inference(
                    inputs=imgs, roi_size=roi_size,
                    sw_batch_size=1, predictor=model, overlap=0.25,
                )
                if isinstance(out, (tuple, list)):
                    out = out[0]
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True)

                pred_oh = torch.zeros(imgs.shape[0], NUM_CLASSES, *imgs.shape[2:], device=device)
                lbl_oh  = torch.zeros_like(pred_oh)
                for c in range(NUM_CLASSES):
                    pred_oh[:, c, ...] = (preds[:, 0, ...] == (c + 1)).float()
                    lbl_oh[:, c, ...]  = (lbls[:, 0, ...] == (c + 1)).float()

                dice_vals, _ = compute_dice_hd95(pred_oh, lbl_oh, NUM_CLASSES)
                valid_vals = [d for d in dice_vals if not np.isnan(d)]
                if valid_vals:
                    dices.append(float(np.mean(valid_vals)))

        return float(np.mean(dices)) if dices else None
    except Exception as err:
        print(f"  Warning: radiologist true-Dice evaluation skipped ({err})")
        return None


# ---------------------------------------------------------------------------
# Per-N training run
# ---------------------------------------------------------------------------

def run_single_sweep(
    n_cases: int,
    all_cases: list,
    output_dir: str,
    epochs: int,
    roi_size: tuple,
    device: torch.device,
    loss_name: str = "asymmetric",
    alpha: float = 2.0,
    beta: float = 1.0,
    rad_cases: list = None,
    gold_standard_cases: list = None,
    seed: int = 42,
    pretrained_weights: str = None,
):
    rng = random.Random(seed)
    if n_cases > len(all_cases):
        print(f"  [N={n_cases}] Only {len(all_cases)} cases available; using all.")
        n_cases = len(all_cases)

    # Use gold-standard cases as the static validation set when available.
    # Strictly ensure training cohort does NOT include any gold standard validation cases or their sibling phases!
    if gold_standard_cases:
        gold_patients = set()
        for g_case in gold_standard_cases:
            p_name = g_case.get("name", "")
            pid = extract_patient_id(p_name)
            gold_patients.add(pid)
            gold_patients.add(p_name)

        eligible_train_cases = []
        for c in all_cases:
            c_name = c.get("name", "")
            c_pid = extract_patient_id(c_name)
            if c_pid not in gold_patients and c_name not in gold_patients:
                eligible_train_cases.append(c)

        actual_n = min(n_cases, len(eligible_train_cases))
        train_cases = rng.sample(eligible_train_cases, actual_n)
        val_cases   = gold_standard_cases
        n_train     = len(train_cases)
        n_val       = len(val_cases)
        print(f"  Validation: static gold-standard ({n_val} cases; excluded {len(all_cases)-len(eligible_train_cases)} training overlap scans)")
    else:
        # Patient-grouped 80/20 train/val split
        pt_map = {}
        for c in all_cases:
            c_name = c.get("name", "")
            c_pid = extract_patient_id(c_name)
            pt_map.setdefault(c_pid, []).append(c)

        pts = sorted(list(pt_map.keys()))
        rng.shuffle(pts)
        n_val_pt = max(1, int(0.2 * len(pts)))
        val_pts = set(pts[:n_val_pt])
        train_pts = set(pts[n_val_pt:])
        if not train_pts and len(pts) > 1:
            train_pts = {pts[-1]}
            val_pts.remove(pts[-1])

        all_train_cases = [c for p in train_pts for c in pt_map[p]]
        val_cases       = [c for p in val_pts for c in pt_map[p]]
        actual_n = min(n_cases, len(all_train_cases))
        train_cases = rng.sample(all_train_cases, actual_n)
        n_train     = len(train_cases)
        n_val       = len(val_cases)
        print(f"  Validation: patient-grouped 20%% split ({n_val} scans across {len(val_pts)} pts)")

    if n_train == 0 or n_val == 0:
        raise ValueError(f"Scaling sweep requires nonempty train and val populations! Got train={n_train}, val={n_val}")

    # Output paths — per (N, seed) to avoid clobbering across multi-seed runs (Findings 2 & 5)
    run_dir = os.path.join(output_dir, f"N{n_train:04d}", f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    n_dir = run_dir

    # Save training membership manifest into run_dir
    train_scan_names = [c.get("name", "") for c in train_cases]
    manifest_info = {
        "n_cases": n_train,
        "n_requested": n_cases,
        "n_train_scans": n_train,
        "n_train_patients": len({extract_patient_id(c.get('name', '')) for c in train_cases}),
        "n_val_scans": n_val,
        "n_val_patients": len({extract_patient_id(c.get('name', '')) for c in val_cases}),
        "seed": seed,
        "architecture": "MedNeXt-B",
        "model_id": "B",
        "kernel_size": 3,
        "num_classes": NUM_CLASSES,
        "roi_size": list(roi_size),
        "data_fingerprint": hashlib.sha256("::".join(sorted(train_scan_names)).encode()).hexdigest(),
        "train_patients": sorted(list({extract_patient_id(c.get('name', '')) for c in train_cases})),
        "val_patients": sorted(list({extract_patient_id(c.get('name', '')) for c in val_cases})),
        "train_cases": train_scan_names,
        "val_cases": [c.get("name", "") for c in val_cases],
    }
    with open(os.path.join(run_dir, "train_manifest.json"), "w") as f:
        json.dump(manifest_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Scaling Sweep: N={n_train} (requested={n_cases}) | seed={seed} | train={n_train} | val={n_val} | loss={loss_name}")
    print(f"{'='*60}")

    train_ds = Dataset(data=train_cases, transform=build_transforms(roi_size, is_train=True))
    val_ds   = Dataset(data=val_cases,   transform=build_transforms(roi_size, is_train=False))

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=monai.data.list_data_collate,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(NUM_CLASSES, device)
    if pretrained_weights and os.path.exists(pretrained_weights):
        print(f"  Loading pretrained weights from {pretrained_weights}")
        checkpoint = torch.load(pretrained_weights, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        try:
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError:
            new_state_dict = {}
            for k, v in state_dict.items():
                k_new = k.replace("module.", "") if not isinstance(model, torch.nn.DataParallel) else f"module.{k.replace('module.', '')}"
                new_state_dict[k_new] = v
            model.load_state_dict(new_state_dict, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    if loss_name == "asymmetric" and AsymmetricPDCELoss is not None:
        loss_fn = AsymmetricPDCELoss(apply_softmax=True, alpha=alpha, beta=beta)
    else:
        if loss_name == "asymmetric":
            print("Warning: AsymmetricPDCELoss not imported; falling back to DiceCELoss.")
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    json_path = os.path.join(n_dir, "epoch_metrics.json")
    csv_path  = os.path.join(n_dir, "epoch_metrics.csv")

    epoch_records = []
    csv_header = (
        ["epoch", "train_loss"]
        + [f"val_dice_{o}" for o in ORGAN_NAMES]
        + [f"val_hd95_{o}" for o in ORGAN_NAMES]
        + ["val_mean_dice", "val_mean_hd95", "val_true_dice"]
    )

    csv_rows = []
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            optimizer.zero_grad()
            out = model(imgs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # ---- Validate ----
        model.eval()
        all_dice, all_hd95 = [[] for _ in range(NUM_CLASSES)], [[] for _ in range(NUM_CLASSES)]

        with torch.no_grad():
            for vbatch in val_loader:
                v_imgs = vbatch["image"].to(device)
                v_lbls = vbatch["label"].to(device)

                v_out = sliding_window_inference(
                    inputs=v_imgs, roi_size=roi_size,
                    sw_batch_size=1, predictor=model, overlap=0.25,
                )
                if isinstance(v_out, (tuple, list)):
                    v_out = v_out[0]

                probs = torch.softmax(v_out, dim=1)
                preds = torch.argmax(probs, dim=1, keepdim=True)

                pred_oh = torch.zeros(
                    v_imgs.shape[0], NUM_CLASSES, *v_imgs.shape[2:], device=device
                )
                lbl_oh  = torch.zeros_like(pred_oh)
                for c in range(NUM_CLASSES):
                    pred_oh[:, c, ...] = (preds[:, 0, ...] == (c + 1)).float()
                    lbl_oh[:, c, ...]  = (v_lbls[:, 0, ...] == (c + 1)).float()

                dice_vals, hd95_vals = compute_dice_hd95(pred_oh, lbl_oh, NUM_CLASSES)
                for c in range(NUM_CLASSES):
                    # Only append valid non-NaN scores to avoid poisoning whole organ mean (F11/F15)
                    if not np.isnan(dice_vals[c]):
                        all_dice[c].append(dice_vals[c])
                    if not np.isnan(hd95_vals[c]):
                        all_hd95[c].append(hd95_vals[c])

        per_organ_dice = [float(np.mean(all_dice[c])) if all_dice[c] else float("nan")
                          for c in range(NUM_CLASSES)]
        per_organ_hd95 = [float(np.mean(all_hd95[c])) if all_hd95[c] else float("nan")
                          for c in range(NUM_CLASSES)]
        mean_dice = float(np.nanmean(per_organ_dice)) if any(not np.isnan(v) for v in per_organ_dice) else float("nan")
        mean_hd95 = float(np.nanmean(per_organ_hd95)) if any(not np.isnan(v) for v in per_organ_hd95) else float("nan")

        true_dice = evaluate_radiologist_true_dice(model, rad_cases, roi_size, device)

        record = {
            "epoch": epoch,
            "n_cases": n_train,
            "n_requested": n_cases,
            "n_train_scans": n_train,
            "n_train_patients": manifest_info["n_train_patients"],
            "train_loss": round(avg_loss, 6),
            "val_dice": {ORGAN_NAMES[c]: round(per_organ_dice[c], 6) for c in range(NUM_CLASSES)},
            "val_hd95": {ORGAN_NAMES[c]: round(per_organ_hd95[c], 4) for c in range(NUM_CLASSES)},
            "val_mean_dice": round(mean_dice, 6),
            "val_mean_hd95": round(mean_hd95, 4) if not np.isnan(mean_hd95) else None,
            "val_true_dice": round(true_dice, 6) if true_dice is not None else None,
        }
        epoch_records.append(record)

        csv_row = (
            [epoch, round(avg_loss, 6)]
            + [round(per_organ_dice[c], 6) for c in range(NUM_CLASSES)]
            + [round(per_organ_hd95[c], 4) if not np.isnan(per_organ_hd95[c]) else ""
               for c in range(NUM_CLASSES)]
            + [round(mean_dice, 6), round(mean_hd95, 4) if not np.isnan(mean_hd95) else "",
               round(true_dice, 6) if true_dice is not None else ""]
        )
        csv_rows.append(csv_row)

        td_str = f" | true_dice={true_dice:.4f}" if true_dice is not None else ""
        print(
            f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
            f"mean_dice={mean_dice:.4f} | mean_hd95={mean_hd95:.2f}{td_str}"
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_ckpt_path = os.path.join(n_dir, "best_checkpoint.pt")
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "val_mean_dice": mean_dice, "n_cases": n_train,
                 "n_requested": n_cases, "n_train_scans": n_train,
                 "n_train_patients": manifest_info["n_train_patients"],
                 "architecture": "MedNeXt-B", "model_id": "B"},
                best_ckpt_path,
            )
            with open(best_ckpt_path, "rb") as bf:
                ckpt_hash = hashlib.sha256(bf.read()).hexdigest()
            manifest_info["checkpoint_sha256"] = ckpt_hash
            manifest_info["checkpoint_file"] = "best_checkpoint.pt"
            with open(os.path.join(run_dir, "train_manifest.json"), "w") as f:
                json.dump(manifest_info, f, indent=2)

        with open(json_path, "w") as f:
            json.dump({
                "n_cases": n_train,
                "n_requested": n_cases,
                "n_train_scans": n_train,
                "n_train_patients": manifest_info["n_train_patients"],
                "epochs": epoch_records,
            }, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)

    print(f"  Saved metrics → {json_path}")
    print(f"  Saved metrics → {csv_path}")
    print(f"  Best val Dice (N={n_train}, requested={n_cases}, seed={seed}): {best_dice:.4f}")

    # Clean up GPU memory before next run
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return json_path, best_dice


# ---------------------------------------------------------------------------
# Sweep orchestration
# ---------------------------------------------------------------------------

def write_sweep_summary(sweep_results: list, output_dir: str):
    """Write per-N best metrics aggregated across seeds (mean ± std)."""
    summary_json_path = os.path.join(output_dir, "sweep_summary.json")
    summary_csv_path  = os.path.join(output_dir, "sweep_summary.csv")

    # Group raw results by N
    from collections import defaultdict
    by_n: dict = defaultdict(list)
    for res in sweep_results:
        n = res.get("n_train_scans", res.get("n_cases"))
        n_req = res.get("n_requested", res.get("n_cases"))
        n_pts = res.get("n_train_patients", 1)
        seed = res.get("seed", 42)
        records = res.get("epochs", res.get("epoch_records", []))
        if not records:
            continue
        best = max(records, key=lambda r: r["val_mean_dice"])
        by_n[n].append({
            "seed": seed,
            "n_train_scans": n,
            "n_requested": n_req,
            "n_train_patients": n_pts,
            "best_epoch": best["epoch"],
            "best_val_mean_dice": best["val_mean_dice"],
            "best_val_mean_hd95": best["val_mean_hd95"],
            "best_val_true_dice": best.get("val_true_dice"),
            **{f"best_dice_{o}": best["val_dice"][o] for o in ORGAN_NAMES},
            **{f"best_hd95_{o}": best["val_hd95"][o] for o in ORGAN_NAMES},
        })

    # Aggregate across seeds: compute mean ± std per N
    summary = []
    for n in sorted(by_n):
        runs = by_n[n]
        dice_vals_across_seeds = [r["best_val_mean_dice"] for r in runs if r["best_val_mean_dice"] is not None]
        hd95_vals_across_seeds = [r["best_val_mean_hd95"] for r in runs if r["best_val_mean_hd95"] is not None]
        true_dice_across_seeds = [r["best_val_true_dice"] for r in runs if r["best_val_true_dice"] is not None]
        n_req = runs[0].get("n_requested", n) if runs else n
        n_pts = runs[0].get("n_train_patients", 1) if runs else 1
        entry = {
            "n_cases": n,
            "n_train_scans": n,
            "n_requested": n_req,
            "n_train_patients": n_pts,
            "num_seeds": len(runs),
            "seeds": [r["seed"] for r in runs],
            "mean_best_val_dice":     float(np.mean(dice_vals_across_seeds))     if dice_vals_across_seeds else None,
            "std_best_val_dice":      float(np.std(dice_vals_across_seeds))      if len(dice_vals_across_seeds) > 1 else None,
            "mean_best_val_hd95":     float(np.mean(hd95_vals_across_seeds))     if hd95_vals_across_seeds else None,
            "std_best_val_hd95":      float(np.std(hd95_vals_across_seeds))      if len(hd95_vals_across_seeds) > 1 else None,
            "mean_best_val_true_dice": float(np.mean(true_dice_across_seeds))   if true_dice_across_seeds else None,
            "std_best_val_true_dice":  float(np.std(true_dice_across_seeds))    if len(true_dice_across_seeds) > 1 else None,
            "per_seed_results": runs,
        }
        # Per-organ aggregation
        for o in ORGAN_NAMES:
            vals = [r[f"best_dice_{o}"] for r in runs if r.get(f"best_dice_{o}") is not None]
            entry[f"mean_dice_{o}"] = float(np.mean(vals)) if vals else None
            entry[f"std_dice_{o}"]  = float(np.std(vals))  if len(vals) > 1 else None
        summary.append(entry)

    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Flat CSV (one row per N)
    if summary:
        flat_keys = [
            "n_cases", "n_train_scans", "n_requested", "n_train_patients", "num_seeds",
            "mean_best_val_dice", "std_best_val_dice",
            "mean_best_val_hd95", "std_best_val_hd95",
            "mean_best_val_true_dice", "std_best_val_true_dice",
        ] + [f"mean_dice_{o}" for o in ORGAN_NAMES] + [f"std_dice_{o}" for o in ORGAN_NAMES]
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)

    print(f"\nSweep summary saved → {summary_json_path}")
    print(f"Sweep summary saved → {summary_csv_path}")


# ---------------------------------------------------------------------------
# Gold-standard NRRD loader
# ---------------------------------------------------------------------------

def load_gold_standard_nrrd_cases(
    gold_standard_dir: str,
    tmp_nifti_dir: str,
) -> list:
    """Load JHU radiologist .seg.nrrd annotations, converting each to a temp .nii.gz.

    The seg.nrrd files use Slicer 3D Segment format.  We extract the voxel array
    with pynrrd and re-save as NIfTI so the MONAI DataLoader can ingest them without
    modification.  The companion CT NIfTI must live next to the .nrrd with the same
    BDMAP_* stem (e.g. BDMAP_00242113.nii.gz for BDMAP_00242113.seg.nrrd).

    Returns a list of {image: <ct.nii.gz>, label: <mask.nii.gz>, name: <id>} dicts.
    """
    try:
        import nrrd  # pynrrd
    except ImportError:
        print("  Warning: pynrrd not installed — gold-standard NRRD loader skipped.")
        print("  Install with: pip install pynrrd")
        return []

    os.makedirs(tmp_nifti_dir, exist_ok=True)
    cases = []

    nrrd_files = sorted(glob.glob(os.path.join(gold_standard_dir, "*.seg.nrrd")))
    if not nrrd_files:
        print(f"  Warning: no .seg.nrrd files found in {gold_standard_dir}")
        return []

    for nrrd_path in nrrd_files:
        stem = os.path.basename(nrrd_path).replace(".seg.nrrd", "")  # e.g. BDMAP_00242113

        # Companion CT: must be a .nii.gz in the same directory
        ct_path = os.path.join(gold_standard_dir, f"{stem}.nii.gz")
        if not os.path.exists(ct_path):
            print(f"  Warning: CT not found for gold-standard case {stem} (expected {ct_path}) — skipping.")
            continue

        # Convert seg.nrrd → temp nii.gz with freshness and version validation (F11/F14)
        out_label_path = os.path.join(tmp_nifti_dir, f"{stem}_gs_label.nii.gz")
        needs_convert = (
            not os.path.exists(out_label_path)
            or os.path.getsize(out_label_path) == 0
            or os.path.getmtime(out_label_path) < os.path.getmtime(nrrd_path)
        )
        if needs_convert:
            try:
                import SimpleITK as sitk
                itk_img = sitk.ReadImage(nrrd_path)
                # SimpleITK automatically converts LPS directions and origin to NIfTI RAS headers (F14)
                sitk.WriteImage(itk_img, out_label_path)
            except Exception as exc:
                print(f"  Warning: failed to convert {nrrd_path} → NIfTI ({exc}) — skipping.")
                continue

        cases.append({"image": ct_path, "label": out_label_path, "name": stem, "schema": "jhu"})
        print(f"  Gold-standard case loaded: {stem}")

    return cases


def main():
    parser = argparse.ArgumentParser(description="Dataset scaling law sweep for GI segmentation.")
    parser.add_argument("--data_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument("--mask_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument(
        "--cohort_sizes", type=int, nargs="+", default=DEFAULT_COHORT_SIZES,
        help="List of sub-cohort sizes N to sweep (default: 4 10 15 20 30 50)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--roi_size", type=str, default="96,96,96")
    parser.add_argument(
        "--loss", type=str, default="asymmetric", choices=["asymmetric", "dice_ce"],
        help="Loss function to use: 'asymmetric' (AsymmetricPDCELoss) or 'dice_ce' (DiceCELoss)",
    )
    parser.add_argument("--alpha", type=float, default=2.0, help="Alpha weight for false negatives in AsymmetricPDCELoss")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta weight for false positives in AsymmetricPDCELoss")
    parser.add_argument(
        "--radiologist_gt_dir", type=str,
        default="/mnt/scratch/user/chrsong/mp-factory/JHU_data_radiologist_corrected",
        help="Directory of radiologist .nii.gz cases for true-Dice evaluation.",
    )
    parser.add_argument(
        "--gold_standard_dir", type=str, default=None,
        help="Directory of gold-standard .seg.nrrd annotations (JHU radiologist-corrected). "
             "When provided, these 7 cases are used as the static validation set instead of "
             "the random 20%% split, anchoring sweep metrics to expert-verified ground truth.",
    )
    parser.add_argument("--output_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/scaling_sweep")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="List of random seeds for multi-seed variance bounding (default: 42). "
             "Example: --seeds 42 123 456",
    )
    # Legacy single-seed arg kept for backwards-compatibility; overridden by --seeds if both given
    parser.add_argument("--seed", type=int, default=None, help="[Deprecated] Use --seeds instead.")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to TS-distillation pretraining weights")
    parser.add_argument(
        "--generate_dummy", action="store_true",
        help="Generate synthetic dummy data for smoke-testing.",
    )
    args = parser.parse_args()

    # Resolve seed list (legacy --seed takes effect if --seeds was not explicitly set)
    seeds = args.seeds
    if args.seed is not None and seeds == [42]:
        seeds = [args.seed]

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Seeds:  {seeds}")

    # ROI size
    roi_size = tuple(int(x) for x in args.roi_size.split(","))
    assert len(roi_size) == 3, "roi_size must have exactly 3 values"

    # ---- Data loading ----
    tmp_dir = None
    rad_cases = []
    gold_standard_cases = []

    if args.generate_dummy or not (
        os.path.exists(args.data_dir) and os.path.exists(args.mask_dir)
    ):
        print("Generating dummy cohort for smoke-test ...")
        tmp_dir = tempfile.mkdtemp()
        max_n = max(args.cohort_sizes)
        all_cases = create_dummy_cohort(tmp_dir, n_cases=max_n + 5)
        rad_dir = os.path.join(tmp_dir, "rad_gt")
        rad_cases = create_dummy_cohort(rad_dir, n_cases=3)
    else:
        all_cases = find_cases(args.data_dir, args.mask_dir)
        if not all_cases:
            print(f"Error: no cases found in {args.data_dir}")
            sys.exit(1)
        if os.path.exists(args.radiologist_gt_dir):
            rad_cases = find_cases(args.radiologist_gt_dir, args.radiologist_gt_dir)

    # Load gold-standard NRRD annotations (static validation anchor)
    if args.gold_standard_dir and os.path.isdir(args.gold_standard_dir):
        gs_tmp_dir = os.path.join(args.output_dir, "_gs_nifti_tmp")
        gold_standard_cases = load_gold_standard_nrrd_cases(args.gold_standard_dir, gs_tmp_dir)
        if gold_standard_cases:
            print(f"Gold-standard static val set: {len(gold_standard_cases)} cases from {args.gold_standard_dir}")
        else:
            print("  Warning: gold_standard_dir provided but no cases could be loaded. Falling back to random split.")

    print(f"Total training cases available: {len(all_cases)}")
    if rad_cases:
        print(f"Radiologist GT cases for true-Dice: {len(rad_cases)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Progress CSV (incremental, survives job timeouts) ----
    progress_csv_path = os.path.join(args.output_dir, "scaling_sweep_progress.csv")
    progress_header = ["cohort_size_N", "seed", "best_generalization_dice"]
    if not os.path.exists(progress_csv_path):
        with open(progress_csv_path, "w", newline="") as f:
            csv.writer(f).writerow(progress_header)

    # ---- Multi-seed outer sweep loop ----
    sweep_results = []
    t0 = time.time()

    for n in sorted(set(args.cohort_sizes)):
        for seed in seeds:
            print(f"\n{'='*68}")
            print(f"  SWEEP: N={n}  |  seed={seed}")
            print(f"{'='*68}")

            json_path, best_dice = run_single_sweep(
                n_cases=n,
                all_cases=all_cases,
                output_dir=args.output_dir,
                epochs=args.epochs,
                roi_size=roi_size,
                device=device,
                loss_name=args.loss,
                alpha=args.alpha,
                beta=args.beta,
                rad_cases=rad_cases,
                gold_standard_cases=gold_standard_cases,
                seed=seed,
                pretrained_weights=args.pretrained_weights,
            )

            with open(json_path) as f:
                result = json.load(f)
                result["seed"] = seed  # tag with seed for write_sweep_summary
                sweep_results.append(result)

            # Incremental progress save — protects against cluster job timeouts
            n_actual = result.get("n_train_scans", n)
            with open(progress_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([n_actual, seed, round(best_dice, 6)])

    write_sweep_summary(sweep_results, args.output_dir)

    elapsed = time.time() - t0
    print(f"\nFull scaling sweep complete in {elapsed/60:.1f} min.")
    print(f"Progress log: {progress_csv_path}")

    if tmp_dir:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
