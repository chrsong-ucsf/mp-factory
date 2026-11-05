#!/usr/bin/env python3
"""
Post-hoc evaluation script for scaling sweep checkpoints.
Loads checkpoints saved under results/scaling_sweep/N*/seed*/best_mednext_phase2.pt
and calculates validation Dice across the 4 GI organs.
"""

import os
import sys
import glob
import csv
import json
import re
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import monai
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance
from monai.data import Dataset

# Add training dir to path to import model creation and transforms
SCRATCH_DEFAULT = "/mnt/scratch/user/chrsong/mp-factory"
sys.path.insert(0, os.path.join(SCRATCH_DEFAULT, "code", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

try:
    import importlib
    _phase2 = importlib.import_module("train_mednext_phase2")
    create_mednext_v1 = getattr(_phase2, "create_mednext_v1", None)
    get_transforms = getattr(_phase2, "get_transforms", None)
    FixLiarLabelAffined = getattr(_phase2, "FixLiarLabelAffined", None)
except Exception as _e:
    print(f"[WARNING] Could not import train_mednext_phase2: {_e}")
    create_mednext_v1 = None
    get_transforms = None
    FixLiarLabelAffined = None

try:
    import importlib
    _mednext_mod = importlib.import_module("mednext")
    MedNeXtB = getattr(_mednext_mod, "MedNeXtB", None)
except Exception:
    MedNeXtB = None

try:
    from monai.networks.nets import UNet
except Exception:
    UNet = None

ORGAN_NAMES = ["stomach", "duodenum", "small_bowel", "colon"]
NUM_CLASSES = 5  # 0: bg, 1: stomach, 2: duodenum, 3: small bowel, 4: colon


JHU_SCAN_TO_PATIENT_MAP = {
    "BDMAP_00242113": "PATIENT_BDMAP_P1", "BDMAP_00242114": "PATIENT_BDMAP_P1",
    "BDMAP_00242115": "PATIENT_BDMAP_P1", "BDMAP_00242116": "PATIENT_BDMAP_P1",
    "BDMAP_00242131": "PATIENT_BDMAP_P2", "BDMAP_00242132": "PATIENT_BDMAP_P2", "BDMAP_00242133": "PATIENT_BDMAP_P2",
    "BDMAP_00242134": "PATIENT_BDMAP_P3", "BDMAP_00242135": "PATIENT_BDMAP_P4", "BDMAP_00242136": "PATIENT_BDMAP_P4",
    "BDMAP_00394224": "PATIENT_BDMAP_P5",
}

def extract_patient_id(subject_id: str) -> str:
    s = str(subject_id).strip()
    if s in JHU_SCAN_TO_PATIENT_MAP:
        return JHU_SCAN_TO_PATIENT_MAP[s]
    parts = s.split("_")
    if len(parts) >= 2 and parts[0] in ("CV", "CancerVerse", "BDMAP"):
        return f"{parts[0]}_{parts[1]}"
    return s


def find_validation_cases(data_dir, ensemble_out_dir, audit_csv, gold_standard_dir=None, max_cases=20, exclude_subjects=None):
    cases = []
    raw_exclude = set(str(s) for s in (exclude_subjects or []))
    exclude_set = set()
    for s in raw_exclude:
        exclude_set.add(s)
        exclude_set.add(extract_patient_id(s))

    # If gold_standard_dir is provided, look for gold standard masks
    if gold_standard_dir and os.path.exists(gold_standard_dir):
        gs_files = sorted(glob.glob(os.path.join(gold_standard_dir, "*.nii.gz")))
        for gf in gs_files:
            sub = os.path.basename(gf).replace(".nii.gz", "")
            if sub in exclude_set or extract_patient_id(sub) in exclude_set:
                continue
            ct = os.path.join(data_dir, sub, "ct.nii.gz")
            if os.path.exists(ct):
                cases.append({"image": ct, "label": gf, "name": sub})
                if len(cases) >= max_cases:
                    break
        if cases:
            print(f"Found {len(cases)} cases from gold standard directory: {gold_standard_dir}")
            return cases

    if audit_csv and os.path.exists(audit_csv):
        df = pd.read_csv(audit_csv)
        clean = df[df["triage_category"] == "CLEAN_HIGH_CONFIDENCE"]
        for _, row in clean.iterrows():
            sub = str(row["subject_id"])
            if sub in exclude_set or extract_patient_id(sub) in exclude_set:
                continue
            ct = os.path.join(data_dir, sub, "ct.nii.gz")
            lbl = os.path.join(ensemble_out_dir, f"{sub}_consensus.nii.gz")
            if os.path.exists(ct) and os.path.exists(lbl):
                cases.append({"image": ct, "label": lbl, "name": sub})
                if len(cases) >= max_cases:
                    break

    if not cases and os.path.exists(ensemble_out_dir):
        for lbl in sorted(glob.glob(os.path.join(ensemble_out_dir, "*_consensus.nii.gz"))):
            sub = os.path.basename(lbl).replace("_consensus.nii.gz", "")
            if sub in exclude_set or extract_patient_id(sub) in exclude_set:
                continue
            ct = os.path.join(data_dir, sub, "ct.nii.gz")
            if os.path.exists(ct):
                cases.append({"image": ct, "label": lbl, "name": sub})
                if len(cases) >= max_cases:
                    break

    return cases


def evaluate_checkpoint(model, ckpt_path, cases, val_tf, roi_size, device):
    # train_mednext_phase2 saves raw.state_dict() directly (a flat OrderedDict)
    # It is NOT wrapped in a dict with 'model_state_dict' key.
    # We try common wrappers first, then fall back to treating the whole thing as a state dict.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "state_dict" in ckpt):
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    else:
        # Saved with torch.save(raw.state_dict(), path) — already the state dict
        sd = ckpt
    clean_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        raise RuntimeError(f"Strict loading failed — missing checkpoint keys: {missing[:5]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys in checkpoint: {unexpected[:3]}")
    model = model.to(device).eval()

    val_ds = Dataset(data=cases, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # 4 organ classes (indices 1..4)
    all_dice = [[] for _ in range(4)]

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
            preds = torch.argmax(out, dim=1, keepdim=True)

            # Mask out ignore index (255) from both predictions and targets (F15)
            valid_mask = (lbls[:, 0, ...] != 255)

            for c in range(4):
                organ_idx = c + 1
                p = ((preds[:, 0, ...] == organ_idx) & valid_mask).float()
                g = ((lbls[:, 0, ...] == organ_idx) & valid_mask).float()
                p_sum = p.sum().item()
                g_sum = g.sum().item()
                if p_sum > 0 or g_sum > 0:
                    inter = (p * g).sum().item()
                    dice = (2.0 * inter) / (p_sum + g_sum + 1e-6)
                    all_dice[c].append(dice)

    per_organ_dice = {
        ORGAN_NAMES[c]: float(np.mean(all_dice[c])) if len(all_dice[c]) > 0 else np.nan
        for c in range(4)
    }
    valid_organ_dices = [d for d in per_organ_dice.values() if not np.isnan(d)]
    mean_dice = float(np.mean(valid_organ_dices)) if valid_organ_dices else np.nan

    return {
        "mean_dice": mean_dice,
        "per_organ_dice": per_organ_dice,
    }


class ManifestDict(dict):
    """Manifest dictionary that supports both direct dict access and 2-element tuple unpacking."""
    def __init__(self, manifest_path: str, data: dict):
        super().__init__(data)
        self.manifest_path = manifest_path
        self["_manifest_path"] = manifest_path

    def __iter__(self):
        yield self.manifest_path
        yield self


def validate_checkpoint_manifest(
    ckpt: str,
    ckpt_dir: str = None,
    expected_fold: int = None,
    manifest_path: str = None,
):
    """Validate that checkpoint has an authentic training manifest with required membership schema (Finding 3)."""
    if ckpt_dir is None:
        ckpt_dir = os.path.dirname(ckpt) or "."
    ckpt_name = os.path.basename(ckpt)
    fold_match = re.search(r'fold_(-?\d+)', ckpt_name) or re.search(r'fold_(-?\d+)', ckpt_dir)

    if manifest_path is not None:
        found_manifest = str(manifest_path)
        if not os.path.exists(found_manifest):
            raise RuntimeError(f"Specified manifest path does not exist: {manifest_path}")
    else:
        manifest_candidates = []
        if expected_fold is not None:
            manifest_candidates.append(os.path.join(ckpt_dir, f"fold_{expected_fold}_manifest.json"))
        if fold_match:
            f_idx = fold_match.group(1)
            manifest_candidates.extend([
                os.path.join(ckpt_dir, f"fold_{f_idx}_manifest.json"),
                os.path.join(ckpt_dir, "train_manifest.json"),
                os.path.join(ckpt_dir, "fold_-1_manifest.json"),
            ])
        else:
            manifest_candidates.extend([
                os.path.join(ckpt_dir, "train_manifest.json"),
                os.path.join(ckpt_dir, "fold_-1_manifest.json"),
            ])
            for p in sorted(glob.glob(os.path.join(ckpt_dir, "fold_*_manifest.json"))):
                if p not in manifest_candidates:
                    manifest_candidates.append(p)

        found_manifest = None
        for mc in manifest_candidates:
            if os.path.exists(mc):
                found_manifest = mc
                break

        if found_manifest is None:
            raise RuntimeError(
                f"Missing required training manifest in {ckpt_dir} for checkpoint {ckpt}. "
                f"Expected one of {[os.path.basename(m) for m in manifest_candidates]}. "
                f"Bound membership/configuration is required to evaluate holdout validation."
            )

    try:
        with open(found_manifest, "r") as mf:
            minfo = json.load(mf)
        if not isinstance(minfo, dict) or not minfo:
            raise ValueError("Manifest is empty or not a valid JSON dictionary")

        # 1. Validate training membership schema and reject blank/malformed identifiers
        raw_train_members = []
        for key in ["train_cases", "train_scans", "train_patients"]:
            val = minfo.get(key)
            if val is not None:
                if not isinstance(val, list):
                    raise ValueError(f"Membership field '{key}' in {found_manifest} must be a list, got {type(val).__name__}")
                raw_train_members.extend(val)

        if not raw_train_members:
            raise ValueError(
                f"Manifest {found_manifest} has empty or missing training membership "
                f"('train_cases', 'train_scans', or 'train_patients')"
            )

        clean_members = []
        for item in raw_train_members:
            if not isinstance(item, str):
                raise ValueError(f"Manifest {found_manifest} contains non-string member identifier: {item!r}")
            s = item.strip()
            if not s:
                raise ValueError(f"Manifest {found_manifest} contains blank or malformed identifiers: {item!r}")
            clean_members.append(s)

        if not clean_members:
            raise ValueError(f"Manifest {found_manifest} contains zero valid training member identifiers after stripping")

        # 2. Validate architecture / configuration identity metadata (non-null and non-blank)
        valid_configs = {}
        for k in ["architecture", "model_id", "model_type", "model_config", "config"]:
            val = minfo.get(k)
            if val is not None:
                if isinstance(val, str) and val.strip():
                    valid_configs[k] = val.strip()
                elif isinstance(val, dict) and val:
                    valid_configs[k] = val
        if not valid_configs:
            raise ValueError(
                f"Manifest {found_manifest} lacks required architecture/configuration identity metadata "
                f"('architecture', 'model_id', 'model_type', or 'config'). Values must not be null or blank."
            )

        # 3. Authenticate checkpoint binding (content hash / fingerprint)
        expected_hash = minfo.get("checkpoint_sha256") or minfo.get("checkpoint_hash")
        if not expected_hash:
            raise ValueError(
                f"Manifest {found_manifest} lacks required checkpoint binding hash ('checkpoint_sha256' or 'checkpoint_hash'). "
                f"Membership cannot be authenticated without cryptographic checkpoint binding."
            )

        # Compute SHA-256 of the actual checkpoint
        with open(ckpt, "rb") as cf:
            actual_ckpt_hash = hashlib.sha256(cf.read()).hexdigest()

        if actual_ckpt_hash != str(expected_hash).strip():
            raise ValueError(
                f"Checkpoint authentication failed for {ckpt}: actual SHA-256 {actual_ckpt_hash} "
                f"does not match manifest recorded hash {expected_hash}. "
                f"Checkpoint contents have been altered or replaced."
            )

        # 4. Validate fold binding if applicable
        target_fold = expected_fold if expected_fold is not None else (int(fold_match.group(1)) if fold_match else None)
        if "fold" in minfo and target_fold is not None:
            if int(minfo["fold"]) != int(target_fold):
                raise ValueError(
                    f"Manifest fold mismatch: manifest specifies fold {minfo['fold']} but expected fold is {target_fold}"
                )
        return ManifestDict(found_manifest, minfo)
    except Exception as me:
        if isinstance(me, (ValueError, RuntimeError)):
            raise
        raise RuntimeError(f"Could not read, parse, or validate training manifest {found_manifest}: {me}")
        raise RuntimeError(f"Could not read, parse, or validate training manifest {found_manifest}: {me}")


def build_eval_model_for_manifest(minfo: dict, device):
    """Construct model matching training manifest architecture, factory, and configuration."""
    raw_arch = str(minfo.get("architecture") or "").strip()
    raw_mid = str(minfo.get("model_id") or "").strip()

    # Extract variant from architecture if present
    arch_variant = None
    for sfx in ["-S", "_S", "-M", "_M", "-L", "_L", "-B", "_B"]:
        if raw_arch.upper().endswith(sfx):
            arch_variant = sfx[1]
            break
    if not arch_variant:
        if raw_arch.upper() in ["MEDNEXT-S", "MEDNEXT_S", "MEDNEXTS"]:
            arch_variant = "S"
        elif raw_arch.upper() in ["MEDNEXT-B", "MEDNEXT_B", "MEDNEXTB"]:
            arch_variant = "B"
        elif raw_arch.upper() in ["MEDNEXT-M", "MEDNEXT_M", "MEDNEXTM"]:
            arch_variant = "M"
        elif raw_arch.upper() in ["MEDNEXT-L", "MEDNEXT_L", "MEDNEXTL"]:
            arch_variant = "L"

    # Extract variant from model_id if present
    mid_variant = None
    if raw_mid.upper() in ["S", "M", "L", "B"]:
        mid_variant = raw_mid.upper()
    elif raw_mid.upper().startswith("MEDNEXT"):
        for sfx in ["-S", "_S", "-M", "_M", "-L", "_L", "-B", "_B"]:
            if raw_mid.upper().endswith(sfx):
                mid_variant = sfx[1]
                break

    # Cross-check 1: MedNeXt variant disagreement (Finding 4)
    if arch_variant is not None and mid_variant is not None and arch_variant != mid_variant:
        raise RuntimeError(
            f"Contradictory manifest identity: architecture '{raw_arch}' contradicts model_id '{raw_mid}'."
        )

    # Cross-check 2: UNet vs MedNeXt disagreement (Finding 4)
    arch_is_unet = raw_arch.upper() in ["UNET", "MONAI-UNET", "MONAI_UNET"]
    mid_is_unet = raw_mid.upper() in ["UNET", "MONAI-UNET", "MONAI_UNET"]
    arch_is_mednext = raw_arch.upper().startswith("MEDNEXT") or arch_variant is not None
    mid_is_mednext = raw_mid.upper().startswith("MEDNEXT") or mid_variant is not None

    if (arch_is_unet and mid_is_mednext) or (arch_is_mednext and mid_is_unet):
        raise RuntimeError(
            f"Contradictory manifest identity: architecture '{raw_arch}' contradicts model_id '{raw_mid}'."
        )

    arch = raw_arch or raw_mid
    effective_variant = mid_variant or arch_variant
    factory = minfo.get("factory")
    if factory is not None:
        factory = str(factory).strip()
    k_size = minfo.get("kernel_size", 7)
    out_ch = minfo.get("out_channels") or minfo.get("num_classes", NUM_CLASSES)
    in_ch = minfo.get("in_channels", 1)

    # 1. Explicit factory dispatch
    if factory == "local_mednext":
        target_variant = effective_variant or str(minfo.get("model_id") or "").strip().upper()
        if not target_variant and arch in ["MedNeXt-B", "MedNeXtB", "MedNeXt_B", "MedNeXt"]:
            target_variant = "B"
        if target_variant and target_variant != "B":
            raise RuntimeError(
                f"Manifest specified factory 'local_mednext', but declared variant '{target_variant}' contradicts local MedNeXtB (variant B)."
            )
        if k_size is not None and k_size != 7:
            raise RuntimeError(
                f"Manifest specified factory 'local_mednext', but declared kernel_size '{k_size}' contradicts local MedNeXtB (kernel 7)."
            )
        if MedNeXtB is not None:
            return MedNeXtB(in_channels=in_ch, n_channels=32, n_classes=out_ch).to(device)
        raise RuntimeError("Manifest specified factory 'local_mednext', but local MedNeXtB is not available.")

    if factory == "nnunet_mednext":
        target_variant = effective_variant or str(minfo.get("model_id", "B")).strip().upper()
        if create_mednext_v1 is not None:
            return create_mednext_v1(
                num_input_channels=in_ch,
                num_classes=out_ch,
                model_id=target_variant if target_variant else "B",
                kernel_size=k_size if isinstance(k_size, int) else 3,
                deep_supervision=False,
            ).to(device)
        raise RuntimeError("Manifest specified factory 'nnunet_mednext', but create_mednext_v1 is not available.")

    if factory in ["monai_unet", "monai"]:
        if UNet is not None:
            return UNet(
                spatial_dims=3, in_channels=in_ch, out_channels=out_ch,
                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
            ).to(device)
        raise RuntimeError("Manifest specified factory 'monai_unet', but MONAI UNet is not available.")

    if factory is not None:
        raise RuntimeError(f"Unsupported model factory '{factory}' in manifest.")

    # 2. Legacy manifests without explicit factory: dispatch based on architecture / kernel_size
    if arch in ["UNet", "MONAI-UNet"]:
        if UNet is not None:
            return UNet(
                spatial_dims=3, in_channels=in_ch, out_channels=out_ch,
                channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
            ).to(device)
        raise RuntimeError("Manifest specified UNet architecture, but MONAI UNet is not available.")

    if arch in ["MedNeXt-B", "MedNeXtB", "MedNeXt_B"] or arch.startswith("MedNeXt-") or arch == "MedNeXt":
        m_id = effective_variant
        if not m_id:
            m_id = "B" if arch in ["MedNeXt-B", "MedNeXtB", "MedNeXt_B", "MedNeXt"] else ""

        # Only variant B with kernel 7 can be served by local MedNeXtB
        if m_id == "B" and k_size == 7:
            if MedNeXtB is not None:
                return MedNeXtB(in_channels=in_ch, n_channels=32, n_classes=out_ch).to(device)
            raise RuntimeError("Legacy MedNeXt-B kernel-7 architecture requires local MedNeXtB, which is unavailable.")

        # Non-B variants or non-7 kernels require create_mednext_v1
        if create_mednext_v1 is not None:
            return create_mednext_v1(
                num_input_channels=in_ch,
                num_classes=out_ch,
                model_id=m_id if m_id else "B",
                kernel_size=k_size if isinstance(k_size, int) else 3,
                deep_supervision=False,
            ).to(device)
        raise RuntimeError(
            f"Legacy MedNeXt-{m_id or 'B'} kernel-{k_size} requires nnunet_mednext (create_mednext_v1), which is not installed."
        )

    raise RuntimeError(f"Unknown or unsupported architecture '{arch}' in manifest; cannot dispatch evaluation model.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate scaling sweep checkpoints post-hoc.")
    parser.add_argument("--results_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/scaling_sweep")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Explicit checkpoint file(s) to evaluate instead of scanning results_dir")
    parser.add_argument("--data_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox")
    parser.add_argument("--ensemble_out_dir", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out")
    parser.add_argument("--audit_csv", type=str, default="/mnt/scratch/user/chrsong/mp-factory/results/ensemble_audit_summary.csv")
    parser.add_argument("--gold_standard_dir", type=str, default=None, help="Path to JHU radiologist-corrected labels directory")
    parser.add_argument("--output_csv", type=str, default=None, help="Custom output CSV path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_val_cases", type=int, default=20)
    parser.add_argument("--roi_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--exclude_manifest", type=str, default=None,
                        help="Optional CSV manifest of training cases to strictly exclude from validation (F06/F13)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 70)
    print("      SCALING SWEEP POST-HOC EVALUATION")
    print(f"      Device:     {device}")
    print(f"      Results:    {args.results_dir}")
    if args.gold_standard_dir:
        print(f"      Gold Std:   {args.gold_standard_dir}")
    print("=" * 70)

    # 1. Discover all checkpoints
    if args.checkpoints:
        ckpts = args.checkpoints
    else:
        # Pattern 1: results/scaling_sweep/N*/seed*/best_mednext_phase2.pt
        ckpts = sorted(glob.glob(os.path.join(args.results_dir, "N*", "seed*", "best_mednext_phase2.pt")))
        if not ckpts:
            # Fallback to any best_*.pt
            ckpts = sorted(glob.glob(os.path.join(args.results_dir, "**", "best_*.pt"), recursive=True))

    if not ckpts:
        print(f"No checkpoints found under {args.results_dir}")
        sys.exit(1)

    print(f"Discovered {len(ckpts)} checkpoints to evaluate.\n")

    # Validate training manifests for all checkpoints upfront (Finding 3)
    for ckpt in ckpts:
        validate_checkpoint_manifest(ckpt)

    exclude_subs = set()
    if args.exclude_manifest:
        if not os.path.exists(args.exclude_manifest):
            raise FileNotFoundError(f"Specified --exclude_manifest does not exist: {args.exclude_manifest}")
        ex_df = pd.read_csv(args.exclude_manifest)
        col = "subject_id" if "subject_id" in ex_df.columns else ("name" if "name" in ex_df.columns else None)
        if col is None:
            raise ValueError(f"--exclude_manifest '{args.exclude_manifest}' is malformed: missing 'subject_id' or 'name' column.")
        exclude_subs = set(ex_df[col].astype(str))
        print(f"Excluding {len(exclude_subs)} training subjects and their sibling phases from validation holdout.")

    # 2. Find validation cases
    val_cases = find_validation_cases(
        args.data_dir, args.ensemble_out_dir, args.audit_csv,
        gold_standard_dir=args.gold_standard_dir, max_cases=args.num_val_cases,
        exclude_subjects=exclude_subs,
    )
    if not val_cases:
        print(f"ERROR: No valid (CT, label) pairs found in {args.data_dir} / {args.ensemble_out_dir}!")
        sys.exit(1)
    print(f"Loaded {len(val_cases)} validation cases for evaluation.")

    # 3. Get transforms with FixLiarLabelAffined
    if get_transforms is None:
        print("ERROR: get_transforms could not be imported from train_mednext_phase2.")
        sys.exit(1)
    _, val_tf = get_transforms(roi_size=tuple(args.roi_size))

    # 4. Checkpoints evaluation loop
    summary = []
    current_model = None
    current_model_key = None
    for ckpt in ckpts:
        rel = os.path.relpath(ckpt, args.results_dir)
        parts = rel.split(os.sep)
        n_folder = parts[0] if len(parts) > 0 else "unknown"
        seed_folder = parts[1] if len(parts) > 1 else "unknown"

        # Ensure per-checkpoint training membership exclusion (Finding 5)
        ckpt_dir = os.path.dirname(ckpt)
        ckpt_excludes = set()
        for s in exclude_subs:
            s_str = str(s)
            ckpt_excludes.add(s_str)
            ckpt_excludes.add(extract_patient_id(s_str))

        # Look for matching checkpoint manifest (train_manifest.json or fold_<n>_manifest.json) (Finding 3)
        found_manifest, minfo = validate_checkpoint_manifest(ckpt, ckpt_dir)
        train_members = (
            minfo.get("train_cases", [])
            + minfo.get("train_scans", [])
            + minfo.get("train_patients", [])
        )
        for c_name in train_members:
            c_str = str(c_name).strip()
            if c_str:
                ckpt_excludes.add(c_str)
                ckpt_excludes.add(extract_patient_id(c_str))

        curr_val_cases = [
            c for c in val_cases
            if c.get("name", "") not in ckpt_excludes
            and extract_patient_id(c.get("name", "")) not in ckpt_excludes
        ]
        if not curr_val_cases:
            curr_val_cases = find_validation_cases(
                args.data_dir, args.ensemble_out_dir, args.audit_csv,
                gold_standard_dir=args.gold_standard_dir, max_cases=args.num_val_cases,
                exclude_subjects=ckpt_excludes,
            )

        if not curr_val_cases:
            raise RuntimeError(
                f"No independent validation cases remain for checkpoint {ckpt} after strict patient/sibling exclusion!"
            )

        # Build or reuse model matching this checkpoint's manifest (Finding 4: Key reuse on complete validated configuration)
        model_key = (minfo.get("architecture"), minfo.get("factory"), minfo.get("model_id"), minfo.get("kernel_size"), minfo.get("in_channels", 1), minfo.get("out_channels") or minfo.get("num_classes"))
        if current_model is None or current_model_key != model_key:
            current_model = build_eval_model_for_manifest(minfo, device)
            current_model_key = model_key

        print(f"Evaluating [{n_folder} | {seed_folder}] on {len(curr_val_cases)} cases -> {ckpt} ...")
        res = evaluate_checkpoint(current_model, ckpt, curr_val_cases, val_tf, tuple(args.roi_size), device)
        print(f"  -> Mean Dice: {res['mean_dice']:.4f}")
        for organ, d in res["per_organ_dice"].items():
            print(f"     {organ:12s}: {d:.4f}")

        row = {
            "cohort": n_folder,
            "seed": seed_folder,
            "checkpoint": ckpt,
            "mean_dice": res["mean_dice"],
            **{f"dice_{k}": v for k, v in res["per_organ_dice"].items()},
        }
        summary.append(row)

    # 5. Output CSV
    if args.output_csv:
        out_csv = args.output_csv
    else:
        out_csv = os.path.join(args.results_dir, "scaling_sweep_posthoc_eval.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\n" + "=" * 70)
    print(f"  Evaluation Complete! Saved summary to: {out_csv}")
    print("=" * 70)

    # Grouped summary by cohort
    df_sum = pd.DataFrame(summary)
    grouped = df_sum.groupby("cohort")["mean_dice"].agg(["mean", "std", "count"])
    print("\nScaling Curve Aggregated (Mean Dice across seeds):")
    print(grouped.to_string())


if __name__ == "__main__":
    main()
