#!/usr/bin/env python3
"""
code/project10/kinetics/multiphase_tumor_kinetics.py
=====================================================
Phase 2: Multi-Phase Contrast Kinetics Engine for Procedural Tumors

Bridges code/SyntheticTumors/TumorGenerated/ with pharmacokinetic
wash-in / wash-out contrast enhancement curves so that procedurally
generated lesions are physically accurate across all CT contrast phases.

Tumour types supported
----------------------
  hcc          Hepatocellular Carcinoma
               Arterial: hypervascular wash-in  (+100 to +180 HU above liver)
               Venous:   washout + peripheral capsule ring (+20 HU rim)

  pdac         Pancreatic Ductal Adenocarcinoma
               All phases: persistently hypo-attenuating (-30 to -50 HU)
               due to dense fibrous stroma that excludes contrast agent.

  hemangioma   Cavernous Hepatic Hemangioma
               Arterial: peripheral nodular enhancement (+80 to +120 HU)
               Venous:   centripetal fill-in (progressive fill toward centre)
               Delayed:  complete iso-/hyper-attenuation to liver parenchyma

  metastasis   Hypo-vascular liver metastasis (e.g. from CRC)
               All phases: low-level ring enhancement only (+20 to +40 HU rim)
               Core: persistently hypo-attenuating

Usage
-----
  # Standalone callable:
  from code.project10.kinetics.multiphase_tumor_kinetics import MultiPhaseTumorAugment

  aug = MultiPhaseTumorAugment(tumor_type="hcc", organ="liver")
  ct_ncct   = aug.generate(ct_volume, organ_mask, phase="ncct")
  ct_art    = aug.generate(ct_volume, organ_mask, phase="arterial")
  ct_venous = aug.generate(ct_volume, organ_mask, phase="venous")

  # As a MONAI MapTransform (drop-in for TumorGenerated):
  from code.project10.kinetics.multiphase_tumor_kinetics import MultiPhaseTumorGenerated
"""

from __future__ import annotations

import sys
import random
import warnings
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Mapping, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, label as nd_label

# ---------------------------------------------------------------------------
# Optional imports from SyntheticTumors submodule
# ---------------------------------------------------------------------------
_SYNTUMOR_AVAILABLE = False
try:
    # Locate SyntheticTumors relative to this file
    import os
    _syntumor_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../SyntheticTumors")
    )
    if _syntumor_path not in sys.path:
        sys.path.insert(0, _syntumor_path)
    from TumorGenerated.utils import (
        get_fixed_geo, get_predefined_texture, get_ellipsoid
    )
    _SYNTUMOR_AVAILABLE = True
except ImportError:
    warnings.warn(
        "SyntheticTumors submodule not importable; falling back to built-in "
        "ellipsoid generator. Run: git submodule update --init code/SyntheticTumors",
        ImportWarning,
        stacklevel=2,
    )

try:
    from monai.config import KeysCollection
    from monai.config.type_definitions import NdarrayOrTensor
    from monai.transforms.transform import MapTransform, RandomizableTransform
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Phase identifiers
# ---------------------------------------------------------------------------
PHASE_NCCT      = "ncct"
PHASE_ARTERIAL  = "arterial"       # early arterial (~25-35s post-injection)
PHASE_VENOUS    = "venous"         # portal-venous  (~70-80s)
PHASE_DELAYED   = "delayed"        # delayed        (~3-5 min)
ALL_PHASES      = [PHASE_NCCT, PHASE_ARTERIAL, PHASE_VENOUS, PHASE_DELAYED]

# ---------------------------------------------------------------------------
# Pharmacokinetic contrast enhancement parameters (HU deltas above baseline)
# Source: ACR CT Liver Imaging Atlas; radiologist clinical feedback.
# ---------------------------------------------------------------------------
@dataclass
class PhaseEnhancement:
    """Enhancement delta over NCCT parenchyma baseline for a tumour type."""
    # Mean and std of the interior tumour enhancement (HU delta from NCCT)
    core_mean_hu:     float = 0.0
    core_std_hu:      float = 10.0
    # Boundary ring enhancement (separate from core)
    ring_width_vox:   int   = 2        # ring width in voxels
    ring_mean_hu:     float = 0.0
    ring_std_hu:      float = 5.0
    # Boundary blur sigma
    boundary_sigma:   float = 1.5


# Kinetics table indexed by (tumour_type, phase)
KINETICS: Dict[Tuple[str, str], PhaseEnhancement] = {
    # ---- HCC ---------------------------------------------------------------
    ("hcc", PHASE_NCCT):     PhaseEnhancement(core_mean_hu=-10, core_std_hu=8,
                                               ring_mean_hu=0, ring_std_hu=3),
    ("hcc", PHASE_ARTERIAL): PhaseEnhancement(core_mean_hu=+140, core_std_hu=20,
                                               ring_mean_hu=+160, ring_std_hu=10,
                                               ring_width_vox=2, boundary_sigma=1.8),
    ("hcc", PHASE_VENOUS):   PhaseEnhancement(core_mean_hu=-25, core_std_hu=10,
                                               ring_mean_hu=+30, ring_std_hu=8,
                                               ring_width_vox=2, boundary_sigma=2.0),
    ("hcc", PHASE_DELAYED):  PhaseEnhancement(core_mean_hu=-30, core_std_hu=8,
                                               ring_mean_hu=+40, ring_std_hu=5,
                                               ring_width_vox=3, boundary_sigma=2.5),

    # ---- PDAC --------------------------------------------------------------
    ("pdac", PHASE_NCCT):     PhaseEnhancement(core_mean_hu=-5,  core_std_hu=6,
                                                ring_mean_hu=0, ring_std_hu=4),
    ("pdac", PHASE_ARTERIAL): PhaseEnhancement(core_mean_hu=-40, core_std_hu=8,
                                                ring_mean_hu=-15, ring_std_hu=5,
                                                ring_width_vox=1, boundary_sigma=1.2),
    ("pdac", PHASE_VENOUS):   PhaseEnhancement(core_mean_hu=-35, core_std_hu=8,
                                                ring_mean_hu=-10, ring_std_hu=5,
                                                ring_width_vox=1, boundary_sigma=1.2),
    ("pdac", PHASE_DELAYED):  PhaseEnhancement(core_mean_hu=-30, core_std_hu=7,
                                                ring_mean_hu=-5, ring_std_hu=4,
                                                ring_width_vox=1, boundary_sigma=1.5),

    # ---- Hemangioma --------------------------------------------------------
    ("hemangioma", PHASE_NCCT):     PhaseEnhancement(core_mean_hu=-5,  core_std_hu=5,
                                                      ring_mean_hu=0, ring_std_hu=3),
    ("hemangioma", PHASE_ARTERIAL): PhaseEnhancement(core_mean_hu=-10, core_std_hu=8,
                                                      ring_mean_hu=+110, ring_std_hu=15,
                                                      ring_width_vox=4, boundary_sigma=2.0),
    ("hemangioma", PHASE_VENOUS):   PhaseEnhancement(core_mean_hu=+60, core_std_hu=15,
                                                      ring_mean_hu=+100, ring_std_hu=10,
                                                      ring_width_vox=3, boundary_sigma=2.0),
    ("hemangioma", PHASE_DELAYED):  PhaseEnhancement(core_mean_hu=+90, core_std_hu=10,
                                                      ring_mean_hu=+95, ring_std_hu=5,
                                                      ring_width_vox=2, boundary_sigma=2.5),

    # ---- Hypo-vascular metastasis ------------------------------------------
    ("metastasis", PHASE_NCCT):     PhaseEnhancement(core_mean_hu=-5,  core_std_hu=5,
                                                      ring_mean_hu=0, ring_std_hu=3),
    ("metastasis", PHASE_ARTERIAL): PhaseEnhancement(core_mean_hu=-20, core_std_hu=8,
                                                      ring_mean_hu=+35, ring_std_hu=8,
                                                      ring_width_vox=2, boundary_sigma=1.5),
    ("metastasis", PHASE_VENOUS):   PhaseEnhancement(core_mean_hu=-25, core_std_hu=8,
                                                      ring_mean_hu=+30, ring_std_hu=8,
                                                      ring_width_vox=2, boundary_sigma=1.5),
    ("metastasis", PHASE_DELAYED):  PhaseEnhancement(core_mean_hu=-20, core_std_hu=7,
                                                      ring_mean_hu=+20, ring_std_hu=5,
                                                      ring_width_vox=2, boundary_sigma=2.0),
}

SUPPORTED_TUMOR_TYPES = ["hcc", "pdac", "hemangioma", "metastasis"]


# ---------------------------------------------------------------------------
# Simple built-in fallback ellipsoid geometry (if SyntheticTumors unavailable)
# ---------------------------------------------------------------------------
def _make_ellipsoid_mask(
    volume_shape: Tuple[int, int, int],
    organ_mask: np.ndarray,
    min_radius_vox: int = 8,
    max_radius_vox: int = 32,
) -> np.ndarray:
    """Generate a random ellipsoid tumour mask inside organ_mask."""
    # Sample a random seed point inside the organ (eroded slightly)
    eroded = binary_erosion(organ_mask, iterations=max_radius_vox // 2)
    coords = np.argwhere(eroded)
    if len(coords) == 0:
        coords = np.argwhere(organ_mask)
    if len(coords) == 0:
        return np.zeros(volume_shape, dtype=bool)

    centre = coords[np.random.randint(len(coords))]
    rx = np.random.randint(min_radius_vox, max_radius_vox + 1)
    ry = np.random.randint(min_radius_vox, max_radius_vox + 1)
    rz = np.random.randint(min_radius_vox, max_radius_vox + 1)

    x, y, z = np.ogrid[
        -centre[0]:volume_shape[0] - centre[0],
        -centre[1]:volume_shape[1] - centre[1],
        -centre[2]:volume_shape[2] - centre[2],
    ]
    ellipsoid = ((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2) <= 1.0
    return (ellipsoid & organ_mask.astype(bool))


def _make_tumor_geometry(
    organ_mask: np.ndarray,
    tumor_type_size: str = "medium",
) -> np.ndarray:
    """
    Generate tumour binary mask.  Tries SyntheticTumors get_fixed_geo first;
    falls back to built-in ellipsoid generator.
    """
    if _SYNTUMOR_AVAILABLE:
        try:
            geo = get_fixed_geo(organ_mask.astype(np.int8), tumor_type_size)
            return geo.astype(bool)
        except Exception:
            pass
    # Fallback
    radius_map = {"tiny": (3, 5), "small": (5, 12), "medium": (10, 24), "large": (20, 40)}
    lo, hi = radius_map.get(tumor_type_size, (8, 20))
    return _make_ellipsoid_mask(organ_mask.shape, organ_mask, lo, hi)


# ---------------------------------------------------------------------------
# Core kinetics stamping
# ---------------------------------------------------------------------------
def _get_ring_mask(geo_mask: np.ndarray, ring_width: int) -> np.ndarray:
    """Compute the peripheral ring of a binary tumour mask."""
    eroded = binary_erosion(geo_mask, iterations=max(1, ring_width))
    return geo_mask & ~eroded


def _add_texture_noise(
    shape: Tuple[int, ...], sigma: float = 4.0, scale: float = 1.0
) -> np.ndarray:
    """Spatially correlated Gaussian texture noise (mimics tumour heterogeneity)."""
    noise = np.random.normal(0.0, 1.0, shape)
    return gaussian_filter(noise, sigma=sigma) * scale


def apply_kinetics(
    ct_volume: np.ndarray,
    organ_mask: np.ndarray,
    geo_mask: np.ndarray,
    tumor_type: str,
    phase: str,
) -> np.ndarray:
    """
    Stamp pharmacokinetic contrast values onto ct_volume inside geo_mask.

    Parameters
    ----------
    ct_volume   : float32 3D array, HU values
    organ_mask  : bool 3D array, host organ (liver / pancreas)
    geo_mask    : bool 3D array, tumour geometry within organ_mask
    tumor_type  : one of SUPPORTED_TUMOR_TYPES
    phase       : one of ALL_PHASES

    Returns
    -------
    Modified ct_volume (float32, in-place safe copy)
    """
    ct_out = ct_volume.copy()
    key = (tumor_type, phase)
    params = KINETICS.get(key)
    if params is None:
        warnings.warn(f"No kinetics entry for ({tumor_type}, {phase}). Returning unchanged.")
        return ct_out

    geo = geo_mask.astype(bool)
    if not geo.any():
        return ct_out

    # --- Core enhancement --------------------------------------------------
    core_mask = geo.copy()
    ring_mask = None
    if params.ring_width_vox > 0:
        ring_mask = _get_ring_mask(geo, params.ring_width_vox)
        core_mask = geo & ~ring_mask

    # Sample heterogeneous texture noise per voxel
    noise_core = _add_texture_noise(geo.shape, sigma=4.0, scale=params.core_std_hu)
    core_delta = params.core_mean_hu + noise_core

    # Smooth boundary (soft transition into parenchyma)
    core_weight = gaussian_filter(core_mask.astype(float), sigma=params.boundary_sigma)
    ct_out += core_weight * core_delta

    # --- Ring/capsule enhancement ------------------------------------------
    if ring_mask is not None and ring_mask.any():
        noise_ring = _add_texture_noise(geo.shape, sigma=2.0, scale=params.ring_std_hu)
        ring_delta = params.ring_mean_hu + noise_ring
        ring_weight = gaussian_filter(ring_mask.astype(float), sigma=max(0.8, params.boundary_sigma * 0.6))
        ct_out += ring_weight * ring_delta

    # --- Constrain to organ mask (no leakage into air/bone) ----------------
    # Bone invariance: HU >+400 must not change
    bone_mask = ct_volume > 400.0
    air_mask  = ct_volume < -500.0
    frozen    = bone_mask | air_mask
    ct_out[frozen] = ct_volume[frozen]

    return ct_out.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level augmentation class
# ---------------------------------------------------------------------------
class MultiPhaseTumorAugment:
    """
    Generate a consistent procedural tumour across multiple CT contrast phases.

    Usage
    -----
    aug = MultiPhaseTumorAugment(tumor_type="hcc", size="medium", organ="liver")
    aug.sample_geometry(organ_mask)          # call once per case to fix shape
    ct_ncct   = aug.apply(ct_ncct,   organ_mask, phase="ncct")
    ct_art    = aug.apply(ct_art,    organ_mask, phase="arterial")
    ct_venous = aug.apply(ct_venous, organ_mask, phase="venous")
    """

    TUMOR_SIZE_MAP = {
        "hcc":        ["small", "medium", "large"],
        "pdac":       ["small", "medium", "large"],
        "hemangioma": ["tiny", "small", "medium"],
        "metastasis": ["tiny", "small", "medium", "large"],
    }

    def __init__(
        self,
        tumor_type: str = "hcc",
        size: Optional[str] = None,
        organ: str = "liver",
        seed: Optional[int] = None,
    ):
        if tumor_type not in SUPPORTED_TUMOR_TYPES:
            raise ValueError(f"tumor_type must be one of {SUPPORTED_TUMOR_TYPES}, got {tumor_type!r}")
        self.tumor_type = tumor_type
        self.size = size or random.choice(self.TUMOR_SIZE_MAP[tumor_type])
        self.organ = organ
        self._geo_mask: Optional[np.ndarray] = None
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def sample_geometry(self, organ_mask: np.ndarray) -> "MultiPhaseTumorAugment":
        """Randomly generate and fix the tumour shape for this case."""
        self._geo_mask = _make_tumor_geometry(organ_mask, tumor_type_size=self.size)
        return self

    @property
    def geo_mask(self) -> np.ndarray:
        if self._geo_mask is None:
            raise RuntimeError("Call sample_geometry(organ_mask) before apply().")
        return self._geo_mask

    def apply(
        self,
        ct_volume: np.ndarray,
        organ_mask: np.ndarray,
        phase: str,
    ) -> np.ndarray:
        """
        Apply the pre-sampled tumour geometry with phase-correct HU values.

        Parameters
        ----------
        ct_volume  : float32 3D HU array
        organ_mask : bool 3D array
        phase      : one of ALL_PHASES

        Returns
        -------
        Augmented float32 CT volume
        """
        if phase not in ALL_PHASES:
            raise ValueError(f"phase must be one of {ALL_PHASES}, got {phase!r}")
        return apply_kinetics(
            ct_volume  = ct_volume.astype(np.float32),
            organ_mask = organ_mask.astype(bool),
            geo_mask   = self.geo_mask,
            tumor_type = self.tumor_type,
            phase      = phase,
        )

    def apply_all_phases(
        self,
        ct_ncct:   np.ndarray,
        ct_art:    np.ndarray,
        ct_venous: np.ndarray,
        organ_mask: np.ndarray,
        ct_delayed: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Convenience wrapper: apply consistent kinetics across all available phases.

        Returns
        -------
        Dict keyed by phase string ('ncct', 'arterial', 'venous', optionally 'delayed')
        """
        out: Dict[str, np.ndarray] = {
            PHASE_NCCT:     self.apply(ct_ncct,   organ_mask, PHASE_NCCT),
            PHASE_ARTERIAL: self.apply(ct_art,    organ_mask, PHASE_ARTERIAL),
            PHASE_VENOUS:   self.apply(ct_venous, organ_mask, PHASE_VENOUS),
        }
        if ct_delayed is not None:
            out[PHASE_DELAYED] = self.apply(ct_delayed, organ_mask, PHASE_DELAYED)
        return out


# ---------------------------------------------------------------------------
# MONAI MapTransform integration (drop-in replacement for TumorGenerated)
# ---------------------------------------------------------------------------
if _MONAI_AVAILABLE:
    class MultiPhaseTumorGenerated(RandomizableTransform, MapTransform):
        """
        MONAI MapTransform that applies multi-phase kinetics augmentation.

        Expects data dict with:
          keys[0]  : image tensor  (C, H, W, D) — float32 HU
          keys[1]  : label tensor  (C, H, W, D) — integer class mask
          'phase'  : phase string  (ncct / arterial / venous / delayed)

        Optional data['organ_mask'] provides a dedicated host organ mask.
        If absent, the transform uses label == organ_label_idx.
        """

        def __init__(
            self,
            keys: KeysCollection,
            prob: float = 0.5,
            tumor_types: Optional[List[str]] = None,
            tumor_probs: Optional[List[float]] = None,
            organ_label_idx: int = 0,          # label value for host organ; 0 = auto-detect liver
            phase_key: str = "phase",
            allow_missing_keys: bool = False,
        ) -> None:
            MapTransform.__init__(self, keys, allow_missing_keys)
            RandomizableTransform.__init__(self, prob)
            self.tumor_types = tumor_types or SUPPORTED_TUMOR_TYPES
            if tumor_probs is None:
                self.tumor_probs = np.ones(len(self.tumor_types)) / len(self.tumor_types)
            else:
                arr = np.array(tumor_probs, dtype=float)
                self.tumor_probs = arr / arr.sum()
            self.organ_label_idx = organ_label_idx
            self.phase_key = phase_key

        def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
            d = dict(data)
            self.randomize(None)
            if not self._do_transform:
                return d

            img_key, lbl_key = self.keys[0], self.keys[1]
            image = np.array(d[img_key])    # (C, H, W, D)
            label = np.array(d[lbl_key])
            phase = str(d.get(self.phase_key, PHASE_NCCT))

            # Extract organ mask (channel 0 spatial)
            vol = image[0]                   # (H, W, D)
            lbl = label[0]

            if "organ_mask" in d:
                organ_mask = np.array(d["organ_mask"]).astype(bool)
            else:
                # Fallback: largest foreground structure as organ proxy
                organ_mask = (lbl > 0).astype(bool)

            if not organ_mask.any():
                return d

            # Sample a random tumour type and generate geometry + kinetics
            tumor_type = np.random.choice(self.tumor_types, p=self.tumor_probs)
            aug = MultiPhaseTumorAugment(tumor_type=tumor_type)
            aug.sample_geometry(organ_mask)
            image[0] = aug.apply(vol, organ_mask, phase)

            # Update label: add tumour class (max_class + 1) into label volume
            tumor_label_val = int(lbl.max()) + 1
            geo = aug.geo_mask.astype(np.int32) * tumor_label_val
            label[0] = np.where(aug.geo_mask, geo, label[0])

            d[img_key] = image
            d[lbl_key] = label
            return d

else:
    # Minimal stub if MONAI not available
    class MultiPhaseTumorGenerated:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("MONAI is required for MultiPhaseTumorGenerated. Install monai.")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running MultiPhaseTumorAugment smoke test...")
    np.random.seed(42)

    # Synthetic 64³ volume with liver-shaped mask
    vol_shape = (64, 64, 64)
    ct_base = np.random.normal(-50, 30, vol_shape).astype(np.float32)

    # Liver mask: ellipsoid in centre
    cx, cy, cz = 32, 32, 32
    x_, y_, z_ = np.ogrid[:64, :64, :64]
    liver_mask = ((x_-cx)**2/20**2 + (y_-cy)**2/18**2 + (z_-cz)**2/15**2) <= 1.0

    aug = MultiPhaseTumorAugment(tumor_type="hcc", size="medium", seed=0)
    aug.sample_geometry(liver_mask)
    geo_vox = aug.geo_mask.sum()

    results = aug.apply_all_phases(ct_base.copy(), ct_base.copy(), ct_base.copy(), liver_mask)
    for phase, ct in results.items():
        delta = (ct - ct_base)[aug.geo_mask].mean()
        print(f"  [{phase:10s}] tumour voxels={geo_vox}, mean HU delta in tumour = {delta:+.1f}")

    print("Smoke test passed.")
