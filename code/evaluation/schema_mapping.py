"""Canonical schema definitions, label normalization, and physical coordinate alignment
for multi-model 3D GI organ segmentation.

Supports:
- TotalSegmentator v2 (official default task 'total'):
    6:  stomach       -> 1
    19: duodenum      -> 2
    18: small_bowel   -> 3
    20: colon         -> 4
- TotalSegmentator v1 (task 'total_v1'):
    6:  stomach       -> 1
    56: duodenum      -> 2
    55: small_bowel   -> 3
    57: colon         -> 4
- BDMAP / AbdomenAtlas:
    2:  stomach       -> 1
    3:  duodenum      -> 2
    4:  small_bowel   -> 3
    5:  ileum         -> 3
    6:  colon         -> 4
- JHU Expert Ground Truth (.nrrd):
    2:  stomach       -> 1
    3:  duodenum      -> 2
    4:  small_bowel   -> 3
    5:  colon         -> 4
    6:  rectum        -> 4 (merged into colon)
- Standard 5-class target:
    0:  background
    1:  stomach
    2:  duodenum
    3:  small_bowel
    4:  colon
    255: IGNORE_INDEX (for boundary/unlabeled voxels)
"""

from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
import nibabel as nib

# Canonical GI Label Constants
GI_BACKGROUND: int = 0
GI_STOMACH: int = 1
GI_DUODENUM: int = 2
GI_SMALL_BOWEL: int = 3
GI_COLON: int = 4
GI_IGNORE_INDEX: int = 255

GI_ORGAN_NAMES: Dict[int, str] = {
    GI_BACKGROUND: "background",
    GI_STOMACH: "stomach",
    GI_DUODENUM: "duodenum",
    GI_SMALL_BOWEL: "small_bowel",
    GI_COLON: "colon",
}

# Explicit Mappings
TOTALSEG_V2_MAP: Dict[int, int] = {
    6: GI_STOMACH,       # stomach
    19: GI_DUODENUM,     # duodenum
    18: GI_SMALL_BOWEL,  # small_bowel
    20: GI_COLON,        # colon
}

TOTALSEG_V1_MAP: Dict[int, int] = {
    6: GI_STOMACH,       # stomach
    56: GI_DUODENUM,     # duodenum
    55: GI_SMALL_BOWEL,  # small_bowel
    57: GI_COLON,        # colon
}

BDMAP_MAP: Dict[int, int] = {
    2: GI_STOMACH,       # stomach
    3: GI_DUODENUM,      # duodenum
    4: GI_SMALL_BOWEL,   # small bowel
    5: GI_SMALL_BOWEL,   # ileum
    6: GI_COLON,         # colon
}

JHU_MAP: Dict[int, int] = {
    2: GI_STOMACH,       # stomach
    3: GI_DUODENUM,      # duodenum
    4: GI_SMALL_BOWEL,   # intestine / small bowel
    5: GI_COLON,         # colon
    6: GI_COLON,         # rectum merged to colon
}


def detect_schema(mask: np.ndarray, model_name: Optional[str] = None) -> str:
    """Infer the organ label schema from unique array values and model name hint."""
    uniques = set(np.unique(mask).tolist())
    uniques.discard(GI_BACKGROUND)
    uniques.discard(GI_IGNORE_INDEX)

    if not uniques:
        return "gi_5class"

    m_lower = (model_name or "").lower()

    if "totalseg" in m_lower or "totalsegmentator" in m_lower:
        if "v1" in m_lower or "total_v1" in m_lower:
            return "totalseg_v1"
        if any(lbl in uniques for lbl in (55, 56, 57)) and not any(lbl in uniques for lbl in (18, 19, 20)):
            return "totalseg_v1"
        return "totalseg_v2"

    if "bdmap" in m_lower or "abdomenatlas" in m_lower:
        return "bdmap"

    if any(k in m_lower for k in ["jhu", "radiologist", "expert"]):
        return "jhu"

    if any(k in m_lower for k in ["mednext", "swin", "ensemble", "consensus", "gi"]):
        return "gi_5class"

    # If already strictly 1..4 without a specific full-body model name
    if uniques.issubset({1, 2, 3, 4}):
        return "gi_5class"

    if any(lbl > 4 for lbl in uniques):
        if model_name:
            raise ValueError(f"Unknown full body schema for model name: {model_name}")
        # Value-based detection when no model name provided
        if any(lbl in uniques for lbl in (18, 19, 20)):
            return "totalseg_v2"
        if any(lbl in uniques for lbl in (55, 56, 57)):
            return "totalseg_v1"
        if 1 in uniques and uniques.issubset({1, 2, 3, 4, 5, 6}):
            return "jhu"
        if uniques.issubset({2, 3, 4, 5, 6}):
            return "bdmap"

    raise ValueError(
        f"Unable to safely detect label schema for unique labels: {sorted(list(uniques))} "
        f"with model_name='{model_name}'. Please specify schema explicitly."
    )


def normalize_to_gi_classes(
    mask: np.ndarray,
    schema: Optional[str] = None,
    model_name: Optional[str] = None,
    preserve_ignore_index: bool = True,
) -> np.ndarray:
    """Normalize input segmentation mask into standard 5-class GI labels (0..4).

    Parameters
    ----------
    mask : np.ndarray
        Input 3D integer label array.
    schema : Optional[str]
        Explicit schema name: 'totalseg_v2', 'totalseg_v1', 'bdmap', 'jhu', 'gi_5class', or 'auto'.
    model_name : Optional[str]
        Model name hint used when schema is 'auto' or None.
    preserve_ignore_index : bool
        Whether to preserve voxels equal to GI_IGNORE_INDEX (255).

    Returns
    -------
    out : np.ndarray
        Array with values in {0, 1, 2, 3, 4} (and 255 if preserved).
    """
    if schema is None or schema == "auto":
        schema = detect_schema(mask, model_name=model_name)

    schema = schema.lower()
    if schema in ("gi_5class", "canonical", "identity"):
        out = mask.copy()
        if not preserve_ignore_index:
            out[out == GI_IGNORE_INDEX] = GI_BACKGROUND
        return out

    if schema == "totalseg_v2":
        mapping = TOTALSEG_V2_MAP
    elif schema == "totalseg_v1":
        mapping = TOTALSEG_V1_MAP
    elif schema == "bdmap":
        mapping = BDMAP_MAP
    elif schema == "jhu":
        mapping = JHU_MAP
    else:
        raise ValueError(f"Unknown schema: {schema}")

    out = np.zeros_like(mask, dtype=np.uint8)
    if preserve_ignore_index and (mask == GI_IGNORE_INDEX).any():
        out[mask == GI_IGNORE_INDEX] = GI_IGNORE_INDEX

    for src_id, dst_id in mapping.items():
        out[mask == src_id] = dst_id

    return out


def compute_world_bounding_box(shape: Tuple[int, ...], affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the min and max world coordinate corners for a 3D volume."""
    corners_vox = np.array([
        [0, 0, 0, 1],
        [shape[0], 0, 0, 1],
        [0, shape[1], 0, 1],
        [0, 0, shape[2], 1],
        [shape[0], shape[1], 0, 1],
        [shape[0], 0, shape[2], 1],
        [0, shape[1], shape[2], 1],
        [shape[0], shape[1], shape[2], 1],
    ]).T
    corners_world = affine @ corners_vox
    world_min = np.min(corners_world[:3, :], axis=1)
    world_max = np.max(corners_world[:3, :], axis=1)
    return world_min, world_max


def check_physical_overlap(
    shape_a: Tuple[int, ...],
    affine_a: np.ndarray,
    shape_b: Tuple[int, ...],
    affine_b: np.ndarray,
) -> bool:
    """Return True if the physical world bounding boxes of volumes A and B intersect."""
    min_a, max_a = compute_world_bounding_box(shape_a, affine_a)
    min_b, max_b = compute_world_bounding_box(shape_b, affine_b)

    # Disjoint if any axis does not overlap
    for d in range(3):
        if min_a[d] >= max_b[d] or min_b[d] >= max_a[d]:
            return False
    return True


def resample_label_to_reference(
    moving_arr: np.ndarray,
    moving_affine: np.ndarray,
    ref_affine: np.ndarray,
    ref_shape: Tuple[int, int, int],
    order: int = 0,
    cval: float = 0.0,
) -> np.ndarray:
    """Resample moving_arr into ref_shape and ref_affine in world physical coordinates
    using nearest-neighbor (or specified order) interpolation and zero-padding.

    If grids match physically (same shape and close affine), returns a copy without resampling.
    If physical bounds do not intersect, returns an all-zero array of ref_shape.
    """
    if moving_arr.shape == ref_shape and np.allclose(moving_affine, ref_affine, atol=1e-3):
        return moving_arr.copy()

    if not check_physical_overlap(moving_arr.shape, moving_affine, ref_shape, ref_affine):
        # Physically disjoint -> no physical overlap
        return np.zeros(ref_shape, dtype=moving_arr.dtype)

    import nibabel.processing as nip
    moving_nii = nib.Nifti1Image(moving_arr.astype(np.float32), moving_affine)
    ref_dummy = nib.Nifti1Image(np.zeros(ref_shape, dtype=np.float32), ref_affine)
    resampled_nii = nip.resample_from_to(moving_nii, ref_dummy, order=order, cval=cval)
    res_arr = np.asanyarray(resampled_nii.dataobj).squeeze()
    return res_arr.astype(moving_arr.dtype)
