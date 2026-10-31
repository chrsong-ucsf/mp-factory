"""
tests/test_audit_remediation_regressions.py
============================================
Comprehensive regression tests directly exercising actual production application functions:
1. JHU physical coordinate resampling & artifact filtering (evaluate_jhu_radiologist_gpu.run_gpu_evaluation).
2. TotalSegmentator schema precedence (sample_multiphase_diffusion.load_anatomy_mask).
3. Anatomy conditioning physical alignment and strict overlap gating (sample_multiphase_diffusion.load_anatomy_mask).
4. Multi-phase tumor host organ discovery across cascading directories (generate_multiphase_tumors.find_and_extract_host_parenchyma).
5. Standardized unified MONAI spatial transform (common_spatial.preprocess_single_volume).
6. Patient-grouping split and manuscript table compilation (compile_manuscript_tables.build_table_2 & resolve_jhu_patient_id).
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
import nibabel as nib
import torch

# Ensure repo root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from code.evaluation.evaluate_jhu_radiologist_gpu import run_gpu_evaluation
from code.project10.evaluation.sample_multiphase_diffusion import load_anatomy_mask
from code.project10.kinetics.generate_multiphase_tumors import find_and_extract_host_parenchyma
from code.project10.common_spatial import preprocess_single_volume
from code.manuscript.compile_manuscript_tables import (
    build_table_2,
    resolve_jhu_patient_id,
    JHU_SCAN_TO_PATIENT_MAP,
)


def test_production_jhu_gpu_evaluation_physical_resampling_and_artifacts():
    """Finding 1 & 2: Production run_gpu_evaluation ignores uncertainty artifacts and physically aligns predictions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        jhu_dir = os.path.join(tmpdir, "jhu")
        pred_dir = os.path.join(tmpdir, "predictions")
        out_csv = os.path.join(tmpdir, "jhu_eval_out.csv")
        os.makedirs(jhu_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        # 1. Create mock ground-truth NRRD for BDMAP_00242113 (32x32x32, 1.5mm, stomach label 2)
        shape = (32, 32, 32)
        gt_arr = np.zeros(shape, dtype=np.uint8)
        gt_arr[10:20, 10:20, 10:20] = 2  # GT Stomach
        gt_itk = sitk.GetImageFromArray(gt_arr)
        gt_itk.SetSpacing((1.5, 1.5, 1.5))
        gt_itk.SetOrigin((0.0, 0.0, 0.0))
        gt_path = os.path.join(jhu_dir, "BDMAP_00242113.seg.nrrd")
        sitk.WriteImage(gt_itk, gt_path)

        # 2. Add an uncertainty artifact that MUST be skipped by production evaluator
        unc_arr = np.ones(shape, dtype=np.float32) * 0.9
        unc_itk = sitk.GetImageFromArray(unc_arr)
        unc_itk.SetSpacing((1.5, 1.5, 1.5))
        sitk.WriteImage(unc_itk, os.path.join(pred_dir, "BDMAP_00242113_uncertainty.nii.gz"))

        # 3. Add valid consensus prediction mask (MedNeXt / Ensemble label 1 = Stomach)
        pred_arr = np.zeros(shape, dtype=np.uint8)
        pred_arr[10:20, 10:20, 10:20] = 1  # Prediction Stomach
        pred_itk = sitk.GetImageFromArray(pred_arr)
        pred_itk.SetSpacing((1.5, 1.5, 1.5))
        pred_path = os.path.join(pred_dir, "BDMAP_00242113_consensus.nii.gz")
        sitk.WriteImage(pred_itk, pred_path)

        # Execute production evaluation pipeline
        run_gpu_evaluation(
            jhu_dir=jhu_dir,
            pred_dirs=[pred_dir],
            model_names=["EnsembleConsensus"],
            out_csv=out_csv,
        )

        assert os.path.exists(out_csv), "Production GPU evaluator must generate summary CSV"
        df = pd.read_csv(out_csv)
        assert len(df) == 1, "Should evaluate exactly the 1 discovered valid candidate"
        # Verify artifact exclusion: candidate used must be the consensus mask, not uncertainty
        assert "uncertainty" not in df.iloc[0]["pred_path"].lower()
        assert "consensus" in df.iloc[0]["pred_path"].lower()
        # Verify physical Dice computation: exact match -> stomach_dice == 1.0
        assert df.iloc[0]["stomach_dice"] == pytest.approx(1.0, abs=1e-3)


def test_production_load_anatomy_mask_physical_alignment_and_strict_gating():
    """Finding 2 & 3: Production load_anatomy_mask enforces physical alignment to CT and rejects 1000mm shifted masks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ct_dir = os.path.join(tmpdir, "ct")
        aligned_dir = os.path.join(tmpdir, "aligned_mask")
        shifted_dir = os.path.join(tmpdir, "shifted_mask")
        os.makedirs(ct_dir, exist_ok=True)
        os.makedirs(aligned_dir, exist_ok=True)
        os.makedirs(shifted_dir, exist_ok=True)

        shape = (32, 32, 32)
        ct_arr = np.zeros(shape, dtype=np.float32)
        ct_itk = sitk.GetImageFromArray(ct_arr)
        ct_itk.SetSpacing((1.5, 1.5, 1.5))
        ct_itk.SetOrigin((0.0, 0.0, 0.0))
        ct_path = os.path.join(ct_dir, "ct.nii.gz")
        sitk.WriteImage(ct_itk, ct_path)

        # 1. Aligned mask (origin [0, 0, 0]) with duodenum voxels
        mask_arr = np.zeros(shape, dtype=np.uint8)
        mask_arr[10:20, 10:20, 10:20] = 2  # Duodenum in GI schema
        mask_aligned_itk = sitk.GetImageFromArray(mask_arr)
        mask_aligned_itk.SetSpacing((1.5, 1.5, 1.5))
        mask_aligned_itk.SetOrigin((0.0, 0.0, 0.0))
        sitk.WriteImage(mask_aligned_itk, os.path.join(aligned_dir, "SUBJ_01_gi_mask.nii.gz"))

        # 2. Shifted mask: identical array, but origin shifted 1000mm in X
        mask_shifted_itk = sitk.GetImageFromArray(mask_arr)
        mask_shifted_itk.SetSpacing((1.5, 1.5, 1.5))
        mask_shifted_itk.SetOrigin((1000.0, 0.0, 0.0))
        sitk.WriteImage(mask_shifted_itk, os.path.join(shifted_dir, "SUBJ_01_gi_mask.nii.gz"))

        # Call production load_anatomy_mask on aligned mask: must succeed and find duodenum
        cond_aligned = load_anatomy_mask(
            subject_id="SUBJ_01",
            mask_dirs=[aligned_dir],
            reference_ct_path=ct_path,
            require_conditioning=True,
            totalseg_task="gi",
        )
        assert cond_aligned.shape == (1, 5, 16, 16, 16)
        # Channel 2 is duodenum; must have non-zero support
        assert torch.sum(cond_aligned[:, 2]) > 0, "Aligned duodenum mask must produce foreground conditioning"

        # Call production load_anatomy_mask on 1000mm shifted mask with require_conditioning=True:
        # Physical resampling onto CT geometry yields 0 overlap -> must raise ValueError / FileNotFoundError
        with pytest.raises((ValueError, FileNotFoundError)):
            load_anatomy_mask(
                subject_id="SUBJ_01",
                mask_dirs=[shifted_dir],
                reference_ct_path=ct_path,
                require_conditioning=True,
                totalseg_task="gi",
            )


def test_production_totalseg_schema_precedence():
    """Finding 4: Production load_anatomy_mask respects explicit total_v1 schema and prevents label 18 collision."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (32, 32, 32)
        mask_arr = np.zeros(shape, dtype=np.uint8)
        mask_arr[5:15, 5:15, 5:15] = 56  # TotalSeg v1 duodenum
        mask_arr[0, 0, 0] = 18           # 1 voxel of 18 (lung in v1, small bowel in v2)

        mask_itk = sitk.GetImageFromArray(mask_arr)
        mask_itk.SetSpacing((1.5, 1.5, 1.5))
        mask_path = os.path.join(tmpdir, "SUBJ_02.nii.gz")
        sitk.WriteImage(mask_itk, mask_path)

        cond = load_anatomy_mask(
            subject_id="SUBJ_02",
            mask_dirs=[tmpdir],
            totalseg_task="total_v1",
            require_conditioning=True,
        )
        # In total_v1: 56 maps to duodenum (channel 2). Label 18 must NOT map to small bowel (channel 3).
        assert torch.sum(cond[:, 2]) > 0, "Label 56 must map to duodenum (channel 2)"
        assert torch.sum(cond[:, 3]) == 0, "Label 18 under total_v1 must not map to small bowel"


def test_production_tumor_host_discovery_organ_search():
    """Finding 4: find_and_extract_host_parenchyma iterates past GI mask to find valid liver host in full-body dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        gi_dir = Path(tmpdir) / "gi_masks"
        full_dir = Path(tmpdir) / "full_masks"
        ct_dir = Path(tmpdir) / "ct"
        gi_dir.mkdir()
        full_dir.mkdir()
        ct_dir.mkdir()

        shape = (32, 32, 32)
        ct_itk = sitk.GetImageFromArray(np.zeros(shape, dtype=np.float32))
        ct_itk.SetSpacing((1.5, 1.5, 1.5))
        ct_path = str(ct_dir / "ct.nii.gz")
        sitk.WriteImage(ct_itk, ct_path)

        # 1. GI directory has GI mask with labels 1..4 (NO liver label 5)
        gi_arr = np.zeros(shape, dtype=np.uint8)
        gi_arr[5:15, 5:15, 5:15] = 4  # Colon
        gi_itk = sitk.GetImageFromArray(gi_arr)
        gi_itk.SetSpacing((1.5, 1.5, 1.5))
        sitk.WriteImage(gi_itk, str(gi_dir / "SUB_100_gi_mask.nii.gz"))

        # 2. Full-body directory has full mask with liver label 5 (500 voxels)
        full_arr = np.zeros(shape, dtype=np.uint8)
        full_arr[10:20, 10:20, 10:15] = 5  # 500 voxels of liver
        full_itk = sitk.GetImageFromArray(full_arr)
        full_itk.SetSpacing((1.5, 1.5, 1.5))
        sitk.WriteImage(full_itk, str(full_dir / "SUB_100.nii.gz"))

        # Search for liver: GI directory comes first in mask_dirs, but lacks liver.
        # find_and_extract_host_parenchyma must NOT halt on the GI mask; it must discover the liver in full_dir.
        parenchyma, matched_path = find_and_extract_host_parenchyma(
            subject_id="SUB_100",
            mask_dirs=[gi_dir, full_dir],
            ct_vol=np.zeros(shape),
            reference_ct_path=ct_path,
            organ_type="liver",
            totalseg_task="total",
            min_voxels=200,
        )
        assert np.sum(parenchyma) >= 200, "Must extract >= 200 liver voxels from full-body mask"
        assert matched_path is not None
        assert "SUB_100.nii.gz" in matched_path, "Must resolve to the full-body mask containing liver"


def test_production_common_spatial_unification():
    """Finding 1: preprocess_single_volume standardizes volumes to 1.5mm / 128^3 grid via unified MONAI pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create non-isotropic 2.0mm CT volume
        shape = (40, 40, 30)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        ct_arr = np.random.uniform(-100.0, 300.0, size=shape).astype(np.float32)
        ct_path = os.path.join(tmpdir, "test_ct.nii.gz")
        nib.save(nib.Nifti1Image(ct_arr, affine), ct_path)

        vol, res_affine = preprocess_single_volume(
            ct_path,
            target_shape=(128, 128, 128),
            spacing=(1.5, 1.5, 1.5),
            is_mask=False,
        )
        assert vol.shape == (128, 128, 128)
        assert vol.dtype == np.float32
        assert vol.min() >= -1000.0
        assert vol.max() <= 2000.0
        # Spacing in affine diagonal must match 1.5mm
        np.testing.assert_allclose(np.abs(np.diag(res_affine)[:3]), [1.5, 1.5, 1.5], atol=1e-2)


def test_production_manuscript_tables_patient_counting():
    """Finding 6: resolve_jhu_patient_id groups 11 multi-phase scans into 5 unique patients across 6 NRRDs."""
    scans = list(JHU_SCAN_TO_PATIENT_MAP.keys())
    assert len(scans) == 11, "Should have 11 multi-phase JHU scans"

    patient_ids = [resolve_jhu_patient_id({"subject_id": sid}) for sid in scans]
    unique_pts = set(patient_ids)
    assert len(unique_pts) == 5, f"11 scans must resolve to exactly 5 unique patients! Got: {unique_pts}"

    # Verify build_table_2 compiles and reports 5 patients
    latex_code, md_code = build_table_2()
    assert latex_code is not None
    assert md_code is not None
    assert "5" in md_code, "Table 2 markdown must report 5 unique patients"
    assert "11 evaluated scans correspond to 5 distinct patients" in md_code
