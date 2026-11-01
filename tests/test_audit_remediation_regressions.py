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
    build_table_1,
    build_table_2,
    resolve_jhu_patient_id,
    JHU_SCAN_TO_PATIENT_MAP,
)
from code.utilities.compare_models import load_ground_truth
from code.project10.evaluation.evaluate_synthesis import (
    resample_real_to_target,
    compute_metrics_3d,
    merge_worker_csvs,
    METRIC_VERSION,
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


def test_production_strict_inference_candidate_fallback():
    """Finding 2: load_anatomy_mask continues past an unsuitable first candidate to discover a valid later mask."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir1 = os.path.join(tmpdir, "dir1")
        dir2 = os.path.join(tmpdir, "dir2")
        ct_dir = os.path.join(tmpdir, "ct")
        os.makedirs(dir1, exist_ok=True)
        os.makedirs(dir2, exist_ok=True)
        os.makedirs(ct_dir, exist_ok=True)

        shape = (32, 32, 32)
        ct_itk = sitk.GetImageFromArray(np.zeros(shape, dtype=np.float32))
        ct_itk.SetSpacing((1.5, 1.5, 1.5))
        ct_path = os.path.join(ct_dir, "ct.nii.gz")
        sitk.WriteImage(ct_itk, ct_path)

        # Candidate 1 in dir1: has label 2 (duodenum in GI schema), but schema requested is 'total' (where duodenum is 19).
        # In 'total' schema, label 2 maps to 0 foreground voxels.
        c1_arr = np.zeros(shape, dtype=np.uint8)
        c1_arr[5:15, 5:15, 5:15] = 2
        c1_itk = sitk.GetImageFromArray(c1_arr)
        c1_itk.SetSpacing((1.5, 1.5, 1.5))
        sitk.WriteImage(c1_itk, os.path.join(dir1, "SUBJ_FALLBACK_gi_mask.nii.gz"))

        # Candidate 2 in dir2: has label 19 (duodenum in TotalSegmentator 'total' schema).
        c2_arr = np.zeros(shape, dtype=np.uint8)
        c2_arr[5:15, 5:15, 5:15] = 19
        c2_itk = sitk.GetImageFromArray(c2_arr)
        c2_itk.SetSpacing((1.5, 1.5, 1.5))
        sitk.WriteImage(c2_itk, os.path.join(dir2, "SUBJ_FALLBACK.nii.gz"))

        # Under strict conditioning with totalseg_task='total', Candidate 1 in dir1 will have 0 foreground voxels.
        # load_anatomy_mask must NOT abort with ValueError on Candidate 1; it must continue to dir2 and succeed!
        cond = load_anatomy_mask(
            subject_id="SUBJ_FALLBACK",
            mask_dirs=[dir1, dir2],
            reference_ct_path=ct_path,
            require_conditioning=True,
            totalseg_task="total",
        )
        assert cond.shape == (1, 5, 16, 16, 16)
        assert torch.sum(cond[:, 2]) > 0, "Candidate 2 in dir2 must be discovered and map duodenum (19) to channel 2"


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

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "mock_jhu_comparison.csv")
        records = []
        for sid in scans:
            records.append({
                "model_name": "MedNeXt",
                "subject_id": sid,
                "source_nrrd": f"{sid}.seg.nrrd",
                "stomach_dice": 0.85,
                "duodenum_dice": 0.75,
                "small_bowel_dice": 0.80,
                "colon_dice": 0.82,
                "mean_dice": 0.805,
                "mean_hd95": 15.0,
            })
        pd.DataFrame(records).to_csv(csv_path, index=False)

        latex_code, md_code = build_table_2(csv_file=csv_path)
        assert latex_code is not None
        assert md_code is not None
        assert "| **MedNeXt** | 11 | 5 |" in md_code, "Table 2 markdown must report 11 scans across 5 unique patients"
        assert "11 evaluated scans correspond to 5 distinct patients" in md_code


def test_production_compare_models_displaced_ground_truth_resampling():
    """Finding 1: load_ground_truth checks full geometry and resamples displaced GT masks into CT space."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sub_dir = os.path.join(tmpdir, "SUBJ_DISPLACED")
        seg_dir = os.path.join(sub_dir, "segmentations")
        os.makedirs(seg_dir, exist_ok=True)

        shape = (8, 8, 8)
        ct_affine = np.diag([1.5, 1.5, 1.5, 1.0])
        ct_arr = np.zeros(shape, dtype=np.float32)
        nib.save(nib.Nifti1Image(ct_arr, ct_affine), os.path.join(sub_dir, "ct.nii.gz"))

        # Create stomach mask translated by +1000mm in physical space (same array shape)
        gt_affine = np.copy(ct_affine)
        gt_affine[0, 3] = 1000.0
        gt_arr = np.zeros(shape, dtype=np.uint8)
        gt_arr[2:6, 2:6, 2:6] = 1
        nib.save(nib.Nifti1Image(gt_arr, gt_affine), os.path.join(seg_dir, "stomach.nii.gz"))

        # Under physical coordinate check, stomach must be resampled onto ct_nii grid and have 0 overlap
        merged_gt, ct_nii = load_ground_truth(sub_dir)
        assert np.count_nonzero(merged_gt == 1) == 0, "Displaced stomach (+1000mm) must have 0 overlap in CT reference grid"


def test_production_evaluate_synthesis_disjoint_physical_overlap_rejection():
    """Finding 2: resample_real_to_target rejects real CT translated +1000mm with insufficient overlap ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (8, 8, 8)
        fake_affine = np.diag([1.5, 1.5, 1.5, 1.0])
        fake_path = os.path.join(tmpdir, "fake.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), fake_affine), fake_path)

        # Real CT translated +1000mm in X
        real_affine = np.copy(fake_affine)
        real_affine[0, 3] = 1000.0
        real_path = os.path.join(tmpdir, "real.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), real_affine), real_path)

        with pytest.raises(ValueError, match="Insufficient physical overlap"):
            resample_real_to_target(real_path, target_ref=fake_path)


def test_production_evaluate_synthesis_summary_aggregation_and_merge_smoke():
    """Finding 3: merge_worker_csvs and summary aggregation exclude text/provenance columns without TypeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "merged_report.csv"
        worker0_csv = Path(tmpdir) / "merged_report_worker_0.csv"
        worker1_csv = Path(tmpdir) / "merged_report_worker_1.csv"

        df0 = pd.DataFrame([{
            "Subject": "CASE_01.nii.gz",
            "SSIM": 0.85,
            "PSNR": 28.5,
            "L1_Error": 45.0,
            "Body_L1_Error": 55.0,
            "RMSE": 60.0,
            "Pearson_R": 0.88,
            "Reference_Coverage": 1.0,
            "run_id": "test_run",
            "fake_sha256": "fake_sha_01",
            "real_sha256": "real_sha_01",
            "real_file": "CASE_01_reg.nii.gz",
            "fake_mtime": 1730000000.0,
            "metric_version": METRIC_VERSION,
        }])
        df1 = pd.DataFrame([{
            "Subject": "CASE_02.nii.gz",
            "SSIM": 0.90,
            "PSNR": 30.0,
            "L1_Error": 40.0,
            "Body_L1_Error": 50.0,
            "RMSE": 50.0,
            "Pearson_R": 0.92,
            "Reference_Coverage": 1.0,
            "run_id": "test_run",
            "fake_sha256": "fake_sha_02",
            "real_sha256": "real_sha_02",
            "real_file": "CASE_02_reg.nii.gz",
            "fake_mtime": 1730000001.0,
            "metric_version": METRIC_VERSION,
        }])
        df0.to_csv(worker0_csv, index=False)
        df1.to_csv(worker1_csv, index=False)

        # Must merge and compute numeric summary without raising TypeError
        merge_worker_csvs(out_csv, total_workers=2)
        assert out_csv.exists()
        merged = pd.read_csv(out_csv)
        assert len(merged) == 2


def test_production_manuscript_table1_literature_qualification():
    """Finding 7: build_table_1 carries literature reference qualification into LaTeX output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "model_comparison.csv")
        records = [{
            "subject_id": "BDMAP_001",
            "swin_stomach_dice": 0.0,
            "mednext_stomach_dice": 0.0,
            "swin_duodenum_dice": 0.0,
            "mednext_duodenum_dice": 0.0,
            "swin_small_bowel_dice": 0.01,
            "mednext_small_bowel_dice": 0.01,
            "swin_colon_dice": 0.0,
            "mednext_colon_dice": 0.0,
            "swin_mean_dice": 0.002,
            "mednext_mean_dice": 0.002,
            "winner": "tie",
        }]
        pd.DataFrame(records).to_csv(csv_path, index=False)
        latex_code, md_code = build_table_1(csv_file=csv_path)
        assert "TotalSegmentator v2 (Ref)" in latex_code, "LaTeX table must carry reference qualification"
        assert "External literature baseline reference" in latex_code, "LaTeX table must have literature reference footnote"


def test_production_sample_multiphase_diffusion_empty_mask_dirs_rejected_when_required():
    """Finding 5: load_anatomy_mask raises FileNotFoundError when conditioning is required and dirs are empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_dir = os.path.join(tmpdir, "empty_masks")
        os.makedirs(empty_dir, exist_ok=True)
        with pytest.raises(FileNotFoundError, match="Required organ conditioning mask"):
            load_anatomy_mask(
                subject_id="CASE_NO_MASK",
                mask_dirs=[empty_dir],
                require_conditioning=True,
            )


def test_production_evaluate_synthesis_mixed_batch_failure_raises():
    """Finding 9: evaluate_synthesis raises RuntimeError when any assigned scan fails (e.g. missing target)."""
    from code.project10.evaluation.evaluate_synthesis import main as eval_main
    import unittest.mock as mock

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_dir = os.path.join(tmpdir, "fake")
        real_dir = os.path.join(tmpdir, "real")
        out_csv = os.path.join(tmpdir, "eval_out.csv")
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        shape = (8, 8, 8)
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        # Case 1: valid fake and real
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), os.path.join(fake_dir, "SUBJ_01.nii.gz"))
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), os.path.join(real_dir, "SUBJ_01.nii.gz"))

        # Case 2: fake exists but real target is missing in real_dir
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), os.path.join(fake_dir, "SUBJ_02.nii.gz"))

        test_args = [
            "evaluate_synthesis.py",
            "--fake_dir", fake_dir,
            "--real_dir", real_dir,
            "--output_csv", out_csv,
            "--device", "cpu",
        ]
        with mock.patch("sys.argv", test_args):
            with pytest.raises(RuntimeError, match="assigned cases failed evaluation"):
                eval_main()


def test_production_compare_models_cli_fallback_layouts():
    """Finding 1: compare_models CLI supports root gi_mask.nii.gz and root organ masks without segmentations/ directory."""
    from code.utilities.compare_models import main as compare_main
    import unittest.mock as mock

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "CancerVerse")
        swin_dir = os.path.join(tmpdir, "swin_preds")
        mednext_dir = os.path.join(tmpdir, "mednext_preds")
        out_csv = os.path.join(tmpdir, "model_comparison.csv")
        os.makedirs(swin_dir, exist_ok=True)
        os.makedirs(mednext_dir, exist_ok=True)

        shape = (16, 16, 16)
        affine = np.diag([1.5, 1.5, 1.5, 1.0])

        # Case 1: SUBJ_01 has root gi_mask.nii.gz, no segmentations/ dir
        subj1_dir = os.path.join(data_dir, "SUBJ_01")
        os.makedirs(subj1_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), os.path.join(subj1_dir, "ct.nii.gz"))
        comb_mask = np.zeros(shape, dtype=np.uint8)
        comb_mask[4:12, 4:12, 4:12] = 1  # Stomach
        nib.save(nib.Nifti1Image(comb_mask, affine), os.path.join(subj1_dir, "gi_mask.nii.gz"))
        nib.save(nib.Nifti1Image(comb_mask, affine), os.path.join(swin_dir, "SUBJ_01_gi_seg.nii.gz"))
        nib.save(nib.Nifti1Image(comb_mask, affine), os.path.join(mednext_dir, "SUBJ_01_gi_seg.nii.gz"))

        # Case 2: SUBJ_02 has root stomach.nii.gz, no segmentations/ dir
        subj2_dir = os.path.join(data_dir, "SUBJ_02")
        os.makedirs(subj2_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), os.path.join(subj2_dir, "ct.nii.gz"))
        stomach_mask = np.zeros(shape, dtype=np.uint8)
        stomach_mask[2:10, 2:10, 2:10] = 1
        nib.save(nib.Nifti1Image(stomach_mask, affine), os.path.join(subj2_dir, "stomach.nii.gz"))
        nib.save(nib.Nifti1Image(stomach_mask, affine), os.path.join(swin_dir, "SUBJ_02_gi_seg.nii.gz"))
        nib.save(nib.Nifti1Image(stomach_mask, affine), os.path.join(mednext_dir, "SUBJ_02_gi_seg.nii.gz"))

        test_args = [
            "compare_models.py",
            "--data_dir", data_dir,
            "--swin_pred_dir", swin_dir,
            "--mednext_pred_dir", mednext_dir,
            "--out_csv", out_csv,
        ]
        with mock.patch("sys.argv", test_args):
            compare_main()

        assert os.path.exists(out_csv), "Comparison CSV must be written"
        df = pd.read_csv(out_csv)
        assert len(df) == 2, "Both fallback layout subjects must be evaluated without crashing"
        assert set(df["subject_id"]) == {"SUBJ_01", "SUBJ_02"}
        assert (df["swin_stomach_dice"] == 1.0).all()
        assert (df["mednext_stomach_dice"] == 1.0).all()


def test_production_clinical_turing_test_physical_grid_alignment_and_disjoint_rejection():
    """Finding 2: run_clinical_turing_test uses exact reference physical grid and rejects disjoint real CT."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shape = (16, 16, 16)
        synth_affine = np.diag([1.5, 1.5, 1.5, 1.0])
        synth_path = os.path.join(tmpdir, "CV_001_synthetic_contrast.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), synth_affine), synth_path)

        # 1. Physically disjoint real CT (shifted 1000mm in X)
        disjoint_affine = synth_affine.copy()
        disjoint_affine[0, 3] = 1000.0
        disjoint_path = os.path.join(tmpdir, "CV_001_disjoint_reg.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), disjoint_affine), disjoint_path)

        with pytest.raises(ValueError, match="Insufficient physical overlap"):
            resample_real_to_target(disjoint_path, target_ref=synth_path)

        # 2. Aligned real CT resamples onto exact synthetic coordinate grid
        aligned_path = os.path.join(tmpdir, "CV_001_aligned_reg.nii.gz")
        nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32) * 50.0, synth_affine), aligned_path)
        resampled = resample_real_to_target(aligned_path, target_ref=synth_path)
        assert resampled.shape == shape
        assert np.isclose(resampled.mean(), 50.0)


def test_production_evaluate_synthesis_merge_worker_manifest_and_version_enforcement():
    """Finding 4: merge_worker_csvs enforces complete manifest, checks metric_version, and rejects stale extras."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "merged_report.csv"
        worker0 = Path(tmpdir) / "merged_report_worker_0.csv"
        worker1 = Path(tmpdir) / "merged_report_worker_1.csv"
        worker_stale = Path(tmpdir) / "merged_report_worker_7.csv"

        df0 = pd.DataFrame([{
            "Subject": "CASE_01.nii.gz",
            "SSIM": 0.85,
            "PSNR": 28.5,
            "L1_Error": 1.0,
            "Reference_Coverage": 1.0,
            "run_id": "run_test",
            "metric_version": METRIC_VERSION,
        }])
        df0.to_csv(worker0, index=False)

        # 1. Missing expected shard -> RuntimeError
        with pytest.raises(RuntimeError, match="missing 1 of 2 expected shard"):
            merge_worker_csvs(out_csv, total_workers=2)

        # 2. Stale / unversioned shard -> ValueError
        df_unversioned = pd.DataFrame([{
            "Subject": "CASE_02.nii.gz",
            "SSIM": 0.80,
            "PSNR": 27.0,
            "L1_Error": 999.0,
        }])
        df_unversioned.to_csv(worker1, index=False)
        with pytest.raises(ValueError, match="missing 'metric_version'"):
            merge_worker_csvs(out_csv, total_workers=2)

        # 3. Valid worker_1 added, but an obsolete worker_7 exists on disk
        df1 = pd.DataFrame([{
            "Subject": "CASE_02.nii.gz",
            "SSIM": 0.90,
            "PSNR": 30.0,
            "L1_Error": 3.0,
            "Reference_Coverage": 1.0,
            "run_id": "run_test",
            "metric_version": METRIC_VERSION,
        }])
        df1.to_csv(worker1, index=False)
        df_unversioned.to_csv(worker_stale, index=False)

        # Merging with total_workers=2 must merge ONLY worker_0 and worker_1 (L1=(1+3)/2=2.0)
        # and ignore obsolete worker_7 (L1=999)
        merge_worker_csvs(out_csv, total_workers=2)
        assert out_csv.exists()
        merged = pd.read_csv(out_csv)
        assert len(merged) == 2
        assert set(merged["Subject"]) == {"CASE_01.nii.gz", "CASE_02.nii.gz"}
        assert merged["L1_Error"].mean() == pytest.approx(2.0, abs=1e-5)


def test_production_showcase_compressor_architecture_label():
    """Finding 5: SHOWCASE.md accurately labels the 3D autoencoder without AutoencoderKL or Gaussian posterior."""
    showcase_path = Path(BASE_DIR) / "docs" / "SHOWCASE.md"
    content = showcase_path.read_text(encoding="utf-8")
    assert "AutoencoderKL" not in content, "SHOWCASE.md must not label compressor as AutoencoderKL"
    assert "Gaussian posterior" not in content, "SHOWCASE.md must not claim Gaussian posterior"
    assert "LightweightAutoEncoder3D" in content, "SHOWCASE.md must reference production LightweightAutoEncoder3D"
    assert "deterministic continuous 3D autoencoder" in content


def test_production_evaluate_synthesis_partial_support_metric_masking():
    """Finding 2 (f9e5ad2): Voxel metrics restrict evaluation to valid reference support and do not score missing data as -1000 HU."""
    shape = (8, 8, 8)
    real_vol = np.ones(shape, dtype=np.float32) * 100.0
    fake_vol = np.ones(shape, dtype=np.float32) * 100.0

    # 50% physical coverage: only half the volume has valid reference support
    support_mask = np.zeros(shape, dtype=bool)
    support_mask[:4, :, :] = True

    # If the uncovered half were filled with -1000 HU and unmasked, whole-volume L1 would be 550 HU.
    # With support masking, supported voxels are identical (100 vs 100) -> L1 is 0.0 and Reference_Coverage is 0.5.
    real_vol[~support_mask] = -1000.0  # fill value for missing reference
    metrics = compute_metrics_3d(real_vol, fake_vol, support_mask=support_mask)

    assert metrics["L1_Error"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["Body_L1_Error"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["Reference_Coverage"] == pytest.approx(0.5, abs=1e-5)


def test_production_evaluate_synthesis_merge_rejects_conflicting_duplicates():
    """Finding 3 (f9e5ad2): merge_worker_csvs rejects conflicting duplicate subject rows across shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "merged_report.csv"
        worker0 = Path(tmpdir) / "merged_report_worker_0.csv"
        worker1 = Path(tmpdir) / "merged_report_worker_1.csv"

        # Worker 0 and Worker 1 have conflicting rows for the same subject CASE_01
        df0 = pd.DataFrame([{
            "Subject": "CASE_01.nii.gz",
            "SSIM": 0.85,
            "PSNR": 28.5,
            "L1_Error": 999.0,
            "Reference_Coverage": 1.0,
            "run_id": "test_run",
            "fake_sha256": "fake_sha_old",
            "real_sha256": "real_sha_old",
            "metric_version": METRIC_VERSION,
        }])
        df1 = pd.DataFrame([{
            "Subject": "CASE_01.nii.gz",
            "SSIM": 0.85,
            "PSNR": 28.5,
            "L1_Error": 1.0,
            "Reference_Coverage": 1.0,
            "run_id": "test_run",
            "fake_sha256": "fake_sha_new",
            "real_sha256": "real_sha_new",
            "metric_version": METRIC_VERSION,
        }])
        df0.to_csv(worker0, index=False)
        df1.to_csv(worker1, index=False)

        with pytest.raises(ValueError, match="Conflicting .* found for duplicate subject 'CASE_01.nii.gz'"):
            merge_worker_csvs(out_csv, total_workers=2)


def test_production_evaluate_synthesis_idle_workers_emit_empty_shards_and_merge():
    """Finding 4 (f9e5ad2): Idle workers with zero assigned cases emit empty schema-bearing shards and merge cleanly."""
    from code.project10.evaluation.evaluate_synthesis import main as eval_main
    import unittest.mock as mock

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_dir = os.path.join(tmpdir, "fake")
        real_dir = os.path.join(tmpdir, "real")
        out_csv = Path(tmpdir) / "merged_eval.csv"
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        shape = (8, 8, 8)
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        # Only 1 volume in cohort
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), os.path.join(fake_dir, "CASE_01.nii.gz"))
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), os.path.join(real_dir, "CASE_01.nii.gz"))

        # Run Worker 0 of 2: processes CASE_01
        args_w0 = [
            "evaluate_synthesis.py",
            "--fake_dir", fake_dir,
            "--real_dir", real_dir,
            "--output_csv", str(out_csv),
            "--array_id", "0",
            "--total_workers", "2",
            "--device", "cpu",
        ]
        with mock.patch("sys.argv", args_w0):
            eval_main()

        # Run Worker 1 of 2: 0 cases assigned (idle worker) -> must emit schema-bearing empty shard
        args_w1 = [
            "evaluate_synthesis.py",
            "--fake_dir", fake_dir,
            "--real_dir", real_dir,
            "--output_csv", str(out_csv),
            "--array_id", "1",
            "--total_workers", "2",
            "--device", "cpu",
        ]
        with mock.patch("sys.argv", args_w1):
            eval_main()

        worker1_csv = out_csv.parent / f"{out_csv.stem}_worker_1{out_csv.suffix}"
        assert worker1_csv.exists(), "Idle worker must write empty schema-bearing shard"
        w1_df = pd.read_csv(worker1_csv)
        assert len(w1_df) == 0
        assert "metric_version" in w1_df.columns

        # Merge with total_workers=2: must succeed without RuntimeError
        merge_worker_csvs(out_csv, total_workers=2)
        assert out_csv.exists()
        merged = pd.read_csv(out_csv)
        assert len(merged) == 1
        assert merged.iloc[0]["Subject"] == "CASE_01.nii.gz"


def test_production_evaluate_synthesis_resume_invalidates_old_metric_version():
    """Finding 1 (81137d6): evaluate_synthesis invalidates cached records from old metric versions or missing Reference_Coverage."""
    from code.project10.evaluation.evaluate_synthesis import main as eval_main
    import unittest.mock as mock

    with tempfile.TemporaryDirectory() as tmpdir:
        fake_dir = os.path.join(tmpdir, "fake")
        real_dir = os.path.join(tmpdir, "real")
        out_csv = Path(tmpdir) / "synthesis_metrics.csv"
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        shape = (8, 8, 8)
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), os.path.join(fake_dir, "CASE_01.nii.gz"))
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), affine), os.path.join(real_dir, "CASE_01.nii.gz"))

        # Pre-seed worker CSV with a cached record from old version "v2_canonical_spatial" lacking Reference_Coverage
        # and reporting obsolete L1_Error=550.0
        old_df = pd.DataFrame([{
            "Subject": "CASE_01.nii.gz",
            "SSIM": 0.50,
            "PSNR": 20.0,
            "L1_Error": 550.0,
            "fake_sha256": "fake_sha_01",
            "real_sha256": "real_sha_01",
            "real_file": "CASE_01.nii.gz",
            "fake_mtime": 1730000000.0,
            "metric_version": "v2_canonical_spatial",
        }])
        old_df.to_csv(out_csv, index=False)

        args = [
            "evaluate_synthesis.py",
            "--fake_dir", fake_dir,
            "--real_dir", real_dir,
            "--output_csv", str(out_csv),
            "--total_workers", "1",
            "--device", "cpu",
        ]
        with mock.patch("sys.argv", args):
            eval_main()

        res_df = pd.read_csv(out_csv)
        assert len(res_df) == 1
        assert res_df.iloc[0]["metric_version"] == METRIC_VERSION
        assert "Reference_Coverage" in res_df.columns
        # Fresh computation on identical 0-volumes produces L1_Error == 0.0 (not obsolete 550.0)
        assert res_df.iloc[0]["L1_Error"] == pytest.approx(0.0, abs=1e-5)


def test_production_evaluate_synthesis_ssim_unsupported_pixels_ignored():
    """Finding 2 (81137d6): SSIM footprint policy evaluates only fully supported windows; returns NaN when none qualify."""
    shape = (32, 32, 8)
    data_range = 3000.0
    fake = np.ones(shape, dtype=np.float32) * 100.0

    # Oblique/triangular support region on each 32x32 slice
    support_mask = np.zeros(shape, dtype=bool)
    for r in range(8, 24):
        for c in range(8, 24):
            if 24 <= r + c <= 40:
                support_mask[r, c, :] = True

    real1 = np.ones(shape, dtype=np.float32) * 100.0
    real1[~support_mask] = -1000.0  # fill air

    real2 = np.ones(shape, dtype=np.float32) * 100.0
    real2[~support_mask] = 2000.0   # contrasting fill in unsupported region

    m1 = compute_metrics_3d(real1, fake, support_mask=support_mask)
    m2 = compute_metrics_3d(real2, fake, support_mask=support_mask)

    # Both must match exactly (1.0) because unsupported pixels are excluded from valid windows
    assert np.isclose(m1["SSIM"], 1.0)
    assert np.isclose(m2["SSIM"], 1.0)
    assert np.isclose(m1["L1_Error"], 0.0)
    assert np.isclose(m2["L1_Error"], 0.0)

    # When support is too narrow for any 7x7 window to fit, SSIM reports NaN, not 0.0
    narrow_support = np.zeros(shape, dtype=bool)
    narrow_support[15:18, 15:18, :] = True  # only 3x3 support -> no 7x7 window fits
    m_narrow = compute_metrics_3d(real1, fake, support_mask=narrow_support)
    assert np.isnan(m_narrow["SSIM"])


def test_production_evaluate_synthesis_merge_requires_run_id_consistency():
    """Finding 4 (81137d6): merge_worker_csvs requires run_id provenance and rejects mismatched run IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "merged_report.csv"
        worker0 = Path(tmpdir) / "merged_report_worker_0.csv"
        worker1 = Path(tmpdir) / "merged_report_worker_1.csv"

        # 1. Shards without run_id must be rejected
        df0_no_run = pd.DataFrame([{
            "Subject": "CASE_01.nii.gz",
            "SSIM": 0.85,
            "PSNR": 28.5,
            "L1_Error": 1.0,
            "Reference_Coverage": 1.0,
            "metric_version": METRIC_VERSION,
        }])
        df1_no_run = pd.DataFrame([{
            "Subject": "CASE_02.nii.gz",
            "SSIM": 0.90,
            "PSNR": 30.0,
            "L1_Error": 2.0,
            "Reference_Coverage": 1.0,
            "metric_version": METRIC_VERSION,
        }])
        df0_no_run.to_csv(worker0, index=False)
        df1_no_run.to_csv(worker1, index=False)
        with pytest.raises(ValueError, match="missing required 'run_id' provenance"):
            merge_worker_csvs(out_csv, total_workers=2)

        # 2. Shards with conflicting run_ids must be rejected
        df0_run_a = df0_no_run.copy()
        df0_run_a["run_id"] = "run_alpha"
        df1_run_b = df1_no_run.copy()
        df1_run_b["run_id"] = "run_beta"
        df0_run_a.to_csv(worker0, index=False)
        df1_run_b.to_csv(worker1, index=False)
        with pytest.raises(ValueError, match="Worker shards originate from conflicting runs"):
            merge_worker_csvs(out_csv, total_workers=2)

        # 3. Shards with matching run_id succeed
        df1_run_a = df1_no_run.copy()
        df1_run_a["run_id"] = "run_alpha"
        df1_run_a.to_csv(worker1, index=False)
        merge_worker_csvs(out_csv, total_workers=2, expected_run_id="run_alpha")
        assert out_csv.exists()
        merged = pd.read_csv(out_csv)
        assert len(merged) == 2
        assert merged["run_id"].unique().tolist() == ["run_alpha"]
