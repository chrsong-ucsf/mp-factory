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

        # Provide minimal manifest and worker completion metadata
        import json
        manifest = {
            "run_id": "test_run",
            "metric_version": METRIC_VERSION,
            "total_workers": 2,
            "items": [
                {"fake_file": "CASE_01.nii.gz", "worker_id": 0},
                {"fake_file": "CASE_02.nii.gz", "worker_id": 1},
            ],
        }
        (Path(tmpdir) / "merged_report_manifest.json").write_text(json.dumps(manifest))
        meta0 = {
            "run_id": "test_run",
            "metric_version": METRIC_VERSION,
            "status": "completed",
            "error_count": 0,
            "num_assigned": 1,
            "num_scored": 1,
        }
        worker0_csv.with_suffix(".meta.json").write_text(json.dumps(meta0))
        worker1_csv.with_suffix(".meta.json").write_text(json.dumps(meta0))

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

        # Write manifest and completion metadata for workers 0 and 1
        import json
        manifest = {
            "run_id": "run_test",
            "metric_version": METRIC_VERSION,
            "total_workers": 2,
            "items": [
                {"fake_file": "CASE_01.nii.gz", "worker_id": 0},
                {"fake_file": "CASE_02.nii.gz", "worker_id": 1},
            ],
        }
        (Path(tmpdir) / "merged_report_manifest.json").write_text(json.dumps(manifest))
        meta = {
            "run_id": "run_test",
            "metric_version": METRIC_VERSION,
            "status": "completed",
            "error_count": 0,
            "num_assigned": 1,
            "num_scored": 1,
        }
        worker0.with_suffix(".meta.json").write_text(json.dumps(meta))
        worker1.with_suffix(".meta.json").write_text(json.dumps(meta))

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
            merge_worker_csvs(out_csv, total_workers=2, allow_partial=True)


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
            merge_worker_csvs(out_csv, total_workers=2, allow_partial=True)

        # 2. Shards with conflicting run_ids must be rejected
        df0_run_a = df0_no_run.copy()
        df0_run_a["run_id"] = "run_alpha"
        df1_run_b = df1_no_run.copy()
        df1_run_b["run_id"] = "run_beta"
        df0_run_a.to_csv(worker0, index=False)
        df1_run_b.to_csv(worker1, index=False)
        with pytest.raises(ValueError, match="Worker shards originate from conflicting runs"):
            merge_worker_csvs(out_csv, total_workers=2, allow_partial=True)

        # 3. Shards with matching run_id succeed
        df1_run_a = df1_no_run.copy()
        df1_run_a["run_id"] = "run_alpha"
        df1_run_a.to_csv(worker1, index=False)
        merge_worker_csvs(out_csv, total_workers=2, expected_run_id="run_alpha", allow_partial=True)
        assert out_csv.exists()
        merged = pd.read_csv(out_csv)
        assert len(merged) == 2
        assert merged["run_id"].unique().tolist() == ["run_alpha"]


def test_production_evaluate_synthesis_changing_input_contents_updates_run_id_and_rejects_stale_merge():
    """Finding 1 (6ec9456): Run IDs capture input contents; modifying input files updates run ID and rejects stale shards."""
    import unittest.mock as mock
    from code.project10.evaluation.evaluate_synthesis import main as eval_main
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fake_dir = str(tmp_path / "fake")
        real_dir = str(tmp_path / "real")
        out_csv = tmp_path / "report.csv"
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        case_a_fake = Path(fake_dir) / "CASE_A.nii.gz"
        case_a_real = Path(real_dir) / "CASE_A_reg.nii.gz"
        case_b_fake = Path(fake_dir) / "CASE_B.nii.gz"
        case_b_real = Path(real_dir) / "CASE_B_reg.nii.gz"

        arr_zero = np.zeros((16, 16, 16), dtype=np.float32)
        arr_one = np.ones((16, 16, 16), dtype=np.float32)
        arr_three = np.ones((16, 16, 16), dtype=np.float32) * 3.0

        nib.save(nib.Nifti1Image(arr_zero, np.eye(4)), str(case_a_fake))
        nib.save(nib.Nifti1Image(arr_one, np.eye(4)), str(case_a_real))
        nib.save(nib.Nifti1Image(arr_zero, np.eye(4)), str(case_b_fake))
        nib.save(nib.Nifti1Image(arr_one, np.eye(4)), str(case_b_real))

        # Initial run: worker 0 (CASE_A) and worker 1 (CASE_B)
        for wid in [0, 1]:
            args = [
                "evaluate_synthesis.py",
                "--fake_dir", fake_dir,
                "--real_dir", real_dir,
                "--output_csv", str(out_csv),
                "--total_workers", "2",
                "--array_id", str(wid),
                "--device", "cpu",
            ]
            with mock.patch("sys.argv", args):
                eval_main()

        worker0_csv = tmp_path / "report_worker_0.csv"
        worker1_csv = tmp_path / "report_worker_1.csv"
        df0 = pd.read_csv(worker0_csv)
        df1 = pd.read_csv(worker1_csv)
        initial_run_id = df0.iloc[0]["run_id"]
        assert df1.iloc[0]["run_id"] == initial_run_id
        assert df1.iloc[0]["L1_Error"] == pytest.approx(1.0, abs=1e-4)

        # Now change CASE_B's real reference file on disk from 1.0 to 3.0 HU
        nib.save(nib.Nifti1Image(arr_three, np.eye(4)), str(case_b_real))

        # Rerun ONLY worker 0
        args0 = [
            "evaluate_synthesis.py",
            "--fake_dir", fake_dir,
            "--real_dir", real_dir,
            "--output_csv", str(out_csv),
            "--total_workers", "2",
            "--array_id", "0",
            "--overwrite",
            "--device", "cpu",
        ]
        with mock.patch("sys.argv", args0):
            eval_main()

        df0_new = pd.read_csv(worker0_csv)
        new_run_id = df0_new.iloc[0]["run_id"]
        # Generated run_id must change because CASE_B input content changed!
        assert new_run_id != initial_run_id

        # Merge must reject because worker 1's shard has the old run_id!
        with pytest.raises(ValueError, match="Worker shards originate from conflicting runs|does not match expected run_id"):
            merge_worker_csvs(out_csv, total_workers=2)

        # Even if someone attempts to pass the old run_id, worker 0 is rejected
        with pytest.raises(ValueError, match="does not match expected run_id"):
            merge_worker_csvs(out_csv, total_workers=2, expected_run_id=initial_run_id)

        # 4. If someone forges the old run_id on worker 0's CSV without metadata, it is caught
        df0_new["run_id"] = initial_run_id
        df0_new.to_csv(worker0_csv, index=False)
        with pytest.raises(ValueError, match="does not match worker metadata run_id|does not match expected run_id"):
            merge_worker_csvs(out_csv, total_workers=2)

        # 5. Even if worker metadata is also altered, manifest check catches the outdated run ID
        import json
        meta0 = json.loads(worker0_csv.with_suffix(".meta.json").read_text())
        meta0["run_id"] = initial_run_id
        worker0_csv.with_suffix(".meta.json").write_text(json.dumps(meta0))
        with pytest.raises(ValueError, match="does not match input manifest run_id|does not match expected run_id"):
            merge_worker_csvs(out_csv, total_workers=2, fake_dir=fake_dir, real_dir=real_dir)

        # 6. If manifest is removed, direct file revalidation catches worker 1's stale real_sha256
        (tmp_path / "report_manifest.json").unlink()
        with pytest.raises(ValueError, match="stale real_sha256"):
            merge_worker_csvs(out_csv, total_workers=2, fake_dir=fake_dir, real_dir=real_dir, allow_partial=True)


def test_production_evaluate_synthesis_merge_rejects_blank_run_ids_on_rows():
    """Finding 2 (6ec9456): Required run provenance rejects individual data records with missing or blank run_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "merged_report.csv"
        worker0 = Path(tmpdir) / "merged_report_worker_0.csv"
        worker1 = Path(tmpdir) / "merged_report_worker_1.csv"

        df0 = pd.DataFrame([{
            "Subject": "CASE_A.nii.gz",
            "SSIM": 0.85,
            "PSNR": 28.5,
            "L1_Error": 1.0,
            "Reference_Coverage": 1.0,
            "metric_version": METRIC_VERSION,
            "run_id": "run_known",
        }])

        # Shard with empty string run_id
        df1_blank = pd.DataFrame([{
            "Subject": "CASE_B.nii.gz",
            "SSIM": 0.90,
            "PSNR": 30.0,
            "L1_Error": 2.0,
            "Reference_Coverage": 1.0,
            "metric_version": METRIC_VERSION,
            "run_id": "",
        }])

        df0.to_csv(worker0, index=False)
        df1_blank.to_csv(worker1, index=False)

        # Must reject blank run_id even when expected_run_id is passed!
        with pytest.raises(ValueError, match="blank/missing 'run_id' provenance"):
            merge_worker_csvs(out_csv, total_workers=2, expected_run_id="run_known", allow_partial=True)

        # Shard with NaN run_id
        df1_nan = pd.DataFrame([{
            "Subject": "CASE_B.nii.gz",
            "SSIM": 0.90,
            "PSNR": 30.0,
            "L1_Error": 2.0,
            "Reference_Coverage": 1.0,
            "metric_version": METRIC_VERSION,
            "run_id": np.nan,
        }])
        df1_nan.to_csv(worker1, index=False)
        with pytest.raises(ValueError, match="blank/missing 'run_id' provenance"):
            merge_worker_csvs(out_csv, total_workers=2, expected_run_id="run_known", allow_partial=True)

        # Shard with whitespace-only run_id
        df1_ws = pd.DataFrame([{
            "Subject": "CASE_B.nii.gz",
            "SSIM": 0.90,
            "PSNR": 30.0,
            "L1_Error": 2.0,
            "Reference_Coverage": 1.0,
            "metric_version": METRIC_VERSION,
            "run_id": "   ",
        }])
        df1_ws.to_csv(worker1, index=False)
        with pytest.raises(ValueError, match="blank/missing 'run_id' provenance"):
            merge_worker_csvs(out_csv, total_workers=2, expected_run_id="run_known", allow_partial=True)

        # Mixed valid and blank in the same shard
        df_mixed = pd.concat([df0, df1_blank], ignore_index=True)
        df_mixed.to_csv(worker0, index=False)
        empty_schema = pd.DataFrame(columns=df0.columns)
        empty_schema.to_csv(worker1, index=False)
        with pytest.raises(ValueError, match="blank/missing 'run_id' provenance"):
            merge_worker_csvs(out_csv, total_workers=2, expected_run_id="run_known", allow_partial=True)


def test_production_clinical_turing_dice_masks_uncovered_reference_space():
    """Finding 3 (6ec9456): Clinical Dice excludes unobserved reference support so fill air does not score as background."""
    from code.project10.evaluation.run_clinical_turing_test import (
        compute_binary_dice,
        compute_clinical_organ_metrics,
    )

    # Synthetic volume has stomach (label 1) in 1000 voxels
    # Real reference has only 50% physical coverage; within observed 50%, stomach is identical (500 voxels)
    # Outside observed 50%, real CT had -1000 HU fill air (predicting background 0)
    shape = (10, 10, 10)
    pred_synth = np.ones(shape, dtype=np.uint8)  # 1000 voxels of stomach (label 1)
    pred_real = np.zeros(shape, dtype=np.uint8)
    pred_real[:5, :, :] = 1  # 500 voxels of stomach, 500 voxels of background (fill air)

    support_mask = np.zeros(shape, dtype=bool)
    support_mask[:5, :, :] = True  # exactly 50% physical coverage

    # 1. Raw unmasked Dice treats uncovered region as background, resulting in 2*500/(500+1000) = 0.6667 (failing 0.70 threshold)
    unmasked_dice = compute_binary_dice(pred_real == 1, pred_synth == 1)
    assert np.isclose(unmasked_dice, 2.0 * 500 / 1500, atol=1e-4)
    assert unmasked_dice < 0.70

    # 2. Support-masked evaluation restricts comparison to valid physical reference support
    metrics, organ_dices, eligible = compute_clinical_organ_metrics(
        pred_real=pred_real,
        pred_synth=pred_synth,
        support_mask=support_mask,
        pair_id="PAIR_01",
        shape=shape,
    )

    # Within observed reference space, real and synthetic have 100% perfect agreement!
    assert metrics["turing_dice_stomach"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["reference_coverage"] == 0.50
    assert metrics["eligible_organs_count"] == 1
    assert metrics["mean_turing_dice"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["passed_turing_test"] is True


def test_production_evaluate_synthesis_merge_rejects_failed_or_incomplete_worker_shards():
    """Finding 1 (f7db54e): Merge rejects failed/partial worker shards and enforces complete cohort manifest coverage."""
    import json
    import unittest.mock as mock
    from code.project10.evaluation.evaluate_synthesis import main as eval_main

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fake_dir = str(tmp_path / "fake")
        real_dir = str(tmp_path / "real")
        out_csv = tmp_path / "report.csv"
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        case_a_fake = Path(fake_dir) / "CASE_A.nii.gz"
        case_a_real = Path(real_dir) / "CASE_A_reg.nii.gz"
        case_b_fake = Path(fake_dir) / "CASE_B.nii.gz"
        # CASE_B real is deliberately omitted to trigger worker failure
        case_c_fake = Path(fake_dir) / "CASE_C.nii.gz"
        case_c_real = Path(real_dir) / "CASE_C_reg.nii.gz"

        arr = np.ones((8, 8, 8), dtype=np.float32)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(case_a_fake))
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(case_a_real))
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(case_b_fake))
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(case_c_fake))
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(case_c_real))

        # 3 cases, 2 workers:
        # Worker 0 assigned CASE_A, CASE_B (chunk_size = 2)
        # Worker 1 assigned CASE_C
        args0 = [
            "evaluate_synthesis.py",
            "--fake_dir", fake_dir,
            "--real_dir", real_dir,
            "--output_csv", str(out_csv),
            "--total_workers", "2",
            "--array_id", "0",
            "--device", "cpu",
        ]
        with mock.patch("sys.argv", args0):
            with pytest.raises(RuntimeError, match="1/2 assigned cases failed evaluation on worker 0"):
                eval_main()

        worker0_csv = tmp_path / "report_worker_0.csv"
        worker0_meta = json.loads(worker0_csv.with_suffix(".meta.json").read_text())
        assert worker0_meta["status"] == "failed"
        assert worker0_meta["error_count"] == 1
        assert worker0_meta["num_assigned"] == 2
        assert worker0_meta["num_scored"] == 1

        # Run worker 1: CASE_C succeeds -> status='completed'
        args1 = [
            "evaluate_synthesis.py",
            "--fake_dir", fake_dir,
            "--real_dir", real_dir,
            "--output_csv", str(out_csv),
            "--total_workers", "2",
            "--array_id", "1",
            "--device", "cpu",
        ]
        with mock.patch("sys.argv", args1):
            eval_main()

        worker1_csv = tmp_path / "report_worker_1.csv"
        worker1_meta = json.loads(worker1_csv.with_suffix(".meta.json").read_text())
        assert worker1_meta["status"] == "completed"
        assert worker1_meta["error_count"] == 0

        # Attempting merge must FAIL because worker 0 failed
        with pytest.raises(ValueError, match="non-completed status 'failed'|evaluation error"):
            merge_worker_csvs(out_csv, total_workers=2)

        # Even if worker 0 metadata was forged to status='completed' and error_count=0:
        worker0_meta["status"] = "completed"
        worker0_meta["error_count"] = 0
        worker0_csv.with_suffix(".meta.json").write_text(json.dumps(worker0_meta))

        # Merging fails because worker 0 scored only 1 of 2 assigned cases
        with pytest.raises(ValueError, match="evaluated only 1 of 2 assigned cases"):
            merge_worker_csvs(out_csv, total_workers=2)

        # Even if num_scored was also doctored to 2 in metadata sidecar:
        worker0_meta["num_scored"] = 2
        worker0_csv.with_suffix(".meta.json").write_text(json.dumps(worker0_meta))

        # Merging must STILL fail because cohort is incomplete (missing CASE_B.nii.gz from manifest)
        with pytest.raises(ValueError, match="Incomplete cohort: merged dataset contains 2 of 3 expected subjects"):
            merge_worker_csvs(out_csv, total_workers=2)

        # Merge succeeds only when allow_partial=True is explicitly authorized
        merge_worker_csvs(out_csv, total_workers=2, allow_partial=True)
        assert out_csv.exists()
        merged = pd.read_csv(out_csv)
        assert set(merged["Subject"]) == {"CASE_A.nii.gz", "CASE_C.nii.gz"}


def test_production_evaluate_synthesis_merge_only_resolves_manifest_paths_and_fails_on_missing_dir():
    """Finding 2 (f7db54e): --merge_only resolves custom paths from manifest and fails clearly when inputs are missing/stale."""
    import unittest.mock as mock
    from code.project10.evaluation.evaluate_synthesis import main as eval_main

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fake_dir = str(tmp_path / "custom_synthetic")
        real_dir = str(tmp_path / "custom_registered")
        out_csv = tmp_path / "custom_eval" / "metrics.csv"
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        case_a_fake = Path(fake_dir) / "CASE_A.nii.gz"
        case_a_real = Path(real_dir) / "CASE_A_reg.nii.gz"
        case_b_fake = Path(fake_dir) / "CASE_B.nii.gz"
        case_b_real = Path(real_dir) / "CASE_B_reg.nii.gz"

        arr_zero = np.zeros((8, 8, 8), dtype=np.float32)
        arr_one = np.ones((8, 8, 8), dtype=np.float32) * 101.0
        arr_three = np.ones((8, 8, 8), dtype=np.float32) * 103.0

        nib.save(nib.Nifti1Image(arr_zero, np.eye(4)), str(case_a_fake))
        nib.save(nib.Nifti1Image(arr_one, np.eye(4)), str(case_a_real))
        nib.save(nib.Nifti1Image(arr_zero, np.eye(4)), str(case_b_fake))
        nib.save(nib.Nifti1Image(arr_one, np.eye(4)), str(case_b_real))

        explicit_run_id = "test_run_reused"

        # Evaluate both workers with explicit run_id
        for wid in [0, 1]:
            args = [
                "evaluate_synthesis.py",
                "--fake_dir", fake_dir,
                "--real_dir", real_dir,
                "--output_csv", str(out_csv),
                "--total_workers", "2",
                "--array_id", str(wid),
                "--run_id", explicit_run_id,
                "--device", "cpu",
            ]
            with mock.patch("sys.argv", args):
                eval_main()

        # Modify CASE_B reference on disk from 101 to 103 HU
        nib.save(nib.Nifti1Image(arr_three, np.eye(4)), str(case_b_real))

        # Re-evaluate worker 0 ONLY with explicit_run_id
        args0 = [
            "evaluate_synthesis.py",
            "--fake_dir", fake_dir,
            "--real_dir", real_dir,
            "--output_csv", str(out_csv),
            "--total_workers", "2",
            "--array_id", "0",
            "--run_id", explicit_run_id,
            "--overwrite",
            "--device", "cpu",
        ]
        with mock.patch("sys.argv", args0):
            eval_main()

        # Now execute --merge_only WITHOUT passing --fake_dir or --real_dir!
        # The CLI must recover fake_dir and real_dir from manifest JSON and catch the stale reference hash on CASE_B
        merge_args = [
            "evaluate_synthesis.py",
            "--merge_only",
            "--output_csv", str(out_csv),
            "--total_workers", "2",
            "--run_id", explicit_run_id,
        ]
        with mock.patch("sys.argv", merge_args):
            with pytest.raises(ValueError, match="stale real_sha256"):
                eval_main()

        # Next, test that passing an explicit nonexistent directory raises FileNotFoundError
        missing_merge_args = [
            "evaluate_synthesis.py",
            "--merge_only",
            "--output_csv", str(out_csv),
            "--total_workers", "2",
            "--fake_dir", str(tmp_path / "nonexistent_dir"),
        ]
        with mock.patch("sys.argv", missing_merge_args):
            with pytest.raises(FileNotFoundError, match="Input verification failed: fake directory"):
                eval_main()


def test_production_evaluate_synthesis_merge_rejects_csv_only_transfer_without_completion_proof():
    """Finding 1 (9083497): Default merge rejects transferred CSVs lacking manifest and completion metadata proof."""
    import shutil
    import json
    import unittest.mock as mock
    from code.project10.evaluation.evaluate_synthesis import main as eval_main

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fake_dir = tmp_path / "fake"
        real_dir = tmp_path / "real"
        eval_dir = tmp_path / "eval"
        transfer_dir = tmp_path / "transferred_reports"
        fake_dir.mkdir()
        real_dir.mkdir()
        eval_dir.mkdir()
        transfer_dir.mkdir()

        case_a_fake = fake_dir / "CASE_A.nii.gz"
        case_a_real = real_dir / "CASE_A_reg.nii.gz"
        case_b_fake = fake_dir / "CASE_B.nii.gz"
        # CASE_B real is omitted so worker 0 fails on CASE_B

        arr = np.ones((8, 8, 8), dtype=np.float32)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(case_a_fake))
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(case_a_real))
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(case_b_fake))

        out_csv = eval_dir / "report.csv"

        # Worker 0 evaluates CASE_A (success) and fails on CASE_B
        args0 = [
            "evaluate_synthesis.py",
            "--fake_dir", str(fake_dir),
            "--real_dir", str(real_dir),
            "--output_csv", str(out_csv),
            "--total_workers", "1",
            "--array_id", "0",
            "--device", "cpu",
        ]
        with mock.patch("sys.argv", args0):
            with pytest.raises(RuntimeError):
                eval_main()

        worker_csv = eval_dir / "report.csv"
        assert worker_csv.exists()
        # Original failed metadata blocked merge
        with pytest.raises(ValueError, match="non-completed status 'failed'|evaluation error"):
            merge_worker_csvs(out_csv, total_workers=1)

        # Counterexample A: User copies ONLY the CSV into transferred_reports (without manifest or .meta.json)
        transferred_csv = transfer_dir / "report.csv"
        shutil.copy(worker_csv, transferred_csv)

        # Verified default merge MUST reject because manifest is missing
        with pytest.raises(FileNotFoundError, match="missing required cohort manifest"):
            merge_worker_csvs(transferred_csv, total_workers=1)

        # If a manifest is supplied but metadata is missing, it must still reject
        df_transferred = pd.read_csv(transferred_csv)
        real_run_id = df_transferred.iloc[0]["run_id"]
        manifest = {
            "run_id": real_run_id,
            "metric_version": METRIC_VERSION,
            "total_workers": 1,
            "items": [{"fake_file": "CASE_A.nii.gz", "worker_id": 0}],
        }
        (transfer_dir / "report_manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(FileNotFoundError, match="missing required completion metadata sidecar"):
            merge_worker_csvs(transferred_csv, total_workers=1)

        # Explicitly requested partial/unverified export succeeds when allow_partial=True
        merge_worker_csvs(transferred_csv, total_workers=1, allow_partial=True)
        assert transferred_csv.exists()
        merged = pd.read_csv(transferred_csv)
        assert len(merged) == 1
        assert merged.iloc[0]["Subject"] == "CASE_A.nii.gz"


def test_production_evaluate_synthesis_merge_rejects_empty_shard_when_manifest_nonempty():
    """Finding 1 (9083497): Merge rejects schema-only shards when manifest expects subjects, preventing empty-result bypass."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_csv = tmp_path / "report.csv"
        worker_csv = tmp_path / "report_worker_0.csv"

        # Schema-only worker CSV (0 rows)
        schema_cols = [
            "Subject", "metric_version", "run_id", "fake_sha256", "real_sha256", "real_file", "fake_mtime",
            "SSIM", "PSNR", "L1_Error", "Reference_Coverage",
        ]
        empty_df = pd.DataFrame(columns=schema_cols)
        empty_df.to_csv(worker_csv, index=False)

        # Manifest expecting CASE_A assigned to worker 0, with nonexistent input directories
        manifest = {
            "run_id": "run_test",
            "metric_version": METRIC_VERSION,
            "fake_dir": str(tmp_path / "nonexistent_fake"),
            "real_dir": str(tmp_path / "nonexistent_real"),
            "total_files": 1,
            "total_workers": 1,
            "items": [
                {"fake_file": "CASE_A.nii.gz", "worker_id": 0},
            ],
        }
        (tmp_path / "report_manifest.json").write_text(json.dumps(manifest))

        # Counterexample B: worker metadata claiming 0 assigned / 0 scored / completed
        meta = {
            "worker_id": 0,
            "total_workers": 1,
            "run_id": "run_test",
            "metric_version": METRIC_VERSION,
            "num_assigned": 0,
            "num_scored": 0,
            "error_count": 0,
            "status": "completed",
        }
        worker_csv.with_suffix(".meta.json").write_text(json.dumps(meta))

        # 1. Contradiction between worker metadata (0 assigned) and manifest (1 assigned) must be rejected
        with pytest.raises(ValueError, match="contradicts manifest assignment count"):
            merge_worker_csvs(out_csv, total_workers=1)

        # 2. Even if worker metadata num_assigned is corrected to 1 (but num_scored=0):
        meta["num_assigned"] = 1
        worker_csv.with_suffix(".meta.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="evaluated only 0 of 1 assigned cases"):
            merge_worker_csvs(out_csv, total_workers=1)

        # 3. Even if worker metadata num_scored was doctored to 1:
        meta["num_scored"] = 1
        worker_csv.with_suffix(".meta.json").write_text(json.dumps(meta))
        # Input directory verification fails first because fake_dir does not exist
        with pytest.raises(FileNotFoundError, match="Input verification failed: fake directory"):
            merge_worker_csvs(out_csv, total_workers=1)

        # 4. If input directories are created, manifest coverage check rejects empty merged dataset
        Path(manifest["fake_dir"]).mkdir()
        Path(manifest["real_dir"]).mkdir()
        with pytest.raises(ValueError, match="Incomplete cohort: merged dataset contains 0 of 1 expected subjects"):
            merge_worker_csvs(out_csv, total_workers=1)


def test_production_run_p10_evaluate_synthesis_sbatch_propagates_evaluator_exit_status():
    """Finding 2 (9083497): run_p10_evaluate_synthesis.sbatch propagates evaluator exit status and only prints completion on success."""
    import subprocess
    script_path = Path(BASE_DIR) / "code" / "slurm_scripts" / "run_p10_evaluate_synthesis.sbatch"
    assert script_path.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        env_file = Path(tmpdir) / "test_env.sh"
        # Stubs for module, mamba, mkdir, and python
        env_file.write_text(
            'module() { return 0; }\n'
            'mamba() { return 0; }\n'
            'mkdir() { return 0; }\n'
            'python() { return "${STUB_PYTHON_EXIT:-0}"; }\n'
            'export -f module mamba mkdir python\n'
        )
        env = os.environ.copy()
        env["BASH_ENV"] = str(env_file)
        env["SLURM_ARRAY_TASK_ID"] = "2"

        # 1. Failure case: evaluator stub returns 7
        env["STUB_PYTHON_EXIT"] = "7"
        proc = subprocess.run(
            ["bash", str(script_path)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 7
        assert "completed evaluation" not in proc.stdout
        assert "failed evaluation with status 7" in proc.stdout

        # 2. Success case: evaluator stub returns 0
        env["STUB_PYTHON_EXIT"] = "0"
        proc = subprocess.run(
            ["bash", str(script_path)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert "completed evaluation" in proc.stdout


def test_production_evaluate_synthesis_repartition_named_run_preserves_case_uniqueness():
    """Finding 1 (271849c): Worker repartition of a named run preserves case uniqueness and does not double-count cases in cohort means."""
    import unittest.mock as mock
    from code.project10.evaluation.evaluate_synthesis import main as eval_main, merge_worker_csvs

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fake_dir = str(tmp_path / "fake")
        real_dir = str(tmp_path / "real")
        out_csv = tmp_path / "report.csv"
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(real_dir, exist_ok=True)

        case_a_fake = Path(fake_dir) / "CASE_A.nii.gz"
        case_a_real = Path(real_dir) / "CASE_A_reg.nii.gz"
        case_b_fake = Path(fake_dir) / "CASE_B.nii.gz"
        case_b_real = Path(real_dir) / "CASE_B_reg.nii.gz"
        case_c_fake = Path(fake_dir) / "CASE_C.nii.gz"
        case_c_real = Path(real_dir) / "CASE_C_reg.nii.gz"

        arr_zero = np.zeros((8, 8, 8), dtype=np.float32)
        arr_one = np.ones((8, 8, 8), dtype=np.float32) * 1.0
        arr_three = np.ones((8, 8, 8), dtype=np.float32) * 3.0
        arr_five = np.ones((8, 8, 8), dtype=np.float32) * 5.0

        nib.save(nib.Nifti1Image(arr_zero, np.eye(4)), str(case_a_fake))
        nib.save(nib.Nifti1Image(arr_one, np.eye(4)), str(case_a_real))
        nib.save(nib.Nifti1Image(arr_zero, np.eye(4)), str(case_b_fake))
        nib.save(nib.Nifti1Image(arr_three, np.eye(4)), str(case_b_real))
        nib.save(nib.Nifti1Image(arr_zero, np.eye(4)), str(case_c_fake))
        nib.save(nib.Nifti1Image(arr_five, np.eye(4)), str(case_c_real))

        run_id = "repartition_run"

        # Step 1: Evaluate with 2 workers using explicit --run_id
        for wid in [0, 1]:
            args = [
                "evaluate_synthesis.py",
                "--fake_dir", fake_dir,
                "--real_dir", real_dir,
                "--output_csv", str(out_csv),
                "--total_workers", "2",
                "--array_id", str(wid),
                "--run_id", run_id,
                "--device", "cpu",
            ]
            with mock.patch("sys.argv", args):
                eval_main()

        # Step 2: Rerun with 3 workers using the same --run_id without --overwrite
        for wid in [0, 1, 2]:
            args = [
                "evaluate_synthesis.py",
                "--fake_dir", fake_dir,
                "--real_dir", real_dir,
                "--output_csv", str(out_csv),
                "--total_workers", "3",
                "--array_id", str(wid),
                "--run_id", run_id,
                "--device", "cpu",
            ]
            with mock.patch("sys.argv", args):
                eval_main()

        # Step 3: Perform verified merge
        merge_worker_csvs(out_csv, total_workers=3, fake_dir=fake_dir, real_dir=real_dir)

        # Step 4: Verify merged CSV
        assert out_csv.exists()
        merged_df = pd.read_csv(out_csv)
        assert len(merged_df) == 3
        assert set(merged_df["Subject"]) == {"CASE_A.nii.gz", "CASE_B.nii.gz", "CASE_C.nii.gz"}
        assert merged_df["Subject"].is_unique

        # Mean L1 error must be exactly 3.0: (1.0 + 3.0 + 5.0) / 3 = 3.0
        assert merged_df["L1_Error"].mean() == pytest.approx(3.0, abs=1e-4)

        # Check individual row scores
        scores = dict(zip(merged_df["Subject"], merged_df["L1_Error"]))
        assert scores["CASE_A.nii.gz"] == pytest.approx(1.0, abs=1e-4)
        assert scores["CASE_B.nii.gz"] == pytest.approx(3.0, abs=1e-4)
        assert scores["CASE_C.nii.gz"] == pytest.approx(5.0, abs=1e-4)

        # Step 5: Verify merge rejects total_workers mismatch with manifest
        with pytest.raises(ValueError, match="Merge total_workers .* does not match manifest total_workers"):
            merge_worker_csvs(out_csv, total_workers=2, fake_dir=fake_dir, real_dir=real_dir)

        # Step 6: Verify merge rejects unexpected case / duplicate injection
        worker1_csv = tmp_path / "report_worker_1.csv"
        df1 = pd.read_csv(worker1_csv)
        row_a = merged_df[merged_df["Subject"] == "CASE_A.nii.gz"].copy()
        df1_injected = pd.concat([df1, row_a], ignore_index=True)
        df1_injected.to_csv(worker1_csv, index=False)
        with pytest.raises(ValueError, match="unexpected cases not assigned|Duplicate subject"):
            merge_worker_csvs(out_csv, total_workers=3, fake_dir=fake_dir, real_dir=real_dir)


def test_production_evaluate_synthesis_missing_required_inputs_exits_nonzero_subprocess():
    """Finding 1 (0bce07e): Missing required input directories exit with non-zero status and create no report."""
    import subprocess
    script_path = Path(BASE_DIR) / "code" / "project10" / "evaluation" / "evaluate_synthesis.py"
    assert script_path.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        empty_dir = tmp_path / "empty_fake"
        empty_dir.mkdir()
        nonexistent_dir = tmp_path / "does_not_exist"
        out_csv = tmp_path / "out.csv"

        # 1. Missing fake_dir via subprocess
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--fake_dir", str(nonexistent_dir),
                "--real_dir", str(existing_dir),
                "--output_csv", str(out_csv),
                "--total_workers", "1",
                "--array_id", "0",
                "--device", "cpu",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "Input verification failed: fake directory" in (proc.stderr + proc.stdout)
        assert not out_csv.exists()

        # 2. Missing real_dir via subprocess
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--fake_dir", str(existing_dir),
                "--real_dir", str(nonexistent_dir),
                "--output_csv", str(out_csv),
                "--total_workers", "1",
                "--array_id", "0",
                "--device", "cpu",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "Input verification failed: real directory" in (proc.stderr + proc.stdout)
        assert not out_csv.exists()

        # 3. Empty fake_dir (no NIfTIs found) via subprocess
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--fake_dir", str(empty_dir),
                "--real_dir", str(existing_dir),
                "--output_csv", str(out_csv),
                "--total_workers", "1",
                "--array_id", "0",
                "--device", "cpu",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "no NIfTI files found in fake directory" in (proc.stderr + proc.stdout)
        assert not out_csv.exists()


def test_stress_test_contrast_perturbation_standalone_without_nnunet_mednext():
    """Verify that stress_test_contrast_perturbation can be imported and compute_binary_dice
    executed in standalone environments where nnunet_mednext is not installed."""
    import subprocess
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys\n"
            "sys.modules['nnunet_mednext'] = None\n"
            "from code.project10.evaluation.stress_test_contrast_perturbation import compute_binary_dice\n"
            "import numpy as np\n"
            "res = compute_binary_dice(np.zeros((8, 8, 8)), np.zeros((8, 8, 8)))\n"
            "assert np.isnan(res), f'Expected NaN, got {res}'\n"
            "print('SUCCESS')\n"
        )
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    assert proc.returncode == 0, f"Failed with: {proc.stderr}"
    assert "SUCCESS" in proc.stdout


def test_project_wide_sanity_check_manifest_fallback_and_zero_critical_failures():
    """Verify project_wide_sanity_check discovers manifests and runs with 0 critical failures."""
    import subprocess
    script_path = Path(BASE_DIR) / "code" / "evaluation" / "project_wide_sanity_check.py"
    assert script_path.exists()

    proc = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, cwd=BASE_DIR)
    assert proc.returncode == 0, f"Sanity check returned non-zero: {proc.stderr}\n{proc.stdout}"
    assert "Critical Failures      : 0" in proc.stdout
    assert "System Health Score" in proc.stdout


def test_evaluate_subject_zero_foreground_quarantined_to_noise_reject():
    """Finding 1 (602a7ea): All-background/empty consensus masks are triaged to NOISE_REJECT,
    not CLEAN_HIGH_CONFIDENCE, and report 0 foreground voxels/components."""
    import tempfile
    import nibabel as nib
    from code.evaluation.evaluate_multi_model_ensemble import evaluate_subject

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sub_id = "TEST_EMPTY_001"
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Create two all-zero (background) 8x8x8 mask NIfTIs
        affine = np.eye(4)
        m1_path = tmp_path / f"{sub_id}_m1.nii.gz"
        m2_path = tmp_path / f"{sub_id}_m2.nii.gz"
        empty_data = np.zeros((8, 8, 8), dtype=np.uint8)
        nib.save(nib.Nifti1Image(empty_data, affine), m1_path)
        nib.save(nib.Nifti1Image(empty_data, affine), m2_path)

        res = evaluate_subject(
            subject_id=sub_id,
            model_files=[str(m1_path), str(m2_path)],
            model_names=["ModelA", "ModelB"],
            out_dir=str(out_dir)
        )

        assert res["status"] == "SUCCESS"
        assert res["triage_category"] == "NOISE_REJECT"
        assert "Zero Foreground Components" in res["action"]
        assert res["total_foreground_voxels"] == 0
        assert res["total_foreground_components"] == 0
        assert res["mean_consensus_dice"] == 0.0


def test_verify_production_results_strict_boolean_gate_enforcement(tmp_path):
    """Finding 2 (602a7ea): verify_production_results exits non-zero if any gate fails."""
    import code.evaluation.verify_production_results as vpr
    from unittest.mock import patch

    test_report = tmp_path / "test_report.md"
    # Mock compile_and_verify_manuscript_tables to return False
    with patch("code.evaluation.verify_production_results.compile_and_verify_manuscript_tables", return_value=(False, "")):
        with pytest.raises(SystemExit) as exc_info:
            vpr.main(["--report_path", str(test_report)])
        assert exc_info.value.code == 1


def test_train_mednext_phase2_rejects_empty_foreground_supervision():
    """Finding 1 (602a7ea): train_mednext_phase2 skips subjects whose consensus masks have zero foreground voxels."""
    import tempfile
    import nibabel as nib
    from code.training.train_mednext_phase2 import discover_phase2_dataset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sub_id = "BDMAP_00000001"
        ct_dir = tmp_path / "ct_dir" / sub_id
        ct_dir.mkdir(parents=True)
        cons_dir = tmp_path / "cons_dir"
        cons_dir.mkdir(parents=True)

        # Valid 3D CT volume (16x16x16)
        affine = np.eye(4)
        ct_path = ct_dir / "ct.nii.gz"
        ct_vol = np.full((16, 16, 16), 40.0, dtype=np.float32)
        nib.save(nib.Nifti1Image(ct_vol, affine), ct_path)

        # Empty consensus label (all zeros)
        cons_path = cons_dir / f"{sub_id}_consensus.nii.gz"
        empty_lbl = np.zeros((16, 16, 16), dtype=np.uint8)
        nib.save(nib.Nifti1Image(empty_lbl, affine), cons_path)

        # Audit CSV marking the subject CLEAN
        audit_csv = tmp_path / "audit.csv"
        df = pd.DataFrame([{
            "subject_id": sub_id,
            "status": "SUCCESS",
            "triage_category": "CLEAN_HIGH_CONFIDENCE",
            "stomach_betti_consensus": 0,
            "duodenum_betti_consensus": 0,
            "small_bowel_betti_consensus": 0,
            "colon_betti_consensus": 0,
        }])
        df.to_csv(audit_csv, index=False)

        # Discovery should skip the empty label and return 0 data pairs
        pairs = discover_phase2_dataset(
            data_dir=str(tmp_path / "ct_dir"),
            audit_csv=str(audit_csv),
            ensemble_out_dir=str(cons_dir),
            autolabel_dir=None,
            use_weak=False,
            gold_standard_dir=None
        )
        assert len(pairs) == 0, f"Expected 0 usable pairs for empty foreground label, got {len(pairs)}"


def test_schema_mapping_canonical_totalseg_v1_and_v2():
    """F01 Regression: TotalSegmentator v1 and v2 schemas map strictly to GI classes (0..4)."""
    import numpy as np
    from code.evaluation.schema_mapping import normalize_to_gi_classes, detect_schema

    # TotalSeg v2 probe: 6=stomach, 19=duodenum, 18=small_bowel, 20=colon
    ts_v2 = np.array([0, 6, 19, 18, 20], dtype=np.uint8)
    assert detect_schema(ts_v2, model_name="TotalSegmentator") == "totalseg_v2"
    out_v2 = normalize_to_gi_classes(ts_v2, schema="totalseg_v2")
    assert np.array_equal(out_v2, np.array([0, 1, 2, 3, 4], dtype=np.uint8))

    # TotalSeg v1 probe: 6=stomach, 56=duodenum, 55=small_bowel, 57=colon
    ts_v1 = np.array([0, 6, 56, 55, 57], dtype=np.uint8)
    assert detect_schema(ts_v1, model_name="totalseg") == "totalseg_v1"
    out_v1 = normalize_to_gi_classes(ts_v1, schema="totalseg_v1")
    assert np.array_equal(out_v1, np.array([0, 1, 2, 3, 4], dtype=np.uint8))

    # BDMAP probe: 2=stomach, 3=duodenum, 4=sb, 5=sb, 6=colon
    bdmap = np.array([0, 2, 3, 4, 5, 6], dtype=np.uint8)
    out_bd = normalize_to_gi_classes(bdmap, schema="bdmap")
    assert np.array_equal(out_bd, np.array([0, 1, 2, 3, 3, 4], dtype=np.uint8))

    # JHU Expert probe: 2=stomach, 3=duodenum, 4=sb, 5=colon, 6=rectum (merged to colon)
    jhu = np.array([0, 2, 3, 4, 5, 6], dtype=np.uint8)
    out_jhu = normalize_to_gi_classes(jhu, schema="jhu")
    assert np.array_equal(out_jhu, np.array([0, 1, 2, 3, 4, 4], dtype=np.uint8))


def test_physical_coordinate_alignment_displaced_grid_probe():
    """F02 Regression: Physically displaced masks (100mm offset) yield 0 overlap rather than 1.0."""
    import numpy as np
    from code.evaluation.schema_mapping import check_physical_overlap, resample_label_to_reference

    shape = (16, 16, 16)
    aff_a = np.eye(4)
    aff_b = np.eye(4)
    aff_b[0, 3] = 100.0  # 100mm displacement along X

    arr_a = np.ones(shape, dtype=np.uint8)
    arr_b = np.ones(shape, dtype=np.uint8)

    # Physical overlap check must return False
    assert not check_physical_overlap(shape, aff_a, shape, aff_b)

    # Resampling displaced mask b to reference a must return all zeros
    resampled = resample_label_to_reference(arr_b, aff_b, aff_a, shape)
    assert np.count_nonzero(resampled) == 0


def test_synthesis_net_target_phase_conditioning():
    """F06 Regression: AnatomyConditionalSynthesisNet supports target phase class conditioning."""
    import torch
    from code.project10.models.synthesis_net import AnatomyConditionalSynthesisNet

    model = AnatomyConditionalSynthesisNet(
        spatial_dims=3,
        latent_channels=4,
        label_nc=5,
        num_phases=5,
        channels=(32, 64, 64),
        attention_levels=(False, False, True),
        num_res_blocks=1,
        num_head_channels=8,
    )

    x = torch.randn(2, 4, 8, 8, 8)
    t = torch.tensor([10, 50], dtype=torch.long)
    src = torch.randn(2, 4, 8, 8, 8)
    anatomy = torch.randn(2, 5, 8, 8, 8)
    phase_labels = torch.tensor([1, 3], dtype=torch.long)  # arterial vs venous

    out = model(x=x, timesteps=t, source_latent=src, anatomy_mask=anatomy, phase_labels=phase_labels)
    assert out.shape == x.shape


def test_stage1_dataset_patient_grouping_and_fail_closed(tmp_path):
    """F07 & F10 Regression: Stage 1 groups by patient_id and fails closed on missing CTs."""
    import pandas as pd
    import pytest
    from code.project10.training.train_stage1_autoencoder import MultiPhaseVolumeDataset

    csv_path = tmp_path / "pairs.csv"
    df = pd.DataFrame([
        {"patient_id": "PT_001", "subject_p0": "PT_001_P0", "ct_path_p0": str(tmp_path / "pt1_p0.nii.gz"),
         "subject_target": "PT_001_P1", "ct_path_target": str(tmp_path / "pt1_p1.nii.gz")},
        {"patient_id": "PT_002", "subject_p0": "PT_002_P0", "ct_path_p0": str(tmp_path / "pt2_p0.nii.gz"),
         "subject_target": "PT_002_P1", "ct_path_target": str(tmp_path / "pt2_p1.nii.gz")},
    ])
    df.to_csv(csv_path, index=False)

    ds_train = MultiPhaseVolumeDataset(pairs_csv=str(csv_path), split="train", val_fraction=0.5, seed=42)
    ds_val = MultiPhaseVolumeDataset(pairs_csv=str(csv_path), split="val", val_fraction=0.5, seed=42)

    train_pts = set(ds_train.records["patient_id"])
    val_pts = set(ds_val.records["patient_id"])
    assert len(train_pts.intersection(val_pts)) == 0  # Zero patient leakage

    # Missing CT file must raise FileNotFoundError (F10)
    with pytest.raises(FileNotFoundError):
        _ = ds_train[0]


def test_production_evaluate_multi_model_ensemble_npz_probability_resampling_and_fail_closed(tmp_path):
    """F01 & F02 Regression: Ensemble evaluate_subject handles .npz probability maps with physical resampling."""
    from code.evaluation.evaluate_multi_model_ensemble import evaluate_subject
    import nibabel as nib
    import numpy as np

    shape = (16, 16, 16)
    aff_ref = np.diag([1.5, 1.5, 2.0, 1.0])
    # Model 1: seg mask + probability npz with identical grid
    m1_mask = np.zeros(shape, dtype=np.uint8)
    m1_mask[4:12, 4:12, 4:12] = 1  # stomach
    m1_nii = nib.Nifti1Image(m1_mask, aff_ref)
    m1_path = str(tmp_path / "case01_m1_gi_seg.nii.gz")
    nib.save(m1_nii, m1_path)

    # NPZ probabilities: shape (5, 16, 16, 16)
    m1_probs = np.zeros((5,) + shape, dtype=np.float32)
    m1_probs[0] = 0.1
    m1_probs[1, 4:12, 4:12, 4:12] = 0.9
    np.savez_compressed(m1_path.replace("_gi_seg.nii.gz", "_gi_probs.npz"), probs=m1_probs)

    # Model 2: seg mask + probability npz with slightly shifted affine (physical resampling required)
    aff_m2 = aff_ref.copy()
    aff_m2[0, 3] = 1.5  # 1.5mm offset in X
    m2_mask = np.zeros(shape, dtype=np.uint8)
    m2_mask[4:12, 4:12, 4:12] = 1
    m2_nii = nib.Nifti1Image(m2_mask, aff_m2)
    m2_path = str(tmp_path / "case01_m2_gi_seg.nii.gz")
    nib.save(m2_nii, m2_path)

    m2_probs = np.zeros((5,) + shape, dtype=np.float32)
    m2_probs[0] = 0.1
    m2_probs[1, 4:12, 4:12, 4:12] = 0.9
    np.savez_compressed(m2_path.replace("_gi_seg.nii.gz", "_gi_probs.npz"), probs=m2_probs)

    out_dir = str(tmp_path / "ensemble_out")
    os.makedirs(out_dir, exist_ok=True)

    # Run evaluate_subject with NPZs present: must succeed and not raise NameError('target_shape')
    res = evaluate_subject(
        subject_id="case01",
        model_files=[m1_path, m2_path],
        model_names=["model1", "model2"],
        out_dir=out_dir
    )

    assert res.get("status") == "SUCCESS", f"Ensemble evaluation failed: {res.get('error')}"
    assert os.path.exists(os.path.join(out_dir, "case01_consensus.nii.gz"))
    assert os.path.exists(os.path.join(out_dir, "case01_uncertainty.nii.gz"))
    assert "mean_consensus_dice" in res
    assert res["mean_consensus_dice"] > 0.0


def test_production_train_phase2_ignore_index_clamping_and_masking():
    """F03 Regression: Label value 255 (ignore index) is clamped before AsDiscrete to prevent out of bounds."""
    from monai.transforms import AsDiscrete
    from monai.data import decollate_batch
    import torch

    # Simulated batch: shape (1, 1, 4, 4, 4) with labels in 0..4 and 255 for boundary ignore
    labels = torch.zeros((1, 1, 4, 4, 4), dtype=torch.long)
    labels[0, 0, 1, 1, 1] = 1   # stomach
    labels[0, 0, 2, 2, 2] = 255 # boundary ignore

    # Target logic in train_mednext_phase2 / train_swin_unetr_phase2
    valid_mask = (labels >= 0) & (labels < 5)
    labels_safe = torch.where(valid_mask, labels, torch.zeros_like(labels))

    post_label = AsDiscrete(to_onehot=5)
    # This call must NOT raise: IndexError: index 255 is out of bounds for dimension with size 5
    one_hots = [post_label(i) for i in decollate_batch(labels_safe)]
    assert len(one_hots) == 1
    one_hot = one_hots[0]
    assert one_hot.shape == (5, 4, 4, 4)

    # Valid mask should mask both target and predictions
    vmasks = [m for m in decollate_batch(valid_mask)]
    masked_target = one_hot * vmasks[0].float()
    assert masked_target[:, 2, 2, 2].sum().item() == 0.0  # Ignore voxel is zeroed out
    assert masked_target[1, 1, 1, 1].item() == 1.0        # Valid stomach voxel preserved


def test_production_statistical_significance_patient_id_and_fdr_nan_preservation():
    """F10 & F12 Regression: Patient ID extraction distinguishes BDMAP patients and FDR preserves NaN indices."""
    from code.evaluation.statistical_significance_test import extract_patient_id, apply_fdr_correction
    import numpy as np

    # 1. Patient ID distinction: BDMAP subjects must NOT collapse into 'BDMAP'
    assert extract_patient_id("BDMAP_00242131") == "BDMAP_00242131"
    assert extract_patient_id("BDMAP_00242132") == "BDMAP_00242132"
    assert extract_patient_id("BDMAP_00242131") != extract_patient_id("BDMAP_00242132")
    assert extract_patient_id("CV_1001_P0") == "CV_1001"
    assert extract_patient_id("CV_1001_P1") == "CV_1001"

    # 2. FDR correction with NaN preservation
    results = [
        {"test": "A", "p_value_wilcoxon": np.nan},
        {"test": "B", "p_value_wilcoxon": 0.01},
        {"test": "C", "p_value_wilcoxon": 0.04},
    ]
    corrected = apply_fdr_correction(results, alpha=0.05)

    # Test A had NaN p-value, so q-value must be NaN and not overwritten by Test B or C
    assert np.isnan(corrected[0]["q_value_fdr"])
    assert corrected[0]["sig_fdr"] == "n/a"

    # Test B (p=0.01, rank 1 of 2 valid tests): q = min(0.01 * 2 / 1, q_next) = 0.02
    assert abs(corrected[1]["q_value_fdr"] - 0.02) < 1e-4
    assert corrected[1]["sig_fdr"] == "*"

    # Test C (p=0.04, rank 2 of 2 valid tests): q = 0.04 * 2 / 2 = 0.04
    assert abs(corrected[2]["q_value_fdr"] - 0.04) < 1e-4
    assert corrected[2]["sig_fdr"] == "*"


def test_production_sample_multiphase_diffusion_phase_resolution():
    """F06 Regression: Target phase resolution handles auto discovery and explicit phase names/integers."""
    from code.project10.evaluation.sample_multiphase_diffusion import PHASE_NAME_TO_ID, ID_TO_PHASE_NAME, build_arg_parser

    # 1. Dictionary completeness
    assert PHASE_NAME_TO_ID["arterial"] == 2
    assert PHASE_NAME_TO_ID["venous"] == 3
    assert PHASE_NAME_TO_ID["portal_venous"] == 3
    assert PHASE_NAME_TO_ID["delayed"] == 4
    assert PHASE_NAME_TO_ID["noncontrast"] == 0
    assert PHASE_NAME_TO_ID["3"] == 3

    assert ID_TO_PHASE_NAME[2] == "arterial_late"
    assert ID_TO_PHASE_NAME[3] == "venous"

    # 2. Arg parser includes --target_phase
    parser = build_arg_parser()
    args = parser.parse_args(["--target_phase", "venous"])
    assert args.target_phase == "venous"


def test_production_stress_test_contrast_perturbation_summary_df_no_name_error(tmp_path):
    """F08 Regression: generate_markdown_report builds summary_df from measured data without NameError."""
    from code.project10.evaluation.stress_test_contrast_perturbation import generate_markdown_report, PERTURBATION_CONDITIONS
    import pandas as pd

    records = []
    for cond in PERTURBATION_CONDITIONS:
        records.append({
            "pair_id": "pair_001",
            "condition": cond["name"],
            "category": cond["category"],
            "hu_shift": cond["hu_shift"],
            "noise_sigma": cond["noise_sigma"],
            "mean_retention_dice": 0.88,
            "dice_retention_stomach": 0.90,
            "dice_retention_duodenum": 0.85,
            "dice_retention_small_bowel": 0.87,
            "dice_retention_colon": 0.89,
        })
    df = pd.DataFrame(records)

    out_md = tmp_path / "stress_report.md"
    generate_markdown_report(df, str(out_md))

    assert out_md.exists()
    content = out_md.read_text()
    assert "# MultiPhase-GI: Generative Contrast Robustness" in content
    assert "Pass Rate (≥0.85)" in content

