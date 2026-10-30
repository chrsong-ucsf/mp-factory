import csv

import nibabel as nib
import numpy as np
import pytest
import torch

from code.project10.datasets.multiphase_paired_dataset import MultiphasePairedDataset


def _volume(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), np.eye(4)), path)


def _dataset(tmp_path, mask_name="CASE_gi_mask.nii.gz", mask=None, make_ncct=True, make_target=True,
             totalseg_task="gi"):
    base = tmp_path
    if make_ncct:
        _volume(base / "CancerVerse" / "CancerVerse" / "CASE" / "ct.nii.gz", np.ones((4, 4, 4)))
    if make_target:
        _volume(base / "registered.nii.gz", np.ones((4, 4, 4)))
    if mask is not None:
        _volume(base / "masks" / mask_name, mask)
    with (base / "registration.csv").open("w", newline="") as f:
        csv.writer(f).writerow(["pair", "patient", "CASE", "TARGET", "VENOUS", "OK", "1", "", "registered.nii.gz"])
    return MultiphasePairedDataset("registration.csv", str(base), "masks", (4, 4, 4), 1.0,
                                   totalseg_task=totalseg_task)


@pytest.mark.parametrize("name", ["CASE_mask.nii.gz", "CASE_gi_mask.nii.gz", "CASE.nii.gz"])
def test_supported_mask_names_are_loaded(tmp_path, name):
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    is_full_body = name.endswith("_mask.nii.gz") and not name.endswith("_gi_mask.nii.gz")
    mask[1, 1, 1] = 19 if is_full_body else 2
    sample = _dataset(tmp_path, name, mask, totalseg_task="total" if is_full_body else "gi")[0]
    assert sample["mask"][0, 1, 1, 1] == 2


def test_full_body_totalseg_ids_map_to_gi_labels(tmp_path):
    mask = np.full((4, 4, 4), 99, dtype=np.uint8)
    mask[0, 0, 0], mask[0, 0, 1], mask[0, 0, 2], mask[0, 0, 3] = 6, 19, 18, 20
    sample = _dataset(tmp_path, "CASE_mask.nii.gz", mask, totalseg_task="total")[0]
    assert [sample["mask"][0, 0, 0, i].item() for i in range(4)] == [1, 2, 3, 4]
    assert set(torch.unique(sample["mask"]).tolist()) <= {0, 1, 2, 3, 4}


@pytest.mark.parametrize("mask", [None, np.full((4, 4, 4), 99, dtype=np.uint8), np.zeros((4, 4, 4), dtype=np.uint8)])
def test_missing_or_unusable_mask_raises(tmp_path, mask):
    with pytest.raises((FileNotFoundError, ValueError), match="CASE"):
        _dataset(tmp_path, mask=mask)[0]


@pytest.mark.parametrize("make_ncct,make_target", [(False, True), (True, False)])
def test_missing_ct_raises(tmp_path, make_ncct, make_target):
    mask = np.ones((4, 4, 4), dtype=np.uint8)
    with pytest.raises(FileNotFoundError, match="CASE"):
        _dataset(tmp_path, mask=mask, make_ncct=make_ncct, make_target=make_target)[0]


def test_malformed_ok_row_raises(tmp_path):
    with (tmp_path / "registration.csv").open("w", newline="") as f:
        csv.writer(f).writerow(["pair", "patient", "", "TARGET", "VENOUS", "OK", "1", "", ""])
    with pytest.raises(ValueError, match="subject_p0"):
        MultiphasePairedDataset("registration.csv", str(tmp_path), "masks")


def test_negative_full_body_label_is_rejected(tmp_path):
    mask = np.zeros((4, 4, 4), dtype=np.int16)
    mask[0, 0, 0], mask[0, 0, 1] = 19, -1
    with pytest.raises(ValueError, match="CASE"):
        _dataset(tmp_path, "CASE_mask.nii.gz", mask, totalseg_task="total")[0]


def test_full_body_mask_requires_version_provenance(tmp_path):
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0, 0, 0] = 18
    with pytest.raises(ValueError, match="totalseg_task"):
        _dataset(tmp_path, "CASE_mask.nii.gz", mask, totalseg_task=None)[0]


def test_v1_mask_uses_v1_organ_ids(tmp_path):
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0, 0, 0], mask[0, 0, 1], mask[0, 0, 2], mask[0, 0, 3] = 6, 56, 55, 57
    sample = _dataset(tmp_path, "CASE_mask.nii.gz", mask, totalseg_task="total_v1")[0]
    assert [sample["mask"][0, 0, 0, i].item() for i in range(4)] == [1, 2, 3, 4]


def test_low_full_body_ids_do_not_pass_as_gi_labels(tmp_path):
    mask = np.zeros((4, 4, 4), dtype=np.uint8)
    mask[0, 0, 0] = 2  # right kidney in TotalSegmentator total
    with pytest.raises(ValueError, match="CASE"):
        _dataset(tmp_path, "CASE_mask.nii.gz", mask, totalseg_task="total")[0]


def test_corrupt_mask_does_not_return_zero_sample(tmp_path):
    dataset = _dataset(tmp_path, mask=np.ones((4, 4, 4), dtype=np.uint8))
    (tmp_path / "masks" / "CASE_gi_mask.nii.gz").write_bytes(b"not a nifti")
    with pytest.raises(RuntimeError, match="CASE"):
        dataset[0]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("ct_name", ["registered.nii.gz", "CancerVerse/CancerVerse/CASE/ct.nii.gz"])
def test_nonfinite_ct_is_rejected_before_intensity_scaling(tmp_path, value, ct_name):
    dataset = _dataset(tmp_path, mask=np.ones((4, 4, 4), dtype=np.uint8))
    ct = np.ones((4, 4, 4), dtype=np.float32)
    ct[0, 0, 0] = value
    _volume(tmp_path / ct_name, ct)
    with pytest.raises(RuntimeError, match="CASE"):
        dataset[0]


def test_ncct_falls_back_to_cancerverse_dbox(tmp_path):
    dataset = _dataset(tmp_path, mask=np.ones((4, 4, 4), dtype=np.uint8), make_ncct=False)
    _volume(tmp_path / "CancerVerse_dbox" / "CASE" / "ct.nii.gz", np.ones((4, 4, 4)))
    assert dataset[0]["subject_id"] == "CASE"
