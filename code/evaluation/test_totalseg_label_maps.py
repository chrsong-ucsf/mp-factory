import numpy as np
import pytest
import sys

import code.evaluation.evaluate_multi_model_ensemble as ensemble
from code.evaluation.evaluate_multi_model_ensemble import normalize_prediction_mask
from code.evaluation.evaluate_jhu_radiologist_gpu import remap_prediction_labels


def test_ensemble_remaps_default_totalseg_and_ignores_non_gi_ids():
    source = np.array([0, 5, 6, 18, 19, 20, 55, 56, 57, 58], dtype=np.uint8)
    expected = np.array([0, 0, 1, 3, 2, 4, 0, 0, 0, 0], dtype=np.uint8)
    np.testing.assert_array_equal(normalize_prediction_mask(source, 'TotalSeg'), expected)


def test_ensemble_preserves_native_gi_mask():
    source = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    np.testing.assert_array_equal(normalize_prediction_mask(source, 'MedNeXt'), source)


def test_ensemble_does_not_infer_schema_from_low_full_body_values():
    source = np.array([0, 1, 2, 3, 4], dtype=np.uint8)
    np.testing.assert_array_equal(
        normalize_prediction_mask(source, 'TotalSegmentator'),
        np.zeros_like(source),
    )


def test_ensemble_recognizes_totalseg_name_case_insensitively():
    np.testing.assert_array_equal(
        normalize_prediction_mask(np.array([6, 18, 19, 20]), 'totalsegmentator'),
        np.array([1, 3, 2, 4]),
    )


def test_ensemble_rejects_unknown_full_body_schema():
    with pytest.raises(ValueError, match='model name'):
        normalize_prediction_mask(np.array([18]), 'model_1')


def test_ensemble_requires_explicit_model_names(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['ensemble', '--pred_dirs', '/unused'])
    with pytest.raises(SystemExit) as exc:
        ensemble.main()
    assert exc.value.code == 2
    assert '--model_names' in capsys.readouterr().err


def test_jhu_remaps_default_totalseg_to_radiologist_labels():
    source = np.zeros((1, 1, 10), dtype=np.uint8)
    source[0, 0] = [0, 5, 6, 18, 19, 20, 55, 56, 57, 58]
    expected = np.zeros_like(source)
    expected[0, 0] = [0, 0, 2, 4, 3, 6, 0, 0, 0, 0]
    np.testing.assert_array_equal(remap_prediction_labels(source, 'TotalSegmentator'), expected)


def test_jhu_preserves_existing_gi_model_remap():
    source = np.zeros((1, 1, 4), dtype=np.uint8)
    source[0, 0] = [1, 2, 3, 4]
    expected = np.zeros_like(source)
    expected[0, 0] = [2, 3, 5, 6]
    np.testing.assert_array_equal(remap_prediction_labels(source, 'MedNeXt'), expected)
