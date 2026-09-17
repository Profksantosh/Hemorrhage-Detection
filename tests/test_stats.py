"""Validate ich_gen.stats against known ground truth (manuscript Section
VI-G). DeLong's test is checked against scikit-learn's AUROC and against
the degenerate identical-predictors case; calibration is checked against
constructed well-calibrated and badly-miscalibrated synthetic predictors.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from ich_gen.stats import (
    delong_paired_auc_test, patient_level_bootstrap_ci, paired_bootstrap_test,
    bonferroni_correct, expected_calibration_error, is_clinically_relevant,
)


def _synthetic_scores(seed=42, n=500):
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, 2, n)
    score_good = y_true * 0.6 + rng.rand(n) * 0.4
    score_random = rng.rand(n)
    return y_true, score_good, score_random


def test_delong_matches_sklearn_auc():
    y_true, score_a, score_b = _synthetic_scores()
    result = delong_paired_auc_test(y_true, score_a, score_b)
    assert abs(result.auc_a - roc_auc_score(y_true, score_a)) < 1e-9
    assert abs(result.auc_b - roc_auc_score(y_true, score_b)) < 1e-9


def test_delong_detects_a_clearly_better_classifier():
    y_true, score_a, score_b = _synthetic_scores()
    result = delong_paired_auc_test(y_true, score_a, score_b)
    assert result.auc_a > result.auc_b
    assert result.p_value < 0.01


def test_delong_identical_predictors_gives_zero_diff_and_p_one():
    y_true, score_a, _ = _synthetic_scores()
    result = delong_paired_auc_test(y_true, score_a, score_a)
    assert abs(result.auc_diff) < 1e-12
    assert result.p_value == 1.0


def test_delong_rejects_non_binary_labels():
    import pytest
    y_true = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        delong_paired_auc_test(y_true, np.random.rand(3), np.random.rand(3))


def test_patient_level_bootstrap_ci_contains_point_estimate():
    rng = np.random.RandomState(1)
    patient_ids = np.repeat(np.arange(50), 10)
    values = (rng.rand(500) > 0.3).astype(float)
    point, lo, hi = patient_level_bootstrap_ci(patient_ids, values, np.mean,
                                                n_bootstrap=200, seed=1)
    assert lo <= point <= hi


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.RandomState(2)
    patient_ids = np.repeat(np.arange(50), 10)
    values_a = (rng.rand(500) > 0.1).astype(float)
    values_b = (rng.rand(500) > 0.6).astype(float)
    p = paired_bootstrap_test(patient_ids, values_a, values_b, np.mean,
                               n_bootstrap=200, seed=2)
    assert p < 0.05


def test_paired_bootstrap_no_difference_gives_large_p():
    rng = np.random.RandomState(3)
    patient_ids = np.repeat(np.arange(50), 10)
    values_a = (rng.rand(500) > 0.5).astype(float)
    values_b = (rng.rand(500) > 0.5).astype(float)
    p = paired_bootstrap_test(patient_ids, values_a, values_b, np.mean,
                               n_bootstrap=200, seed=3)
    assert p > 0.05


def test_bonferroni_correction():
    raw = np.array([0.01, 0.02, 0.5])
    corrected = bonferroni_correct(raw)
    assert np.allclose(corrected, [0.03, 0.06, 1.0])


def test_ece_low_for_well_calibrated_predictions():
    rng = np.random.RandomState(4)
    probs = rng.rand(2000)
    labels = (rng.rand(2000) < probs).astype(float)
    result = expected_calibration_error(labels, probs)
    assert result.ece < 0.1


def test_ece_high_for_overconfident_predictions():
    rng = np.random.RandomState(5)
    probs = np.full(1000, 0.99)
    labels = (rng.rand(1000) < 0.5).astype(float)
    result = expected_calibration_error(labels, probs)
    assert result.ece > 0.4


def test_clinically_relevant_threshold():
    assert not is_clinically_relevant(0.03)
    assert is_clinically_relevant(0.08)
    assert is_clinically_relevant(-0.08)  # magnitude, not signed
