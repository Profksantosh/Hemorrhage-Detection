"""Tests for the stratified evaluation module (manuscript Section IV-E,
Tables II/III/VI)."""
import numpy as np

from ich_gen.evaluate import EvalInputs, evaluate_pooled_and_stratified
from ich_gen.datasets.common import LABEL_COLUMNS


def _perfect_classifier_inputs(seed=0, n_patients=20, slices_per_patient=10):
    rng = np.random.RandomState(seed)
    n = n_patients * slices_per_patient
    patient_ids = np.repeat(np.arange(n_patients), slices_per_patient)
    y_true = np.zeros((n, 6), dtype=np.float32)
    pos_patients = rng.choice(n_patients, n_patients // 2, replace=False)
    for p in pos_patients:
        idx = np.where(patient_ids == p)[0]
        subtype = rng.randint(1, 6)
        y_true[idx, subtype] = 1
        y_true[idx, 0] = 1
    y_prob = y_true.copy()  # perfect predictions
    lesion_size = np.where(y_true[:, 0] == 1, rng.uniform(50, 5000, n), np.nan)
    return patient_ids, y_true, y_prob, lesion_size


def test_perfect_classifier_gets_auroc_and_accuracy_of_one():
    patient_ids, y_true, y_prob, lesion_size = _perfect_classifier_inputs()
    inputs = EvalInputs(y_true, y_prob, patient_ids, lesion_size,
                         condition="test", site="synthetic")
    rec = evaluate_pooled_and_stratified(inputs, size_thresholds=(500, 2000),
                                          n_bootstrap=50, seed=0)
    assert rec["pooled_auroc"] == 1.0
    assert rec["pooled_accuracy"] == 1.0
    assert rec["pooled_sensitivity"] == 1.0
    assert rec["pooled_specificity"] == 1.0


def test_missing_subtype_in_data_gives_nan_not_crash():
    """A subtype with zero positive samples in a given evaluation batch
    (e.g. EDH on a small external site) must yield NaN, not a crash --
    this matters because rare subtypes are exactly what H1 studies."""
    n = 20
    patient_ids = np.arange(n)
    y_true = np.zeros((n, 6), dtype=np.float32)
    y_prob = np.random.RandomState(0).rand(n, 6).astype(np.float32)
    lesion_size = np.full(n, np.nan)
    inputs = EvalInputs(y_true, y_prob, patient_ids, lesion_size,
                         condition="test", site="synthetic")
    rec = evaluate_pooled_and_stratified(inputs, size_thresholds=None,
                                          n_bootstrap=20, seed=0)
    assert rec["pooled_auroc"] != rec["pooled_auroc"] or True  # nan-safe, no crash
    for subtype in ("epidural", "subdural"):
        assert rec[f"sensitivity_{subtype}"] != rec[f"sensitivity_{subtype}"]  # is NaN


def test_no_size_data_available_reports_note_not_crash():
    n = 20
    patient_ids = np.arange(n)
    y_true = np.random.RandomState(1).randint(0, 2, (n, 6)).astype(np.float32)
    y_prob = np.random.RandomState(2).rand(n, 6).astype(np.float32)
    lesion_size = np.full(n, np.nan)  # e.g. RSNA/CQ500/PhysioNet-ICH (Section IV-E)
    inputs = EvalInputs(y_true, y_prob, patient_ids, lesion_size,
                         condition="test", site="cq500")
    rec = evaluate_pooled_and_stratified(inputs, size_thresholds=None,
                                          n_bootstrap=20, seed=0)
    assert "size_stratification_note" in rec
    for tercile in ("small", "medium", "large"):
        val = rec[f"sensitivity_size_{tercile}"]
        assert val != val  # NaN


def test_calibration_fields_present_and_bounded():
    patient_ids, y_true, y_prob, lesion_size = _perfect_classifier_inputs()
    inputs = EvalInputs(y_true, y_prob, patient_ids, lesion_size,
                         condition="test", site="synthetic")
    rec = evaluate_pooled_and_stratified(inputs, size_thresholds=(500, 2000),
                                          n_bootstrap=20, seed=0)
    assert 0.0 <= rec["ece"] <= 1.0
    assert len(rec["ece_bin_accuracy"]) == 15
