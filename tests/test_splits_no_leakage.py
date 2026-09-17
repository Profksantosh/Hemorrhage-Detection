"""Tests for the leakage-control machinery manuscript Section IV-F commits
to running before any result is reported. These are the most important
tests in this repository: the paper's entire methodological claim (H1)
depends on the multi-site benchmark actually being leakage-free, and
Section I-B(1) specifically calls out a published paper [12] whose
claimed "patient-level split" was not actually enforced by its described
code. This file exists to make sure that mistake cannot happen here
undetected.
"""
import dataclasses

import numpy as np
import pytest

from ich_gen.datasets.splits import (
    patient_grouped_kfold, verify_no_patient_overlap,
    verify_disjoint_from_external,
)


def _make_multi_slice_patients(n_patients=200, min_slices=5, max_slices=30, seed=0):
    rng = np.random.RandomState(seed)
    ids = []
    for i in range(n_patients):
        n = rng.randint(min_slices, max_slices)
        ids += [f"P{i:04d}"] * n
    return np.array(ids)


def test_no_patient_appears_in_two_folds():
    patient_ids = _make_multi_slice_patients()
    folds = patient_grouped_kfold(patient_ids, n_splits=5, seed=0)
    verify_no_patient_overlap(folds)  # must not raise


def test_folds_partition_all_patients_exactly_once():
    patient_ids = _make_multi_slice_patients(n_patients=100)
    folds = patient_grouped_kfold(patient_ids, n_splits=5, seed=0)
    all_val_patients = set()
    for f in folds:
        assert not (all_val_patients & f.val_patients), "val sets must be disjoint"
        all_val_patients |= f.val_patients
    assert all_val_patients == set(patient_ids.tolist())


def test_verify_no_patient_overlap_catches_injected_leakage():
    patient_ids = _make_multi_slice_patients(n_patients=50)
    folds = patient_grouped_kfold(patient_ids, n_splits=5, seed=0)
    leaked_patient = next(iter(folds[0].train_patients))
    corrupted = dataclasses.replace(
        folds[0], val_patients=folds[0].val_patients | {leaked_patient})
    with pytest.raises(AssertionError):
        verify_no_patient_overlap([corrupted])


def test_verify_no_patient_overlap_catches_cross_fold_leakage():
    patient_ids = _make_multi_slice_patients(n_patients=50)
    folds = patient_grouped_kfold(patient_ids, n_splits=5, seed=0)
    shared_patient = next(iter(folds[0].val_patients))
    corrupted_fold1 = dataclasses.replace(
        folds[1], val_patients=folds[1].val_patients | {shared_patient})
    with pytest.raises(AssertionError):
        verify_no_patient_overlap([folds[0], corrupted_fold1])


def test_raises_if_fewer_unique_patients_than_folds():
    patient_ids = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError):
        patient_grouped_kfold(patient_ids, n_splits=5)


def test_verify_disjoint_from_external_passes_for_disjoint_ids():
    train_ids = [f"RSNA_{i}" for i in range(10)]
    external_ids = [f"CQ500_{i}" for i in range(5)]
    verify_disjoint_from_external(train_ids, external_ids, "CQ500")  # must not raise


def test_verify_disjoint_from_external_catches_id_collision():
    train_ids = [f"RSNA_{i}" for i in range(10)]
    external_ids = ["RSNA_3", "CQ500_1"]  # deliberately colliding ID
    with pytest.raises(AssertionError):
        verify_disjoint_from_external(train_ids, external_ids, "CQ500")


def test_reproducibility_same_seed_same_split():
    patient_ids = _make_multi_slice_patients(n_patients=80, seed=1)
    folds_a = patient_grouped_kfold(patient_ids, n_splits=5, seed=7)
    folds_b = patient_grouped_kfold(patient_ids, n_splits=5, seed=7)
    for fa, fb in zip(folds_a, folds_b):
        assert fa.val_patients == fb.val_patients
