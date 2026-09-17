"""Unified evaluation: pooled + subtype- + size-stratified metrics,
calibration, and bootstrap CIs -- implements manuscript Section IV-E
(stratification) and Section VI-E (calibration), and produces rows in
exactly the structure manuscript Tables II, III, and VI describe.

Every condition (all five baseline families and WICL) is scored by the
SAME function here, on the SAME held-out data for a given site, so that
no scoring-methodology difference can confound a cross-condition
comparison -- mirroring the shared-dataset-class discipline of
ich_gen.datasets.common.MultiSiteICHDataset.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ich_gen.datasets.common import LABEL_COLUMNS, SUBTYPES, assign_size_tercile
from ich_gen.stats import (expected_calibration_error, patient_level_bootstrap_ci)

SIZE_TERCILE_NAMES = ("small", "medium", "large")


@dataclasses.dataclass
class EvalInputs:
    """Everything needed to score one (condition, site) pair."""
    y_true: np.ndarray            # (N, 6), order = LABEL_COLUMNS
    y_prob: np.ndarray            # (N, 6), sigmoid outputs
    patient_ids: np.ndarray       # (N,)
    lesion_size_mm3: np.ndarray   # (N,), NaN where unknown/negative
    condition: str
    site: str


def _sensitivity(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    positives = y_true_bin.sum()
    if positives == 0:
        return float("nan")
    tp = ((y_true_bin == 1) & (y_pred_bin == 1)).sum()
    return float(tp / positives)


def _specificity(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    negatives = (y_true_bin == 0).sum()
    if negatives == 0:
        return float("nan")
    tn = ((y_true_bin == 0) & (y_pred_bin == 0)).sum()
    return float(tn / negatives)


def _safe_auroc(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_bin, y_score))


def evaluate_pooled_and_stratified(inputs: EvalInputs,
                                    size_thresholds: tuple[float, float] | None,
                                    threshold: float = 0.5,
                                    n_bootstrap: int = 1000,
                                    seed: int = 0) -> dict:
    """Produces one result record per (condition, site): pooled metrics,
    per-subtype metrics, per-size-tercile metrics (Table III columns),
    calibration (Table VI), all with patient-level bootstrap 95% CIs
    (Section VI-G) where a scalar summary is reported.
    """
    y_true, y_prob = inputs.y_true, inputs.y_prob
    y_pred = (y_prob >= threshold).astype(int)
    any_idx = LABEL_COLUMNS.index("any")

    record: dict = {"condition": inputs.condition, "site": inputs.site,
                     "n_samples": int(len(y_true)),
                     "n_patients": int(len(np.unique(inputs.patient_ids)))}

    # --- pooled ("any ICH") metrics ---
    record["pooled_auroc"] = _safe_auroc(y_true[:, any_idx], y_prob[:, any_idx])
    acc_point, acc_lo, acc_hi = patient_level_bootstrap_ci(
        inputs.patient_ids, (y_pred[:, any_idx] == y_true[:, any_idx]).astype(float),
        np.mean, n_bootstrap, seed=seed)
    record["pooled_accuracy"] = acc_point
    record["pooled_accuracy_ci"] = (acc_lo, acc_hi)

    # Sensitivity/specificity are ratio statistics (TP/positives etc.), not
    # simple means, so they cannot be bootstrapped via
    # patient_level_bootstrap_ci's mean-of-per-sample-values interface used
    # for accuracy above; _bootstrap_ratio_ci recomputes the ratio directly
    # on each patient resample instead.
    record["pooled_sensitivity"], record["pooled_sensitivity_ci"] = \
        _bootstrap_ratio_ci(inputs.patient_ids, y_true[:, any_idx], y_pred[:, any_idx],
                             _sensitivity, n_bootstrap, seed)
    record["pooled_specificity"], record["pooled_specificity_ci"] = \
        _bootstrap_ratio_ci(inputs.patient_ids, y_true[:, any_idx], y_pred[:, any_idx],
                             _specificity, n_bootstrap, seed)

    # --- per-subtype metrics (Table III) ---
    for subtype in SUBTYPES:
        idx = LABEL_COLUMNS.index(subtype)
        record[f"auroc_{subtype}"] = _safe_auroc(y_true[:, idx], y_prob[:, idx])
        sens, sens_ci = _bootstrap_ratio_ci(
            inputs.patient_ids, y_true[:, idx], y_pred[:, idx], _sensitivity,
            n_bootstrap, seed)
        spec, spec_ci = _bootstrap_ratio_ci(
            inputs.patient_ids, y_true[:, idx], y_pred[:, idx], _specificity,
            n_bootstrap, seed)
        record[f"sensitivity_{subtype}"] = sens
        record[f"sensitivity_{subtype}_ci"] = sens_ci
        record[f"specificity_{subtype}"] = spec
        record[f"specificity_{subtype}_ci"] = spec_ci

    # --- per-size-tercile metrics (Table III, tests H1) ---
    has_size = ~np.isnan(inputs.lesion_size_mm3)
    if has_size.any() and size_thresholds is not None:
        tercile, _ = assign_size_tercile(inputs.lesion_size_mm3, size_thresholds)
        for t_idx, t_name in enumerate(SIZE_TERCILE_NAMES):
            mask = (tercile == t_idx) & has_size
            if mask.sum() == 0:
                record[f"sensitivity_size_{t_name}"] = float("nan")
                continue
            # size stratification is defined over ICH-positive ("any")
            # samples only, per manuscript Section IV-E
            pos_mask = mask & (y_true[:, any_idx] == 1)
            if pos_mask.sum() == 0:
                record[f"sensitivity_size_{t_name}"] = float("nan")
                continue
            sens = _sensitivity(y_true[pos_mask, any_idx], y_pred[pos_mask, any_idx])
            record[f"sensitivity_size_{t_name}"] = sens
    else:
        for t_name in SIZE_TERCILE_NAMES:
            record[f"sensitivity_size_{t_name}"] = float("nan")
        if not has_size.any():
            record["size_stratification_note"] = (
                "no lesion_size_mm3 available for this site (expected for "
                "RSNA/CQ500/PhysioNet-ICH per Section IV-E; BHSD-derived "
                "data is the primary size-stratification source)")

    # --- calibration (Table VI, tests H3) ---
    cal = expected_calibration_error(y_true[:, any_idx], y_prob[:, any_idx])
    record["ece"] = cal.ece
    record["ece_bin_accuracy"] = cal.bin_accuracy.tolist()
    record["ece_bin_confidence"] = cal.bin_confidence.tolist()
    record["ece_bin_count"] = cal.bin_count.tolist()

    return record


def _bootstrap_ratio_ci(patient_ids: np.ndarray, y_true_bin: np.ndarray,
                         y_pred_bin: np.ndarray, ratio_fn, n_bootstrap: int,
                         seed: int) -> tuple[float, tuple[float, float]]:
    """Patient-level bootstrap CI for a ratio statistic (sensitivity or
    specificity) that cannot be expressed as a simple mean of per-sample
    values. Resamples patients with replacement and recomputes the ratio
    on each resample's pooled samples."""
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    point = ratio_fn(y_true_bin, y_pred_bin)

    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([np.where(patient_ids == p)[0] for p in sampled])
        boot[b] = ratio_fn(y_true_bin[idx], y_pred_bin[idx])
    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        return point, (float("nan"), float("nan"))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, (float(lo), float(hi))


def results_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Flatten a list of evaluate_pooled_and_stratified() records into a
    DataFrame suitable for direct export to the manuscript's Tables
    II/III/VI (one row per condition x site)."""
    return pd.DataFrame(records)
