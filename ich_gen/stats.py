"""Statistical analysis plan (manuscript Section VI-G): patient-level
bootstrap confidence intervals, DeLong's test for paired AUROC
comparisons, paired bootstrap significance tests, Bonferroni correction,
and Expected Calibration Error (manuscript Section VI-E, testing H3).

The DeLong implementation below is the standard "fast DeLong" algorithm
(DeLong, DeLong & Clarke-Pearson, 1988; efficient formulation per Sun &
Xu, 2014), a well-established, widely reimplemented method -- NOT a
manuscript contribution. It is validated in tests/test_stats.py against
scikit-learn's roc_auc_score (the AUCs the two methods compute must
agree) and against known degenerate cases (identical predictors -> AUC
difference exactly 0).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# DeLong's test for two correlated (paired) ROC AUCs
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def _fast_delong(predictions_sorted: np.ndarray, m: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """predictions_sorted: (k_classifiers, n_samples), columns sorted so
    that the m positive-label samples come first. Returns (aucs, cov)."""
    n = predictions_sorted.shape[1] - m
    k = predictions_sorted.shape[0]
    positive = predictions_sorted[:, :m]
    negative = predictions_sorted[:, m:]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


@dataclasses.dataclass
class DeLongResult:
    auc_a: float
    auc_b: float
    auc_diff: float
    p_value: float
    z_statistic: float


def delong_paired_auc_test(y_true: np.ndarray,
                            y_score_a: np.ndarray,
                            y_score_b: np.ndarray) -> DeLongResult:
    """Two-sided DeLong test for the difference between two AUCs computed
    on the SAME (paired) set of samples/labels -- e.g. baseline-2 vs.
    WICL on the same external test set (manuscript Section VI-G)."""
    y_true = np.asarray(y_true)
    if set(np.unique(y_true).tolist()) - {0, 1, 0.0, 1.0}:
        raise ValueError("y_true must be binary (0/1)")
    order = np.argsort(-y_true)
    m = int(y_true.sum())
    if m == 0 or m == len(y_true):
        raise ValueError("DeLong test requires at least one positive and "
                          "one negative sample")
    stacked = np.vstack([np.asarray(y_score_a), np.asarray(y_score_b)])[:, order]
    aucs, cov = _fast_delong(stacked, m)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        # degenerate case (e.g. identical predictors): no measurable
        # variance in the difference, so treat as no significant difference
        z = 0.0
        p = 1.0
    else:
        z = diff / np.sqrt(var)
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    return DeLongResult(float(aucs[0]), float(aucs[1]), diff, float(p), float(z))


# ---------------------------------------------------------------------------
# Patient-level bootstrap (manuscript Section VI-G)
# ---------------------------------------------------------------------------

def patient_level_bootstrap_ci(patient_ids: np.ndarray,
                                per_sample_values: np.ndarray,
                                statistic_fn,
                                n_bootstrap: int = 1000,
                                ci: float = 0.95,
                                seed: int = 0) -> tuple[float, float, float]:
    """Resample PATIENTS (not individual slices) with replacement, so the
    non-independence structure induced by multiple slices per patient is
    respected at evaluation time -- matching manuscript Section IV-F's
    concern applied to bootstrap CIs specifically (Section VI-G: "resampling
    patients rather than slices").

    `statistic_fn(values_for_resampled_slices)` should compute the metric
    of interest (e.g. sensitivity) given the per-sample values belonging
    to the resampled patients (with repeats).

    Returns (point_estimate, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    point = statistic_fn(per_sample_values)

    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients),
                                       replace=True)
        idx = np.concatenate([np.where(patient_ids == p)[0] for p in sampled_patients])
        boot_stats[b] = statistic_fn(per_sample_values[idx])

    alpha = 1 - ci
    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_bootstrap_test(patient_ids: np.ndarray,
                           values_a: np.ndarray,
                           values_b: np.ndarray,
                           statistic_fn,
                           n_bootstrap: int = 1000,
                           seed: int = 0) -> float:
    """Two-sided paired-bootstrap p-value for statistic_fn(values_a) -
    statistic_fn(values_b), resampling patients (see
    patient_level_bootstrap_ci), following the percentile-bootstrap
    paired-samples approach used by Fang et al. [16] in this literature.
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    observed_diff = statistic_fn(values_a) - statistic_fn(values_b)

    diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients),
                                       replace=True)
        idx = np.concatenate([np.where(patient_ids == p)[0] for p in sampled_patients])
        diffs[b] = statistic_fn(values_a[idx]) - statistic_fn(values_b[idx])

    # Standard percentile-bootstrap p-value: fraction of bootstrap
    # replicates on the opposite side of zero from the observed effect.
    if observed_diff >= 0:
        p = 2 * min((diffs <= 0).mean(), 0.5)
    else:
        p = 2 * min((diffs >= 0).mean(), 0.5)
    return float(min(p, 1.0))


# ---------------------------------------------------------------------------
# Multiple-comparison correction (manuscript Section VI-C/VI-G)
# ---------------------------------------------------------------------------

def bonferroni_correct(p_values: np.ndarray) -> np.ndarray:
    """manuscript Section VI-C: "Bonferroni correction applied across the
    five subtypes, three size terciles, and three external sites (45
    comparisons per baseline pair)"."""
    n = len(p_values)
    return np.clip(np.asarray(p_values) * n, 0, 1)


# ---------------------------------------------------------------------------
# Calibration: Expected Calibration Error (manuscript Section VI-E, H3)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CalibrationResult:
    ece: float
    bin_accuracy: np.ndarray
    bin_confidence: np.ndarray
    bin_count: np.ndarray


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                                n_bins: int = 15) -> CalibrationResult:
    """15-bin ECE, per manuscript Section VI-E."""
    y_true = np.asarray(y_true).astype(np.float64)
    y_prob = np.asarray(y_prob).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)

    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        bin_count[b] = mask.sum()
        if bin_count[b] > 0:
            bin_acc[b] = y_true[mask].mean()
            bin_conf[b] = y_prob[mask].mean()

    n = len(y_true)
    ece = float(np.sum(bin_count / n * np.abs(bin_acc - bin_conf)))
    return CalibrationResult(ece, bin_acc, bin_conf, bin_count)


# ---------------------------------------------------------------------------
# Minimum clinically relevant effect size (manuscript Section VI-G)
# ---------------------------------------------------------------------------

MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA = 0.05  # 5 percentage points,
# an order-of-magnitude anchor against the ~0.45 EDH recall gap documented
# in manuscript refs [8], [11] -- see manuscript Section VI-G.


def is_clinically_relevant(effect_size: float,
                            threshold: float = MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
                            ) -> bool:
    return abs(effect_size) >= threshold
