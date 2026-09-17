"""Failure-mode and mechanism-attribution analysis (manuscript Section
VI-H). Two components:

  1. Grad-CAM, for the qualitative figure panels (Fig. 2b of the pending
     Results section).
  2. A per-image "windowing-outlier score" (HU-histogram deviation from
     the training-domain's typical HU statistics) and its correlation
     with misclassification, which is the manuscript's specific test of
     WHETHER an observed WICL benefit operates through the hypothesized
     causal channel (windowing/HU-calibration invariance) rather than
     through some other, unidentified correlate of the intervention
     (Section VI-H, Section VIII).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats


class GradCAM:
    """Standard hook-based Grad-CAM (Selvaraju et al., 2017) against a
    given convolutional layer. Not a manuscript contribution -- included
    because Grad-CAM is used qualitatively throughout the reviewed
    literature (manuscript refs [8], [18], [20]) and quantitatively,
    against expert segmentation, by [10]; this manuscript uses it only
    qualitatively (Fig. 2b) plus as an input to the windowing-outlier
    correlation analysis below, not for a Dice-validated localization
    claim, which is out of this manuscript's classification-only scope
    (Section III-A).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """x: (1, 3, H, W). Returns a (H, W) heatmap in [0, 1]."""
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1,1,h,w)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return np.zeros_like(cam)
        return (cam - cam_min) / (cam_max - cam_min)


def windowing_outlier_score(hu: np.ndarray,
                              reference_hist: np.ndarray,
                              reference_bin_edges: np.ndarray) -> float:
    """Wasserstein (earth-mover's) distance between a single image's HU
    histogram and the training-domain's reference HU histogram, as a
    model-independent measure of how "out of distribution", from a pure
    windowing/HU-statistics standpoint, a given image is -- manuscript
    Section VI-H.

    Parameters
    ----------
    hu : raw HU array for one slice.
    reference_hist, reference_bin_edges : precomputed from the pooled
        RSNA TRAINING set only (compute once via
        `compute_reference_histogram`, reuse for every scored image).
    """
    hist, _ = np.histogram(hu, bins=reference_bin_edges, density=True)
    # normalize both to proper probability mass functions before comparing
    hist = hist / max(hist.sum(), 1e-12)
    ref = reference_hist / max(reference_hist.sum(), 1e-12)
    bin_centers = 0.5 * (reference_bin_edges[:-1] + reference_bin_edges[1:])
    return float(scipy_stats.wasserstein_distance(bin_centers, bin_centers,
                                                    u_weights=hist, v_weights=ref))


def compute_reference_histogram(training_hu_samples: list[np.ndarray],
                                  hu_range: tuple[float, float] = (-1000.0, 3000.0),
                                  n_bins: int = 100
                                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pooled HU histogram over a (sub)sample of the RSNA training set,
    used as the reference distribution for windowing_outlier_score."""
    bin_edges = np.linspace(hu_range[0], hu_range[1], n_bins + 1)
    total_hist = np.zeros(n_bins)
    for hu in training_hu_samples:
        h, _ = np.histogram(hu, bins=bin_edges)
        total_hist += h
    return total_hist, bin_edges


def correlate_outlier_score_with_errors(outlier_scores: np.ndarray,
                                          is_correct: np.ndarray
                                          ) -> dict:
    """manuscript Section VI-H: test whether misclassifications are
    disproportionately concentrated among high windowing-outlier-score
    cases. Uses a two-sided Mann-Whitney U test comparing the outlier-
    score distribution of correct vs. incorrect predictions (non-
    parametric, appropriate since outlier scores are not assumed
    normally distributed), plus the point-biserial correlation as an
    effect-size companion to the significance test.
    """
    correct_scores = outlier_scores[is_correct.astype(bool)]
    incorrect_scores = outlier_scores[~is_correct.astype(bool)]
    if len(correct_scores) == 0 or len(incorrect_scores) == 0:
        return {"u_statistic": float("nan"), "p_value": float("nan"),
                "point_biserial_r": float("nan"),
                "note": "one of the two groups (correct/incorrect) is empty"}

    u_stat, p_value = scipy_stats.mannwhitneyu(
        incorrect_scores, correct_scores, alternative="greater")
    r, _ = scipy_stats.pointbiserialr(1 - is_correct.astype(float), outlier_scores)

    return {"u_statistic": float(u_stat), "p_value": float(p_value),
            "point_biserial_r": float(r),
            "mean_outlier_score_correct": float(correct_scores.mean()),
            "mean_outlier_score_incorrect": float(incorrect_scores.mean())}
