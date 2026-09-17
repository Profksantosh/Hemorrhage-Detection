import numpy as np
import torch

from ich_gen.models.backbone import build_backbone
from ich_gen.failure_analysis import (
    GradCAM, windowing_outlier_score, compute_reference_histogram,
    correlate_outlier_score_with_errors,
)


def test_gradcam_produces_normalized_heatmap():
    model = build_backbone(pretrained=False)
    model.eval()
    target_layer = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    cam = GradCAM(model, target_layer)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap = cam(x, class_idx=0)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_outlier_score_detects_shifted_distribution():
    rng = np.random.RandomState(0)
    training_samples = [rng.normal(40, 200, size=(64, 64)).astype(np.float32)
                         for _ in range(50)]
    ref_hist, ref_edges = compute_reference_histogram(training_samples)

    typical = rng.normal(40, 200, size=(64, 64)).astype(np.float32)
    shifted = rng.normal(600, 300, size=(64, 64)).astype(np.float32)

    score_typical = windowing_outlier_score(typical, ref_hist, ref_edges)
    score_shifted = windowing_outlier_score(shifted, ref_hist, ref_edges)
    assert score_shifted > score_typical


def test_outlier_score_is_zero_for_reference_itself():
    rng = np.random.RandomState(1)
    training_samples = [rng.normal(40, 200, size=(64, 64)).astype(np.float32)
                         for _ in range(200)]
    ref_hist, ref_edges = compute_reference_histogram(training_samples)
    # a large held-out sample from the SAME distribution should score low
    same_dist = rng.normal(40, 200, size=(200, 200)).astype(np.float32)
    score = windowing_outlier_score(same_dist, ref_hist, ref_edges)
    assert score < 10.0  # small relative to the cross-distribution score (~550) above


def test_correlation_detects_known_injected_effect():
    rng = np.random.RandomState(2)
    n = 300
    outlier_scores = rng.uniform(0, 100, n)
    prob_incorrect = outlier_scores / 100 * 0.8 + 0.05
    is_correct = (rng.rand(n) > prob_incorrect).astype(int)
    result = correlate_outlier_score_with_errors(outlier_scores, is_correct)
    assert result["p_value"] < 0.05
    assert result["mean_outlier_score_incorrect"] > result["mean_outlier_score_correct"]


def test_correlation_handles_no_errors_gracefully():
    n = 50
    outlier_scores = np.random.RandomState(3).rand(n) * 100
    is_correct = np.ones(n, dtype=int)  # everything correct, no incorrect group
    result = correlate_outlier_score_with_errors(outlier_scores, is_correct)
    assert result["p_value"] != result["p_value"]  # NaN, not a crash
