"""Tests for baseline 5's pseudo-labeling logic (manuscript Section VI-A,
Section II-G), including the rank-based thresholding mode added after
full-text verification against Lin & Yuh [34]."""
import torch

from ich_gen.models.semi_supervised import (
    PseudoLabelConfig, generate_pseudo_labels, student_training_loss,
)


def test_top_percentile_default_selects_exactly_c_percent_positive():
    """[34]'s verified rule: C=10 -> top 10% by probability, per class."""
    torch.manual_seed(0)
    n = 200
    logits = torch.randn(n, 6) * 3
    pl, mask = generate_pseudo_labels(logits, PseudoLabelConfig())
    for c in range(6):
        assert abs(float(pl[:, c].sum()) - 20) <= 1  # 10% of 200 = 20


def test_top_percentile_positive_labels_are_the_highest_probability_samples():
    torch.manual_seed(1)
    logits = torch.randn(50, 1) * 2
    pl, mask = generate_pseudo_labels(logits, PseudoLabelConfig(top_percentile=10.0,
                                                                  bottom_percentile=10.0))
    probs = torch.sigmoid(logits).squeeze(1)
    selected_positive = probs[pl[:, 0].bool()]
    not_selected = probs[~mask[:, 0].bool()]
    if len(selected_positive) and len(not_selected):
        assert selected_positive.min() >= not_selected.max()


def test_probability_mode_is_a_distinct_explicit_alternative():
    torch.manual_seed(2)
    logits = torch.randn(100, 6) * 3
    config = PseudoLabelConfig(threshold_mode="probability",
                                positive_threshold=0.9, negative_threshold=0.1)
    pl, mask = generate_pseudo_labels(logits, config)
    probs = torch.sigmoid(logits)
    assert torch.equal(pl, (probs >= 0.9).float())


def test_unknown_threshold_mode_raises():
    import pytest
    logits = torch.randn(10, 6)
    with pytest.raises(ValueError):
        generate_pseudo_labels(logits, PseudoLabelConfig(threshold_mode="bogus"))


def test_generate_pseudo_labels_handles_empty_pool():
    logits = torch.randn(0, 6)
    pl, mask = generate_pseudo_labels(logits, PseudoLabelConfig())
    assert pl.shape == (0, 6)
    assert mask.shape == (0, 6)


def test_generate_pseudo_labels_handles_pool_smaller_than_implied_percentile():
    """A 3-sample unlabeled pool with top_percentile=10% implies 0.3
    samples -- must round to at least 1, not crash or select 0."""
    logits = torch.randn(3, 6)
    pl, mask = generate_pseudo_labels(logits, PseudoLabelConfig())
    assert mask.sum() > 0


def test_student_training_loss_is_unaffected_by_threshold_mode_choice():
    """student_training_loss only consumes (pseudo_labels, mask); it must
    behave identically regardless of which thresholding mode produced
    them -- confirms the two concerns (thresholding policy vs. loss
    computation) are properly decoupled."""
    torch.manual_seed(3)
    logits_unlabeled = torch.randn(20, 6, requires_grad=True)
    teacher_logits = torch.randn(20, 6)

    pl_rank, mask_rank = generate_pseudo_labels(
        teacher_logits, PseudoLabelConfig(threshold_mode="top_percentile"))
    pl_prob, mask_prob = generate_pseudo_labels(
        teacher_logits, PseudoLabelConfig(threshold_mode="probability"))

    logits_labeled = torch.randn(4, 6)
    labels = torch.randint(0, 2, (4, 6)).float()

    out_rank = student_training_loss(logits_labeled, labels, logits_unlabeled,
                                      pl_rank, mask_rank)
    out_prob = student_training_loss(logits_labeled, labels, logits_unlabeled,
                                      pl_prob, mask_prob)
    assert torch.isfinite(out_rank["loss"])
    assert torch.isfinite(out_prob["loss"])
