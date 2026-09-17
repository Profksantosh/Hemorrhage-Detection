"""Baseline 5 (manuscript Section VI-A): semi-supervised self-training,
reimplementing Lin & Yuh's teacher-student protocol [34] -- the strongest
previously validated, externally tested countermeasure for cross-
institution ICH generalization identified in this manuscript's
supplementary literature search (Section II-G). Included as a MANDATORY
comparator: omitting it would leave the paper's central generalization
claim untested against the closest directly on-topic published method.

Protocol (per manuscript Section VI-A):
  1. Train a "teacher" model on labeled RSNA training-fold data (any of
     the other conditions' training routine may serve as the teacher;
     the default here is the fixed-three-window baseline, i.e. baseline 2,
     since [34]'s own teacher is a standard supervised model).
  2. Use the teacher to generate pseudo-labels on a held-out UNLABELED
     pool (the manuscript specifies RSNA's own held-out-unlabeled portion,
     or an additional unlabeled public CT pool if RSNA alone is
     insufficient after the patient-grouped split -- see manuscript
     Section VI-A, baseline 5).
  3. Train a "student" model, WARM-STARTED from the teacher's weights
     (confirmed against [34]'s full text, not assumed -- see below), on
     the COMBINED labeled + pseudo-labeled data, with stronger input
     noise/augmentation on the pseudo-labeled portion (the "noisy" in
     Noisy Student).

FULL-TEXT VERIFICATION AGAINST [34] (completed; corrects two assumptions
made before this verification pass):
  - Pseudo-labeling in [34] is RANK-BASED, not a fixed probability
    cutoff: "a threshold of C = 10 ... only the 10% of images with the
    greatest probability of hemorrhage were considered positive" (their
    pixel-level segmentation step separately uses fixed 0.7/0.3
    probability bands, which does not directly apply to this
    classification-only reimplementation). `PseudoLabelConfig` below now
    defaults to this rank-based mode (`top_percentile=10.0`); the
    original fixed-probability mode is retained as an explicit
    alternative (`threshold_mode="probability"`) since it remains a
    legitimate design choice for an ablation, just not what [34] did.
  - [34]'s student "was initialized based on the weights of the teacher
    model" -- i.e. warm-started, not trained from a fresh random
    initialization. train.run_semi_supervised now does the same.
  - [34]'s own "unlabeled" pool is, concretely, the RSNA 2019 challenge
    dataset itself (25,000 exams) with its image-level labels present in
    the release but DELIBERATELY DISCARDED for the SSL pseudo-labeling
    step; their TEACHER, in contrast, is trained on a genuinely separate,
    single-institution cohort (457 pixel-labeled scans from one US
    institution, 2010-2017) that is NOT part of RSNA. This corrects an
    earlier, less precise characterization in this codebase (previously
    stated as "a genuinely separate, never-labeled multi-institution pool
    (RSNA + ASNR sources)") -- the "unlabeled" side was never claimed by
    [34] to be un-labeled at the source, only treated as such for their
    method. The real remaining difference between this reimplementation
    and [34] is therefore specifically on the LABELED/teacher side:
    train.run_semi_supervised's teacher is trained on a patient-disjoint
    subset of RSNA itself (Section VI-A's `--unlabeled-fraction` split),
    not on a genuinely separate institutional cohort like [34]'s -- this
    remains the one documented simplification, now stated precisely.

This module implements steps 2-3's core logic (pseudo-label generation
with confidence thresholding, and the combined-loss training step);
step 1 reuses the standard training loop in train.py with condition
"fixed_three_window".
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class PseudoLabelConfig:
    """threshold_mode="top_percentile" (the default) reimplements [34]'s
    verified method: per class, the top `top_percentile`% of the
    unlabeled pool BY PREDICTED PROBABILITY RANK are accepted as positive
    pseudo-labels, matching [34]'s reported C=10 image-level threshold.
    The bottom `bottom_percentile`% are accepted as negative pseudo-labels
    by symmetry; NOTE this symmetric negative-side rule is our own
    extension, not independently confirmed against [34]'s exact
    image-level negative-selection criterion (only their positive-side
    C=10 rule and their SEPARATE pixel-level 0.7/0.3 probability bands
    were confirmed via full-text verification) -- flagged here rather
    than presented as equally verified.

    threshold_mode="probability" is the fixed-cutoff alternative (NOT
    what [34] does), retained as an explicit, clearly-labeled ablation
    option rather than removed.

    Rank-based thresholding requires ranking over the FULL unlabeled
    pool at once, not per mini-batch -- callers must pass the complete
    concatenated teacher-logits tensor for the unlabeled set to
    generate_pseudo_labels, not one batch at a time (see
    train.run_semi_supervised, Stage 2).
    """
    threshold_mode: str = "top_percentile"  # "top_percentile" | "probability"
    top_percentile: float = 10.0    # C=10, matches [34]'s verified value
    bottom_percentile: float = 10.0  # symmetric extension, NOT independently
                                       # verified against [34] -- see docstring
    positive_threshold: float = 0.9  # used only if threshold_mode == "probability"
    negative_threshold: float = 0.1  # used only if threshold_mode == "probability"
    min_labels_per_sample: int = 0  # 0 = keep sample even if all 6 labels
                                      # fall in the ambiguous band (they
                                      # will simply contribute ~0 loss
                                      # weight via pseudo_label_mask)


@torch.no_grad()
def generate_pseudo_labels(teacher_logits: torch.Tensor,
                            config: PseudoLabelConfig | None = None
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert teacher logits on the FULL unlabeled pool into
    (pseudo_labels, mask), where mask[i, c] = 1 iff class c's pseudo-label
    for sample i is confident enough to use for training (see
    PseudoLabelConfig for the two supported threshold_mode options)."""
    config = config or PseudoLabelConfig()
    probs = torch.sigmoid(teacher_logits)  # (N, n_labels)

    if config.threshold_mode == "probability":
        pseudo_labels = (probs >= config.positive_threshold).float()
        confident = (probs >= config.positive_threshold) | (probs <= config.negative_threshold)
        return pseudo_labels, confident.float()

    if config.threshold_mode == "top_percentile":
        n = probs.shape[0]
        if n == 0:
            return probs.new_zeros(probs.shape), probs.new_zeros(probs.shape)
        k_top = max(1, min(n, int(round(n * config.top_percentile / 100.0))))
        k_bottom = max(1, min(n, int(round(n * config.bottom_percentile / 100.0))))
        pseudo_labels = torch.zeros_like(probs)
        confident = torch.zeros_like(probs)
        for c in range(probs.shape[1]):
            col = probs[:, c]
            top_idx = torch.topk(col, k_top, largest=True).indices
            bottom_idx = torch.topk(col, k_bottom, largest=False).indices
            pseudo_labels[top_idx, c] = 1.0
            confident[top_idx, c] = 1.0
            confident[bottom_idx, c] = 1.0  # pseudo_labels stays 0.0 here (negative)
        return pseudo_labels, confident

    raise ValueError(f"unknown threshold_mode {config.threshold_mode!r}, "
                      f"must be 'top_percentile' or 'probability'")


def student_training_loss(student_logits_labeled: torch.Tensor,
                           labels: torch.Tensor,
                           student_logits_unlabeled: torch.Tensor,
                           pseudo_labels: torch.Tensor,
                           pseudo_mask: torch.Tensor,
                           unlabeled_loss_weight: float = 1.0
                           ) -> dict[str, torch.Tensor]:
    """Combined labeled + masked-pseudo-labeled loss for one student
    training step. `pseudo_mask` zeroes out the loss contribution of any
    (sample, class) pair whose teacher confidence fell in the ambiguous
    band (see generate_pseudo_labels), so uninformative pseudo-labels do
    not corrupt the student's training signal.
    """
    labeled_loss = F.binary_cross_entropy_with_logits(
        student_logits_labeled, labels)

    per_element = F.binary_cross_entropy_with_logits(
        student_logits_unlabeled, pseudo_labels, reduction="none")
    denom = pseudo_mask.sum().clamp_min(1.0)
    unlabeled_loss = (per_element * pseudo_mask).sum() / denom

    total = labeled_loss + unlabeled_loss_weight * unlabeled_loss
    return {
        "loss": total,
        "labeled_loss": labeled_loss.detach(),
        "unlabeled_loss": unlabeled_loss.detach(),
        "pseudo_label_coverage": (pseudo_mask.mean()).detach(),
    }
