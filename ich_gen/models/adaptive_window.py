"""Adaptive/learned window-selection baselines (manuscript Section VI-A,
baseline 3a/3b): reimplementations of two previously published methods,
included as required comparators, not as part of this manuscript's claimed
contribution (Section II-G).

Both are reimplemented from published method DESCRIPTIONS since neither
paper's code is stated to be public; any simplification relative to the
original is documented in each class's docstring, per the manuscript's
explicit commitment (Section VI-A) to document such deviations rather
than silently assume equivalence.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ich_gen.windowing import window_hu, CANONICAL_WINDOWS


# ---------------------------------------------------------------------------
# Baseline 3a: Songsaeng et al.'s HRT (manuscript ref [14])
# ---------------------------------------------------------------------------

def _boundary_point_count(hu: np.ndarray, level: float, width: float,
                           n_bins: int = 32) -> int:
    """Proxy for the "active-contour boundary-point tracking" heuristic
    described in Songsaeng et al. [14]: counts intensity-histogram bin
    edges with a large first-derivative jump after windowing, as a
    simplified stand-in for full active-contour boundary detection.

    SIMPLIFICATION NOTE: [14]'s original method uses an active-contour
    (snake) model to trace anatomical boundary curves and counts
    boundary points directly in image space; that is a substantially
    more expensive per-image optimization than the histogram-based proxy
    implemented here. This simplification is a deliberate, documented
    deviation (per the manuscript's Section VI-A commitment) made to keep
    baseline 3a tractable to run across the full multi-site protocol; it
    should be flagged explicitly in the completed manuscript's Methods
    section as a reimplementation detail, and, time permitting, replaced
    with a full active-contour implementation (e.g. via
    skimage.segmentation.active_contour) before final submission for a
    more faithful comparison.
    """
    windowed = window_hu(hu, level, width)
    hist, _ = np.histogram(windowed, bins=n_bins, range=(0, 255))
    diffs = np.abs(np.diff(hist.astype(np.float64)))
    # a "boundary" is signalled by a large jump between adjacent bins;
    # count bins whose jump exceeds 1 std above the mean jump
    threshold = diffs.mean() + diffs.std()
    return int((diffs > threshold).sum())


CANDIDATE_WINDOWS = [
    (30.0, 10.0), (35.0, 10.0), (40.0, 10.0), (45.0, 10.0), (50.0, 10.0),
    (55.0, 10.0), (60.0, 10.0), (65.0, 10.0), (70.0, 10.0), (75.0, 10.0),
    (80.0, 10.0),
]  # 11 candidate windows, matching [14]'s "eleven predefined window
   # settings (WW=10, WL={30...80})"


def select_hrt_window(hu: np.ndarray) -> tuple[float, float]:
    """Select the window whose boundary-point count shows the largest
    drop relative to the previous candidate, per [14]'s described
    "largest drop in boundary-point count signals the correct HU cutoff"
    criterion. Falls back to the canonical brain window if no clear drop
    is found."""
    counts = [_boundary_point_count(hu, l, w) for (l, w) in CANDIDATE_WINDOWS]
    drops = [counts[i] - counts[i + 1] for i in range(len(counts) - 1)]
    if not drops or max(drops) <= 0:
        return CANONICAL_WINDOWS["brain"]
    best_idx = int(np.argmax(drops))
    return CANDIDATE_WINDOWS[best_idx]


def hrt_composite(hu: np.ndarray) -> np.ndarray:
    """Per-image adaptive window selected via select_hrt_window, stacked
    to 3 channels (baseline 3a uses one selected window, not three
    canonical ones, per [14]'s single-window design -- unlike baseline 2's
    fixed three-window compositing)."""
    level, width = select_hrt_window(hu)
    single = window_hu(hu, level, width)
    return np.stack([single, single, single], axis=0)


# ---------------------------------------------------------------------------
# Baseline 3b: Karki et al.'s Window Estimator Module (manuscript ref [33])
# ---------------------------------------------------------------------------

class WindowEstimatorModule(nn.Module):
    """A small CNN that predicts window level/width directly from the raw
    HU slice (downsampled), jointly trained with the classifier -- a
    reimplementation of the "distant-supervised" window estimator module
    (WEM) of Karki et al. [33], simplified to predict ONE window (rather
    than [33]'s top-4-settings-and-combine scheme) for tractability as a
    single required baseline among five baseline families; this
    simplification should be documented in the completed manuscript's
    Methods section alongside the [14] reimplementation note above.

    The predicted (level, width) are used to window the input via
    ich_gen.windowing.differentiable_window_hu, a torch-differentiable
    reimplementation of the windowing formula, so that end-to-end
    gradient flow from the classification loss into this module's
    parameters IS supported here (see WEMClassifier below and
    differentiable_window_hu's docstring for why this is a deliberate,
    documented deviation from [33]'s own "distant supervision" training
    scheme rather than an attempt to reproduce it exactly).
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 5, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 5, stride=4, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(hidden_dim * 2, 2)  # -> (level, width) raw

    def forward(self, hu_downsampled: torch.Tensor) -> torch.Tensor:
        """hu_downsampled: (B, 1, H, W) raw HU, roughly normalized (e.g.
        clipped to [-1000, 3000] and scaled) before being passed in.
        Returns raw (level, width) predictions, to be mapped into a
        physically valid range by `decode_window_params`."""
        feats = self.net(hu_downsampled).flatten(1)
        return self.head(feats)


def decode_window_params(raw: torch.Tensor,
                          level_range: tuple[float, float] = (0.0, 100.0),
                          width_range: tuple[float, float] = (10.0, 400.0)
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Map the WEM's unconstrained output to physically valid
    (level, width) ranges via a sigmoid squashing, so training cannot
    propose a degenerate window (width <= 0)."""
    level = torch.sigmoid(raw[:, 0]) * (level_range[1] - level_range[0]) + level_range[0]
    width = torch.sigmoid(raw[:, 1]) * (width_range[1] - width_range[0]) + width_range[0]
    return level, width


class WEMClassifier(nn.Module):
    """Composite module for baseline 3b, end to end: WindowEstimatorModule
    predicts one (level, width) per input slice; that window is applied
    via a differentiable windowing op; the resulting single-channel image
    (replicated to 3 channels, matching every other condition's input
    shape) is classified by the same shared backbone constructor used by
    every other condition (models/backbone.py), so that baseline 3b's
    classifier capacity is not confounded with the other baselines'.

    Both the WEM and the classifier are updated by ONE optimizer over
    ONE combined forward/backward pass per training step (train.py's
    train_one_epoch, "adaptive_wem" branch) -- there is no separate WEM
    pretraining stage in this reimplementation.
    """

    def __init__(self, backbone_name: str = None, pretrained: bool = True,
                 n_labels: int = 6, wem_hidden_dim: int = 32):
        super().__init__()
        from ich_gen.models.backbone import build_backbone, DEFAULT_BACKBONE
        backbone_name = backbone_name or DEFAULT_BACKBONE
        self.wem = WindowEstimatorModule(hidden_dim=wem_hidden_dim)
        self.classifier = build_backbone(backbone_name, pretrained, n_labels)

    def forward(self, hu_for_wem: torch.Tensor, hu_raw: torch.Tensor) -> torch.Tensor:
        """hu_for_wem: (B,1,H,W) clipped/scaled HU, WEM's input.
        hu_raw: (B,1,H,W) the SAME slice at full HU range (unclipped,
        unscaled), the array the predicted window is actually applied to.
        Returns (B, n_labels) logits."""
        raw_params = self.wem(hu_for_wem)
        level, width = decode_window_params(raw_params)
        from ich_gen.windowing import differentiable_window_hu
        windowed = differentiable_window_hu(hu_raw, level, width)  # (B,1,H,W), [0,255]
        windowed_01 = windowed / 255.0
        x = windowed_01.repeat(1, 3, 1, 1)  # replicate to 3 channels
        # match the ImageNet normalization every other condition's input
        # receives via ich_gen.windowing.to_tensor, so the classifier sees
        # inputs on a comparable scale across all nine conditions
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.classifier(x)
