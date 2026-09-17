"""Baseline 4 family (manuscript Section VI-A): generic and windowing-
specific augmentation/alignment baselines that do NOT use WICL's
consistency objective.

Baseline 4c (random_window_single_view_loss) is, per the manuscript's
Novelty Audit (Section IX) and Section II-G, THE single most important
comparator in the entire experimental design: it reimplements Ostmo et
al.'s single-view random-window augmentation [31], [32] using the exact
same stochastic windowing operator WICL uses (ich_gen.windowing), so that
the WICL-vs-4c difference isolates ONLY the dual-view consistency
mechanism, holding the underlying augmentation distribution identical.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Baseline 4a: heavy pixel-space augmentation (RandAugment/AugMix-style)
# ---------------------------------------------------------------------------

class CTAppropriateRandAugment:
    """A reduced RandAugment-style policy restricted to operations that are
    physically sensible for an already-windowed CT composite (unlike
    ImageNet RandAugment's default policy set, which includes operations
    such as Color/Posterize/Solarize that are not meaningful for a
    windowed medical grayscale-derived image and are deliberately
    EXCLUDED here): random rotation, translation, mild brightness/contrast
    jitter, and cutout. Operates on an already-windowed (3, H, W) tensor
    in [0, 1] (post to_tensor, pre-normalization) -- call BEFORE
    ImageNet normalization in the data pipeline.
    """

    def __init__(self, n_ops: int = 2, magnitude: float = 0.3, seed: int = 0):
        self.n_ops = n_ops
        self.magnitude = magnitude
        self.gen = torch.Generator().manual_seed(seed)

    def _rotate_translate(self, x: torch.Tensor) -> torch.Tensor:
        angle = (torch.rand(1, generator=self.gen).item() * 2 - 1) * 15 * self.magnitude
        theta = torch.tensor([
            [torch.cos(torch.deg2rad(torch.tensor(angle))), -torch.sin(torch.deg2rad(torch.tensor(angle))), 0.0],
            [torch.sin(torch.deg2rad(torch.tensor(angle))), torch.cos(torch.deg2rad(torch.tensor(angle))), 0.0],
        ]).unsqueeze(0).float()
        grid = F.affine_grid(theta, [1, *x.shape], align_corners=False)
        return F.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)

    def _brightness_contrast(self, x: torch.Tensor) -> torch.Tensor:
        brightness = 1.0 + (torch.rand(1, generator=self.gen).item() * 2 - 1) * 0.2 * self.magnitude
        contrast = 1.0 + (torch.rand(1, generator=self.gen).item() * 2 - 1) * 0.2 * self.magnitude
        mean = x.mean()
        return ((x - mean) * contrast + mean) * brightness

    def _cutout(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        size = int(min(h, w) * 0.15 * self.magnitude)
        if size < 1:
            return x
        cy = torch.randint(0, h, (1,), generator=self.gen).item()
        cx = torch.randint(0, w, (1,), generator=self.gen).item()
        y0, y1 = max(0, cy - size // 2), min(h, cy + size // 2)
        x0, x1 = max(0, cx - size // 2), min(w, cx + size // 2)
        x = x.clone()
        x[:, y0:y1, x0:x1] = 0.0
        return x

    OPS = ("rotate_translate", "brightness_contrast", "cutout")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ops = torch.randperm(len(self.OPS), generator=self.gen)[: self.n_ops]
        for i in ops.tolist():
            name = self.OPS[i]
            x = getattr(self, f"_{name}")(x)
        return x


# ---------------------------------------------------------------------------
# Baseline 4b: feature-alignment regularization (CORAL)
# ---------------------------------------------------------------------------

def coral_loss(source_features: torch.Tensor,
                target_features: torch.Tensor) -> torch.Tensor:
    """CORrelation ALignment loss (Sun & Saenko 2016): squared Frobenius
    distance between source/target feature covariance matrices, applied
    across RSNA training folds treated as pseudo-domains (manuscript
    Section VI-A: "no external-domain data is available at training time
    in a zero-shot protocol"). `source_features`/`target_features`:
    (B, D) penultimate-layer embeddings from two different pseudo-domain
    batches.
    """
    d = source_features.shape[1]

    def _cov(feats: torch.Tensor) -> torch.Tensor:
        feats = feats - feats.mean(dim=0, keepdim=True)
        n = feats.shape[0]
        if n < 2:
            return torch.zeros(d, d, device=feats.device, dtype=feats.dtype)
        return (feats.t() @ feats) / (n - 1)

    cs = _cov(source_features)
    ct = _cov(target_features)
    loss = (cs - ct).pow(2).sum() / (4 * d * d)
    return loss


# ---------------------------------------------------------------------------
# Baseline 4c: random-window augmentation WITHOUT consistency regularization
# (reimplementation of Ostmo et al. [31], [32]) -- see module docstring.
# ---------------------------------------------------------------------------

def random_window_single_view_loss(logits: torch.Tensor,
                                     labels: torch.Tensor) -> torch.Tensor:
    """The entire baseline-4c training objective: standard supervised
    multi-label BCE on a SINGLE stochastically windowed view (produced by
    MultiSiteICHDataset(view_mode="random_window_single"), i.e. ONE call
    to ich_gen.windowing.sample_three_window_composite per training
    step). No consistency term, no second view -- this is deliberately
    the minimal possible reduction of WICL's loss (Section V-C) with
    lambda_consistency = lambda_embedding = 0 AND only one forward pass
    instead of two, so that comparing this baseline's compute cost against
    WICL's is itself informative (WICL costs roughly 2x the forward passes
    per training step; this should be reported explicitly alongside the
    accuracy comparison in the completed manuscript, as a practical
    deployment-cost consideration the current draft does not yet discuss).
    """
    return F.binary_cross_entropy_with_logits(logits, labels)
