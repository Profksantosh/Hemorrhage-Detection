"""Windowing-Invariant Consistency Learning (manuscript Section V).

Implements the loss of Section V-C exactly:

    L = 1/2 [BCE(p1, y) + BCE(p2, y)]
        + lambda_c * KL( sg[p1] || p2 )
        + lambda_e * (1 - cos(z1, z2))

applied to two independently, stochastically windowed renderings of the
same underlying raw-HU slice (manuscript Section V-B).

As documented in manuscript Section II-G / Section V-A (added after a
supplementary literature search located directly relevant prior art), the
dual-view consistency terms here are WICL's differentiated contribution
over single-view random-window augmentation (Ostmo et al. [31], [32],
reimplemented as baseline 4c in models/augmentation_baselines.py) --
*not* the stochastic windowing operator itself, which is shared code
(ich_gen.windowing) used by both WICL and baseline 4c. Baseline 4c is
constructed by training the SAME backbone with the SAME windowing
operator but with K=1 draw and ONLY the supervised term of the loss below
(lambda_c = lambda_e = 0, single view) -- see
augmentation_baselines.random_window_single_view_loss for that exact
reduction, so the two conditions differ ONLY in the presence of the
consistency terms, not in any other confounded detail.
"""
from __future__ import annotations

import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F

from ich_gen.models.backbone import EmbeddingBackbone, DEFAULT_BACKBONE


@dataclasses.dataclass
class WICLLossWeights:
    """lambda_c, lambda_e, tuned on patient-grouped RSNA validation folds
    ONLY (never on any external site), per manuscript Section V-C."""
    lambda_consistency: float = 1.0   # prediction (KL) term weight
    lambda_embedding: float = 0.5     # embedding (cosine) term weight


class WICLModel(nn.Module):
    """Dual-view backbone wrapper implementing the WICL forward pass and
    loss. Both views share the SAME backbone weights (manuscript Section
    V-C: "Both are passed through the same-weight encoder-classifier
    f_theta")."""

    def __init__(self, backbone_name: str = DEFAULT_BACKBONE,
                 pretrained: bool = True, n_labels: int = 6,
                 weights: WICLLossWeights | None = None):
        super().__init__()
        self.backbone = EmbeddingBackbone(backbone_name, pretrained, n_labels)
        self.weights = weights or WICLLossWeights()

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time forward pass: fixed canonical window, single
        view, logits only (manuscript Section V-E -- WICL is trained with
        randomized windows but evaluated with the fixed canonical
        composite, for equal inference cost vs. baseline 2)."""
        logits, _ = self.backbone(x)
        return logits

    def training_step(self, image_1: torch.Tensor, image_2: torch.Tensor,
                       labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """One WICL training step given the two independently windowed
        views of a batch of slices (as produced by
        MultiSiteICHDataset(view_mode="random_window_dual")).

        Returns a dict of scalar loss components plus the total loss,
        so callers (train.py) can log each term separately -- this
        matters for the ablations of manuscript Section VI-D (isolating
        the prediction-only vs. embedding-only vs. both consistency terms).
        """
        logits_1, emb_1 = self.backbone(image_1)
        logits_2, emb_2 = self.backbone(image_2)

        # supervised term: multi-label BCE on both branches, averaged
        bce_1 = F.binary_cross_entropy_with_logits(logits_1, labels)
        bce_2 = F.binary_cross_entropy_with_logits(logits_2, labels)
        supervised_loss = 0.5 * (bce_1 + bce_2)

        # prediction-consistency term: KL(sg[p1] || p2), stop-gradient on
        # branch 1 to avoid the trivial collapsed solution (both branches
        # converging to a constant), per manuscript Section V-C. Treated
        # as independent multi-label Bernoulli distributions per class.
        p1 = torch.sigmoid(logits_1).detach()  # stop-gradient
        p2 = torch.sigmoid(logits_2)
        eps = 1e-7
        p1c = p1.clamp(eps, 1 - eps)
        p2c = p2.clamp(eps, 1 - eps)
        kl_pos = p1c * (torch.log(p1c) - torch.log(p2c))
        kl_neg = (1 - p1c) * (torch.log(1 - p1c) - torch.log(1 - p2c))
        consistency_loss = (kl_pos + kl_neg).mean()

        # embedding-consistency term: 1 - cosine_similarity(z1, z2)
        cos_sim = F.cosine_similarity(emb_1, emb_2, dim=-1)
        embedding_loss = (1.0 - cos_sim).mean()

        total = (supervised_loss
                 + self.weights.lambda_consistency * consistency_loss
                 + self.weights.lambda_embedding * embedding_loss)

        return {
            "loss": total,
            "supervised_loss": supervised_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
            "embedding_loss": embedding_loss.detach(),
        }


def build_wicl_model(backbone_name: str = DEFAULT_BACKBONE,
                      pretrained: bool = True,
                      lambda_consistency: float = 1.0,
                      lambda_embedding: float = 0.5) -> WICLModel:
    weights = WICLLossWeights(lambda_consistency, lambda_embedding)
    return WICLModel(backbone_name, pretrained, weights=weights)
