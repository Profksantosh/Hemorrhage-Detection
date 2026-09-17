"""Shared backbone wrapper (manuscript Section VI-B).

Every condition compared in this study -- all five baseline families and
WICL -- uses this SAME function to construct its backbone, with the SAME
default architecture, so that no cross-condition performance difference
can be attributed to a confounded difference in backbone capacity
(manuscript Section VI-B: "Holding all other factors fixed across
conditions is necessary so that any external performance difference can
be attributed to the specific intervention being tested").
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover
    _HAS_TIMM = False

N_LABELS = 6  # any + 5 subtypes, order = ich_gen.datasets.common.LABEL_COLUMNS

# The specific backbone to use is a deliberately open design parameter
# (manuscript "Notes for Manuscript Completion"): densenet121 is used
# elsewhere in this literature specifically for ICH windowing work
# (Songsaeng et al. [14]), which makes it a reasonable, literature-grounded
# default, but any timm model name may be substituted via `backbone_name`.
DEFAULT_BACKBONE = "densenet121"


def build_backbone(backbone_name: str = DEFAULT_BACKBONE,
                    pretrained: bool = True,
                    n_labels: int = N_LABELS) -> nn.Module:
    """Return an ImageNet-pretrained backbone with its classifier head
    replaced by an `n_labels`-way multi-label (sigmoid-trained) head.
    `pretrained=True` requires internet access to download timm weights;
    set False for fully offline smoke-testing (random init)."""
    if not _HAS_TIMM:
        raise ImportError("timm is required; `pip install timm`")
    model = timm.create_model(backbone_name, pretrained=pretrained,
                               num_classes=n_labels, in_chans=3)
    return model


class EmbeddingBackbone(nn.Module):
    """Wraps a timm backbone to also expose its penultimate-layer
    embedding, needed by WICL's embedding-consistency loss term
    (manuscript Section V-C, the cosine-similarity term). timm backbones
    expose this via `forward_features` + `forward_head(..., pre_logits=True)`.
    """

    def __init__(self, backbone_name: str = DEFAULT_BACKBONE,
                 pretrained: bool = True, n_labels: int = N_LABELS):
        super().__init__()
        if not _HAS_TIMM:
            raise ImportError("timm is required; `pip install timm`")
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=n_labels,
            in_chans=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, embedding)."""
        features = self.backbone.forward_features(x)
        embedding = self.backbone.forward_head(features, pre_logits=True)
        logits = self.backbone.forward_head(features, pre_logits=False)
        return logits, embedding
