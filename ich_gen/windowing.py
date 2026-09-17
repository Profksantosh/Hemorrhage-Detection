"""CT Hounsfield-unit windowing operators.

Implements the windowing operator W_(l,w) of manuscript Section V-B, the
three canonical clinical windows used throughout the manuscript (brain,
subdural, bone; Section IV-A), and the stochastic windowing distributions
used by WICL (Section V-B) and by the random-window-augmentation-only
baseline 4c (Section VI-A), which reimplements the single-view mechanism
of Ostmo et al. (2023, 2025) as a required comparator.

All functions operate on raw Hounsfield-unit (HU) arrays, i.e. CT pixel
data *after* applying the DICOM RescaleSlope/RescaleIntercept, but before
any display windowing. See datasets/rsna.py for HU recovery from DICOM.
"""
from __future__ import annotations

import dataclasses
import random
from typing import Tuple

import numpy as np
import torch

# Canonical clinical window settings (level, width), in Hounsfield units.
# These match the parameterization used across the reviewed literature
# (Chen et al. 2024 [8]; Malik & Vidyarthi 2024 [9]; Juan et al. 2026 [18]).
CANONICAL_WINDOWS = {
    "brain": (40.0, 80.0),
    "subdural": (80.0, 200.0),
    "bone": (500.0, 3000.0),
}
WINDOW_ORDER = ("brain", "subdural", "bone")


def window_hu(hu: np.ndarray, level: float, width: float) -> np.ndarray:
    """Apply a single HU window, mapping to [0, 255] uint8-range floats.

    Implements manuscript Eq. (Section V-B):
        W_(l,w)(x)_ij = clip((x_ij - (l - w/2)) / w * 255, 0, 255)

    Parameters
    ----------
    hu : array of raw Hounsfield-unit values, any shape.
    level, width : window level (center) and width, in HU. width must be > 0.

    Returns
    -------
    array of the same shape, dtype float32, range [0, 255].
    """
    if width <= 0:
        raise ValueError(f"window width must be positive, got {width}")
    lo = level - width / 2.0
    out = (hu.astype(np.float32) - lo) / width * 255.0
    return np.clip(out, 0.0, 255.0)


def three_window_composite(hu: np.ndarray,
                            windows: dict | None = None) -> np.ndarray:
    """Fixed three-window HU compositing (baseline 2, Section VI-A).

    Stacks brain/subdural/bone windows as a pseudo-RGB image, following
    the preprocessing used by Chen et al. [8], Malik & Vidyarthi [9], and
    Juan et al. [18].

    Returns an array of shape (3, H, W), dtype float32, range [0, 255],
    channel order (brain, subdural, bone).
    """
    windows = windows or CANONICAL_WINDOWS
    channels = [window_hu(hu, *windows[name]) for name in WINDOW_ORDER]
    return np.stack(channels, axis=0)


@dataclasses.dataclass
class WindowSamplingRange:
    """Sampling range for one canonical window, used by WICL (Section V-B)
    and by baseline 4c (single-view random-window augmentation).

    delta_level / delta_width are the +/- HU ranges around the canonical
    (level, width) setting. Per manuscript Section VI-D, these are
    hyperparameters ablated at "narrow", "moderate" (main condition), and
    "wide" breadths; the defaults below are a reasonable starting point
    informed by the vendor default-window variation documented in the
    manuscript (CQ500 spans six scanner models) and should be re-estimated
    empirically from the training data's HU histograms before final
    experiments, per Section V-B.
    """
    level0: float
    width0: float
    delta_level: float
    delta_width: float

    def sample(self, rng: random.Random | None = None) -> Tuple[float, float]:
        rng = rng or random
        level = rng.uniform(self.level0 - self.delta_level,
                             self.level0 + self.delta_level)
        width = rng.uniform(max(self.width0 - self.delta_width, 1.0),
                             self.width0 + self.delta_width)
        return level, width


# Default sampling ranges: "moderate" breadth (Section VI-D main condition).
# delta_width intentionally excludes width -> 0 (division blow-up) via the
# max(..., 1.0) floor in WindowSamplingRange.sample.
DEFAULT_SAMPLING_RANGES = {
    "narrow": {
        "brain": WindowSamplingRange(40.0, 80.0, delta_level=5.0, delta_width=10.0),
        "subdural": WindowSamplingRange(80.0, 200.0, delta_level=10.0, delta_width=25.0),
        "bone": WindowSamplingRange(500.0, 3000.0, delta_level=100.0, delta_width=300.0),
    },
    "moderate": {
        "brain": WindowSamplingRange(40.0, 80.0, delta_level=15.0, delta_width=30.0),
        "subdural": WindowSamplingRange(80.0, 200.0, delta_level=25.0, delta_width=60.0),
        "bone": WindowSamplingRange(500.0, 3000.0, delta_level=250.0, delta_width=700.0),
    },
    "wide": {
        "brain": WindowSamplingRange(40.0, 80.0, delta_level=30.0, delta_width=60.0),
        "subdural": WindowSamplingRange(80.0, 200.0, delta_level=50.0, delta_width=120.0),
        "bone": WindowSamplingRange(500.0, 3000.0, delta_level=500.0, delta_width=1400.0),
    },
}


def sample_three_window_composite(hu: np.ndarray,
                                   breadth: str = "moderate",
                                   rng: random.Random | None = None) -> np.ndarray:
    """Draw ONE stochastic three-window composite (single-view augmentation).

    This is the mechanism used, standing alone (single draw per training
    step, standard supervised loss only, no cross-view consistency term),
    by baseline 4c in Section VI-A, which reimplements Ostmo et al.'s
    random/shifted-window augmentation [31], [32]. WICL (models/wicl.py)
    calls this function TWICE per training slice and adds a consistency
    loss between the two resulting composites; that additional step is
    WICL's differentiated contribution over this function alone (see
    manuscript Section II-G and Section V-A).
    """
    ranges = DEFAULT_SAMPLING_RANGES[breadth]
    channels = []
    for name in WINDOW_ORDER:
        level, width = ranges[name].sample(rng)
        channels.append(window_hu(hu, level, width))
    return np.stack(channels, axis=0)


def two_independent_draws(hu: np.ndarray,
                           breadth: str = "moderate",
                           rng: random.Random | None = None
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Draw the two independent windowed views WICL's consistency loss
    operates on (manuscript Section V-C: x^(1), x^(2))."""
    view1 = sample_three_window_composite(hu, breadth, rng)
    view2 = sample_three_window_composite(hu, breadth, rng)
    return view1, view2


def to_tensor(composite_255: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """Convert a (3, H, W) float32 [0,255] composite to a normalized tensor.

    ImageNet-style normalization is applied (matching the ImageNet-pretrained
    backbones used throughout the reviewed literature and this manuscript's
    Section VI-B), operating on the [0,1]-scaled composite.
    """
    x = torch.from_numpy(composite_255).float() / 255.0
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = (x - mean) / std
    return x


def resize_composite(composite: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize a (C, H, W) composite to (C, size, size).

    Real DICOM series are not guaranteed to share one native pixel-array
    resolution across RSNA/CQ500/PhysioNet-ICH/BHSD (or even within one
    source, across scanners/protocols), so a fixed-size resize step is
    required before batching for any real run -- this was a gap in the
    original single-shape-only synthetic testing and is fixed here rather
    than left implicit. Uses order=1 (bilinear) interpolation via
    skimage, applied per channel to avoid mixing across the pseudo-RGB
    window channels.
    """
    from skimage.transform import resize as sk_resize
    c = composite.shape[0]
    out = np.empty((c, size, size), dtype=np.float32)
    for i in range(c):
        out[i] = sk_resize(composite[i], (size, size), order=1,
                            preserve_range=True, anti_aliasing=True)
    return out.astype(np.float32)


def resize_hu(hu: np.ndarray, size: int) -> np.ndarray:
    """Resize a raw (unwindowed) HU array to (size, size) -- used for the
    adaptive_wem condition (baseline 3b), which windows on-the-fly inside
    the training loop using per-batch predicted parameters, so the resize
    must happen to the raw HU array rather than to a pre-windowed
    composite (models/adaptive_window.py, models/wicl.py's dual-view
    path resizes AFTER windowing instead, since their windowing is fixed
    or dataset-time-sampled and resizing the composite is equivalent and
    simpler there).
    """
    from skimage.transform import resize as sk_resize
    return sk_resize(hu, (size, size), order=1, preserve_range=True,
                      anti_aliasing=True).astype(np.float32)


def differentiable_window_hu(hu: torch.Tensor, level: torch.Tensor,
                              width: torch.Tensor) -> torch.Tensor:
    """torch-differentiable equivalent of window_hu, supporting a
    per-sample (batched) level/width -- required so that
    models.adaptive_window.WindowEstimatorModule (baseline 3b, a
    reimplementation of Karki et al. [33]) can be trained end-to-end: the
    classification loss must be able to backpropagate through the chosen
    window parameters into the estimator module, not just into the
    classifier.

    hu : (B, 1, H, W) raw HU tensor.
    level, width : (B,) tensors, one window setting per sample.
    Returns (B, 1, H, W) in [0, 255].

    NOTE on deviation from Karki et al. [33]: as documented in
    models/adaptive_window.py, [33]'s own WEM is NOT described as trained
    end-to-end through the windowing operation (it uses "distant
    supervision" from precomputed statistics instead); making the
    windowing operator differentiable so gradients CAN flow through it is
    a deliberate simplification/improvement made for this reimplementation
    so that a single, ordinary classification-loss training loop suffices
    -- this should be flagged explicitly as a reimplementation detail in
    the completed manuscript's Methods section, per the Section VI-A
    commitment to document such deviations rather than silently assume
    equivalence to the original method.
    """
    lo = (level - width / 2.0).view(-1, 1, 1, 1)
    w = width.view(-1, 1, 1, 1).clamp_min(1e-3)
    out = (hu - lo) / w * 255.0
    return out.clamp(0.0, 255.0)
