"""Shared label schema, size-tercile logic, and a common base Dataset class
used by all four site-specific loaders (rsna.py, cq500.py, physionet_ich.py,
bhsd.py).

IMPORTANT — read before pointing this at real data
====================================================
None of the four loaders in this package have been executed against the
real RSNA / CQ500 / PhysioNet-ICH / BHSD releases. Each was written against
the dataset's *documented* public schema as described in its official
release notes / the papers that introduced it (see each file's docstring
for the specific source). Public medical-imaging dataset releases
occasionally change column names or directory layouts between versions.
Each loader therefore validates the columns/files it finds against an
explicit expected schema and raises a clear, actionable error (not a
silent misparse) if something does not match -- see `require_columns`
below. Treat a schema-validation failure as "go look at your actual
downloaded files and adjust the SCHEMA constant at the top of that file",
not as a bug in the splitting/windowing/model logic, which is independent
of any single dataset's file layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ich_gen.windowing import (
    three_window_composite,
    two_independent_draws,
    sample_three_window_composite,
    resize_composite,
    resize_hu,
    to_tensor,
)

# Six-way multi-label schema shared by RSNA, CQ500, PhysioNet-ICH, and the
# BHSD-derived labels, matching the schema used across the reviewed
# literature (manuscript refs [8], [9], [10], [11], [18], [20]).
SUBTYPES: tuple[str, ...] = (
    "epidural", "intraparenchymal", "intraventricular",
    "subarachnoid", "subdural",
)
LABEL_COLUMNS: tuple[str, ...] = ("any",) + SUBTYPES  # 6 columns, "any" first


def require_columns(present: Sequence[str], required: Sequence[str],
                     context: str) -> None:
    missing = [c for c in required if c not in present]
    if missing:
        raise ValueError(
            f"[{context}] expected columns {list(required)} but the "
            f"following are missing from the file actually found: "
            f"{missing}. This dataset's public release format may differ "
            f"from what this loader assumes -- inspect your downloaded "
            f"file's actual columns and update the loader/SCHEMA "
            f"accordingly before proceeding; do not silently coerce.")


@dataclass
class Sample:
    """One (slice, patient, labels) record, independent of source dataset.
    site-specific loaders construct a list[Sample] and hand it to
    MultiSiteICHDataset below."""
    sample_id: str          # unique per slice, e.g. SOPInstanceUID
    patient_id: str         # used for grouped splitting -- NEVER the same
                             # namespace across two different source sites
    site: str                # "rsna" | "cq500" | "physionet_ich" | "bhsd"
    hu_loader: object        # zero-arg callable returning the raw HU array
    labels: np.ndarray       # shape (6,), float32, order = LABEL_COLUMNS
    lesion_size_mm3: float | None = None  # None if unknown (Section IV-E)


def assign_size_tercile(lesion_size_mm3: np.ndarray,
                         thresholds: tuple[float, float] | None = None
                         ) -> tuple[np.ndarray, tuple[float, float]]:
    """Assign each positive-lesion sample to small/medium/large (Section
    IV-E). Thresholds are computed from the POOLED TRAINING SET distribution
    and then held FIXED across every external site, per Section IV-E --
    callers must compute `thresholds` once on RSNA training data and pass
    the same tuple to every subsequent call on external data. Passing
    thresholds=None computes them from whatever array is given, which is
    only correct when called on the training pool itself.
    """
    positive = lesion_size_mm3[~np.isnan(lesion_size_mm3)]
    if thresholds is None:
        if positive.size == 0:
            raise ValueError("cannot compute tercile thresholds: no sized "
                              "lesions in the array provided")
        t1, t2 = np.percentile(positive, [33.33, 66.67])
        thresholds = (float(t1), float(t2))
    t1, t2 = thresholds
    tercile = np.full(lesion_size_mm3.shape, fill_value=-1, dtype=np.int8)
    tercile[lesion_size_mm3 < t1] = 0       # small
    tercile[(lesion_size_mm3 >= t1) & (lesion_size_mm3 < t2)] = 1  # medium
    tercile[lesion_size_mm3 >= t2] = 2      # large
    return tercile, thresholds


class MultiSiteICHDataset(Dataset):
    """A torch Dataset over a list[Sample], with a `view_mode` controlling
    which windowing condition (Section VI-A) is applied at __getitem__
    time. All nine conditions (baselines 1-5 with their sub-variants, and
    WICL) share this one dataset class -- only `view_mode` differs -- so
    that no accidental preprocessing discrepancy between conditions can
    creep in (a confound the manuscript explicitly designs against,
    Section VI-B).

    `image_size` (default 224, matching the ImageNet-pretrained backbones'
    conventional input resolution) is applied via
    ich_gen.windowing.resize_composite / resize_hu so that samples of
    differing native DICOM/NIfTI resolution -- expected across
    RSNA/CQ500/PhysioNet-ICH/BHSD, and not guaranteed uniform even within
    one site -- can be batched together without a DataLoader collation
    error.
    """

    VIEW_MODES = (
        "fixed_single_window",   # baseline 1
        "fixed_three_window",    # baseline 2 (and the inference-time mode
                                  # for WICL and for baseline 4c, per
                                  # manuscript Section V-E)
        "random_window_single",  # baseline 4c training-time mode
        "random_window_dual",    # WICL training-time mode (Section V-C)
        "adaptive_hrt",           # baseline 3a (Songsaeng et al. [14]
                                    # reimplementation, models/adaptive_window.py)
        "raw_hu_for_wem",         # baseline 3b (Karki et al. [33]
                                    # reimplementation): returns the resized
                                    # RAW HU array, unwindowed -- windowing
                                    # happens inside the training loop using
                                    # the WindowEstimatorModule's per-batch
                                    # predicted parameters (see train.py)
    )

    def __init__(self, samples: list[Sample], view_mode: str,
                 sampling_breadth: str = "moderate", seed: int = 0,
                 image_size: int = 224):
        if view_mode not in self.VIEW_MODES:
            raise ValueError(f"unknown view_mode {view_mode!r}, must be "
                              f"one of {self.VIEW_MODES}")
        self.samples = samples
        self.view_mode = view_mode
        self.sampling_breadth = sampling_breadth
        self.image_size = image_size
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def set_view_mode(self, view_mode: str) -> None:
        """Switch a dataset instance between training-time randomized
        views and the fixed canonical view used at evaluation time
        (Section V-E: WICL is trained with randomized windows but
        evaluated with the fixed canonical composite, for fair,
        equal-inference-cost comparison against baseline 2)."""
        if view_mode not in self.VIEW_MODES:
            raise ValueError(f"unknown view_mode {view_mode!r}")
        self.view_mode = view_mode

    def _meta(self, s: Sample) -> dict:
        return {"patient_id": s.patient_id, "sample_id": s.sample_id,
                "site": s.site,
                "lesion_size_mm3": np.nan if s.lesion_size_mm3 is None
                else s.lesion_size_mm3}

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        hu = s.hu_loader()
        labels = torch.from_numpy(s.labels)
        meta = self._meta(s)

        if self.view_mode == "fixed_single_window":
            from ich_gen.windowing import window_hu, CANONICAL_WINDOWS
            level, width = CANONICAL_WINDOWS["brain"]
            single = window_hu(hu, level, width)
            composite = np.stack([single, single, single], axis=0)
            composite = resize_composite(composite, self.image_size)
            x = to_tensor(composite)
            return {"image": x, "labels": labels, **meta}

        if self.view_mode == "fixed_three_window":
            composite = three_window_composite(hu)
            composite = resize_composite(composite, self.image_size)
            x = to_tensor(composite)
            return {"image": x, "labels": labels, **meta}

        if self.view_mode == "random_window_single":
            composite = sample_three_window_composite(
                hu, self.sampling_breadth, self._rng)
            composite = resize_composite(composite, self.image_size)
            x = to_tensor(composite)
            return {"image": x, "labels": labels, **meta}

        if self.view_mode == "random_window_dual":
            v1, v2 = two_independent_draws(hu, self.sampling_breadth, self._rng)
            v1 = resize_composite(v1, self.image_size)
            v2 = resize_composite(v2, self.image_size)
            x1, x2 = to_tensor(v1), to_tensor(v2)
            return {"image_1": x1, "image_2": x2, "labels": labels, **meta}

        if self.view_mode == "adaptive_hrt":
            from ich_gen.models.adaptive_window import hrt_composite
            composite = hrt_composite(hu)
            composite = resize_composite(composite, self.image_size)
            x = to_tensor(composite)
            return {"image": x, "labels": labels, **meta}

        if self.view_mode == "raw_hu_for_wem":
            hu_resized = resize_hu(hu, self.image_size)
            # clip to a physically sensible full-body CT range and scale to
            # roughly unit magnitude for network stability, WITHOUT
            # windowing -- windowing parameters are predicted per-batch by
            # WindowEstimatorModule inside the training loop (train.py)
            hu_clipped = np.clip(hu_resized, -1000.0, 3000.0) / 1000.0
            x = torch.from_numpy(hu_clipped).float().unsqueeze(0)  # (1,H,W)
            # the raw (unclipped, unscaled) resized HU is ALSO needed, to
            # perform the actual windowing at full HU range once WEM
            # predicts (level, width) -- returned separately so train.py's
            # loop can apply differentiable_window_hu to it directly
            hu_raw_tensor = torch.from_numpy(hu_resized).float().unsqueeze(0)
            return {"hu_for_wem": x, "hu_raw": hu_raw_tensor,
                    "labels": labels, **meta}

        raise AssertionError("unreachable")  # view_mode validated in __init__
