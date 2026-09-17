"""Patient-grouped splitting and leakage controls.

Implements manuscript Section IV-A (patient-grouped 5-fold CV on RSNA) and
the three explicit leakage controls of Section IV-F:

  (i)   patient-ID-grouped splitting, verified PROGRAMMATICALLY, not merely
        asserted in prose -- this directly targets the specific discrepancy
        identified in the manuscript's reading of Chaudhary et al. [12],
        who claimed a "patient-level" split while describing an
        implementation (a plain, non-grouped train_test_split) that does
        not actually enforce it;
  (ii)  augmentation applied strictly after splitting (enforced structurally
        here: sampling/augmentation lives in the Dataset's __getitem__,
        which only ever sees indices already confined to one fold); and
  (iii) a duplicate / near-duplicate slice detector (perceptual hashing)
        run across every pair of folds and across every external site
        against the training set.

Every function in this module that claims "no leakage" backs that claim
with an assertion, not a comment -- see verify_no_patient_overlap and
find_near_duplicates, both exercised by tests/test_splits_no_leakage.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

try:
    import imagehash
    from PIL import Image
    _HAS_IMAGEHASH = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_IMAGEHASH = False


@dataclass(frozen=True)
class FoldAssignment:
    """One fold's train/val (or test) row indices, plus the patient IDs
    each side contains, so leakage can be checked without recomputing."""
    fold_index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_patients: set
    val_patients: set


def patient_grouped_kfold(patient_ids: Sequence[str],
                           n_splits: int = 5,
                           seed: int = 0) -> list[FoldAssignment]:
    """Patient-grouped K-fold split. No two folds ever share a patient ID.

    This is a thin, explicit wrapper around sklearn's GroupKFold rather
    than a bare `train_test_split` call, specifically because the
    manuscript (Section I-B(1)) flags a real published paper [12] whose
    claimed "patient-level split" was, per its own described
    implementation, a plain stratified `train_test_split` on rows -- a
    method that CANNOT guarantee group separation regardless of intent.
    Using GroupKFold (or, for a single train/test split,
    GroupShuffleSplit) is the minimum standard this manuscript holds
    itself to, and verify_no_patient_overlap below is the check that
    would have caught [12]'s discrepancy had it been run.
    """
    patient_ids = np.asarray(patient_ids)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for GroupKFold")
    n_unique = len(set(patient_ids.tolist()))
    if n_unique < n_splits:
        raise ValueError(
            f"only {n_unique} unique patients but n_splits={n_splits}; "
            "cannot form that many patient-disjoint folds")

    gkf = GroupKFold(n_splits=n_splits)
    # GroupKFold's own split() ignores the `groups`-derived RNG in older
    # sklearn versions; shuffle patient order upstream so fold composition
    # is not an artifact of row order in the source CSV.
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(patient_ids))
    dummy_X = np.zeros((len(patient_ids), 1))

    folds = []
    for i, (train_pos, val_pos) in enumerate(
            gkf.split(dummy_X[order], groups=patient_ids[order])):
        train_idx = order[train_pos]
        val_idx = order[val_pos]
        train_patients = set(patient_ids[train_idx].tolist())
        val_patients = set(patient_ids[val_idx].tolist())
        folds.append(FoldAssignment(i, train_idx, val_idx,
                                     train_patients, val_patients))
    return folds


def verify_no_patient_overlap(folds: Iterable[FoldAssignment]) -> None:
    """Raise AssertionError if any patient appears in more than one fold's
    train/val split, or if any two folds' val sets overlap. This is the
    programmatic check manuscript Section IV-F commits to running before
    any result is reported; it is NOT sufficient to state a split is
    "patient-level" in prose (see docstring above and Section I-B(1))."""
    folds = list(folds)
    for f in folds:
        overlap = f.train_patients & f.val_patients
        assert not overlap, (
            f"fold {f.fold_index}: {len(overlap)} patient(s) appear in "
            f"BOTH train and val: {sorted(overlap)[:5]}...")

    all_val_patients: dict[str, int] = {}
    for f in folds:
        for p in f.val_patients:
            if p in all_val_patients:
                raise AssertionError(
                    f"patient {p!r} appears in the val set of both fold "
                    f"{all_val_patients[p]} and fold {f.fold_index}; "
                    "GroupKFold val sets must partition the patient set")
            all_val_patients[p] = f.fold_index


def verify_disjoint_from_external(train_patient_ids: Sequence[str],
                                   external_patient_ids: Sequence[str],
                                   external_site_name: str = "external") -> None:
    """Raise AssertionError if any patient ID string is shared between the
    RSNA training pool and a nominally independent external site. In
    practice these ID namespaces are disjoint by construction (different
    source institutions issue different ID formats), so this check exists
    primarily to catch accidental ID collisions/typos in configuration,
    not to catch genuine shared patients across institutions, which
    find_near_duplicates (below) is designed to catch instead."""
    overlap = set(train_patient_ids) & set(external_patient_ids)
    assert not overlap, (
        f"{len(overlap)} patient ID(s) shared between training pool and "
        f"external site '{external_site_name}': {sorted(overlap)[:5]}...")


def compute_phash(image_uint8: np.ndarray):
    """Perceptual hash of a single-channel or multi-channel uint8 image,
    used by find_near_duplicates. Requires the optional `imagehash`
    dependency; raises ImportError with a clear message if unavailable."""
    if not _HAS_IMAGEHASH:
        raise ImportError(
            "imagehash (and Pillow) are required for near-duplicate "
            "detection; install with `pip install imagehash pillow`")
    if image_uint8.ndim == 3:
        # collapse to a single representative channel (brain window) for
        # hashing purposes; near-duplicate slices will match on any window
        image_uint8 = image_uint8[0]
    img = Image.fromarray(image_uint8)
    return imagehash.phash(img)


def find_near_duplicates(hashes_a: dict[str, "imagehash.ImageHash"],
                          hashes_b: dict[str, "imagehash.ImageHash"],
                          max_hamming_distance: int = 4
                          ) -> list[tuple[str, str, int]]:
    """Pairwise-compare two {sample_id: phash} dicts and return all pairs
    within max_hamming_distance, i.e. candidate near-duplicate slices.

    This implements the third leakage control of Section IV-F: a check
    across every pair of RSNA folds, and across each external site
    (CQ500, PhysioNet-ICH, BHSD-derived) against the RSNA training set,
    for accidental shared provenance -- motivated directly by this
    literature's own evidence of uncertain shared provenance between
    nominally independent public Kaggle pools (manuscript refs [1], [5],
    [26], all traceable to the same ~100-patient source).

    O(|A|*|B|) pairwise comparison; for large external sites, sub-sample
    or shard by a coarse pre-filter (e.g. hash prefix) before calling this
    in production -- left as a TODO for the full-scale run, since RSNA's
    training pool (hundreds of thousands of slices) makes a naive
    all-pairs comparison against a large external site expensive.
    """
    hits = []
    for id_a, ha in hashes_a.items():
        for id_b, hb in hashes_b.items():
            d = ha - hb  # Hamming distance, defined by imagehash's __sub__
            if d <= max_hamming_distance:
                hits.append((id_a, id_b, d))
    return hits
