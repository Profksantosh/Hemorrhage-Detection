"""RSNA 2019 Brain CT Hemorrhage Challenge loader (manuscript ref [27]).

Training source for all conditions (manuscript Section IV-A). Assumes the
standard Kaggle challenge release layout:

    <root>/
      stage_2_train.csv          # columns: ID, Label
      stage_2_train/
        ID_xxxxxxxxx.dcm         # one DICOM file per slice

`stage_2_train.csv` is in long format: one row per (slice, subtype) pair,
with ID = "ID_<12-hex-char SOPInstanceUID suffix>_<subtype>" and
subtype in {any, epidural, intraparenchymal, intraventricular,
subarachnoid, subdural}. This loader pivots it to the wide, one-row-per-
slice schema used by ich_gen.datasets.common (SUBTYPES / LABEL_COLUMNS).

Patient IDs are read from the DICOM PatientID tag (0010,0020), which the
RSNA release populates for every file; this is what patient_grouped_kfold
groups on. Because reading ~750k DICOM headers to build the patient-ID
index is slow, `build_patient_index` caches the result to a parquet/csv
file next to the DICOM directory on first run.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ich_gen.datasets.common import LABEL_COLUMNS, Sample, require_columns

logger = logging.getLogger(__name__)

try:
    import pydicom
    _HAS_PYDICOM = True
except ImportError:  # pragma: no cover
    _HAS_PYDICOM = False


def load_labels_wide(labels_csv: Path) -> pd.DataFrame:
    """Pivot the challenge's long-format stage_2_train.csv into one row
    per slice with columns [sop_uid] + LABEL_COLUMNS."""
    df = pd.read_csv(labels_csv)
    require_columns(df.columns, ["ID", "Label"], "RSNA stage_2_train.csv")

    # "ID_63eb1e259_epidural" -> sop_uid="ID_63eb1e259", subtype="epidural"
    split = df["ID"].str.rsplit("_", n=1, expand=True)
    df = df.assign(sop_uid=split[0], subtype=split[1])

    unknown = set(df["subtype"].unique()) - set(LABEL_COLUMNS)
    if unknown:
        raise ValueError(
            f"RSNA labels CSV contains unexpected subtype token(s) "
            f"{unknown}; expected one of {LABEL_COLUMNS}. The public "
            f"release schema may have changed -- verify stage_2_train.csv "
            f"manually before proceeding.")

    # BUG (found 2026-09-11 while implementing the BHSD/RSNA patient-
    # overlap remediation; self-reported per this project's convention):
    # the real stage_2_train.csv ships with a handful of fully-duplicated
    # (ID, Label) rows -- as of this file, exactly 4 sop_uids each appear
    # twice, back-to-back, as an identical 6-subtype block (48 duplicate
    # rows total out of 4,516,842). This is a known quirk of the official
    # Kaggle release, not a corrupted download (verified: every duplicate
    # pair agrees on Label, i.e. no conflicting-label duplicates). Without
    # this drop_duplicates() call, df.pivot() below deterministically
    # raises "Index contains duplicate entries, cannot reshape" whenever
    # this function is called on the untruncated CSV (verified directly
    # against the real file, byte-identical/unmodified since its 2026-
    # 08-19 download per `stat`). Why no PRIOR logged real-data run hit
    # this is unresolved -- this codebase has no version control history
    # to check, so it's possible an earlier revision of this function
    # already deduplicated and was later reverted, or something else
    # changed; NOT claiming this is new. It is unambiguously a real,
    # reproducible bug in the current code against the current file, and
    # it blocks this task's exclude_patient_ids path, which (per the
    # build_samples docstring) must pivot/merge the FULL, untruncated
    # pool before max_samples truncation. Dropping exact duplicate rows
    # loses zero information since values agree.
    n_before = len(df)
    # BUG (found 2026-09-11, self-reported, fixed same day): this affected-
    # sop_uid count must be computed on the PRE-dedup `df` -- computing it
    # AFTER drop_duplicates() (as an earlier version of this fix did)
    # always reports 0, because drop_duplicates() has, by construction,
    # already collapsed every exact-duplicate group down to one row, so
    # nothing remains duplicated to find afterwards. The dedup itself was
    # always correct (verified via n_dropped, the row-count delta, which
    # does NOT depend on this ordering); only this log message's affected-
    # sop_uid count was wrong (always printed 0 regardless of the true
    # value) until this fix.
    dup_key_preview = df.duplicated(subset=["ID", "Label"], keep=False)
    n_sop_uids_affected_pre_dedup = df.loc[dup_key_preview, "sop_uid"].nunique()
    df = df.drop_duplicates(subset=["ID", "Label"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        n_sop_uids_affected = n_sop_uids_affected_pre_dedup
        logger.warning(
            "stage_2_train.csv contained %d fully-duplicated (ID, Label) "
            "row(s) (%d unique sop_uid(s) affected); dropped as exact "
            "duplicates before pivoting -- see load_labels_wide comment "
            "for provenance. This is a pre-existing RSNA release data "
            "quirk, not introduced by this pipeline.",
            n_dropped, n_sop_uids_affected)

    dup_key = df.duplicated(subset=["sop_uid", "subtype"], keep=False)
    if dup_key.any():
        bad_ids = sorted(set(df.loc[dup_key, "sop_uid"]))
        raise ValueError(
            f"{len(bad_ids)} sop_uid(s) have duplicate (sop_uid, subtype) "
            f"rows with CONFLICTING Label values after removing exact "
            f"duplicates -- cannot safely resolve automatically: {bad_ids}. "
            f"Inspect stage_2_train.csv manually before proceeding.")

    wide = df.pivot(index="sop_uid", columns="subtype", values="Label")
    wide = wide.reindex(columns=LABEL_COLUMNS)
    if wide.isna().any().any():
        n_bad = int(wide.isna().any(axis=1).sum())
        raise ValueError(
            f"{n_bad} slice(s) are missing one or more of the six "
            f"required subtype labels after pivoting -- the labels CSV "
            f"may be truncated or corrupted.")
    wide = wide.astype("float32").reset_index()
    return wide


def load_bhsd_overlap_exclusion_ids(json_path: Path) -> set[str]:
    """Load the BHSD/RSNA patient-ID overlap exclusion list produced by
    code/scripts/derive_bhsd_rsna_overlap.py (see
    code/data/bhsd_rsna_overlap_patient_ids.json for provenance:
    independently re-derived 2026-09-11, 191 unique RSNA PatientIDs
    confirmed -- via direct DICOM PatientID + StudyInstanceUID tag
    reads, 192/192 matched, 0 unmatched -- to also be among BHSD's 192
    labeled external-validation volumes). Returns RAW RSNA PatientID
    strings (no "RSNA_" prefix), suitable for build_samples's
    `exclude_patient_ids` argument.
    """
    import json
    with open(json_path) as f:
        data = json.load(f)
    ids = set(data["excluded_patient_ids"])
    if not ids:
        raise ValueError(f"{json_path} contained an empty exclusion list; "
                          f"this is almost certainly a bug (191 ids expected "
                          f"as of the 2026-09-11 derivation) -- refusing to "
                          f"silently train on the un-excluded RSNA pool.")
    return ids


def build_patient_index(dicom_dir: Path, sop_uids: list[str],
                         cache_path: Path | None = None,
                         force_rebuild: bool = False) -> pd.DataFrame:
    """Return a DataFrame [sop_uid, patient_id] by reading DICOM headers
    (PatientID tag), with an on-disk cache since this is the slow step."""
    if not _HAS_PYDICOM:
        raise ImportError("pydicom is required to read RSNA DICOM headers; "
                           "install with `pip install pydicom`")
    cache_path = cache_path or (dicom_dir.parent / "patient_index_cache.parquet")
    if cache_path.exists() and not force_rebuild:
        cached = pd.read_parquet(cache_path)
        if set(sop_uids).issubset(set(cached["sop_uid"])):
            logger.info("using cached patient index at %s", cache_path)
            return cached[cached["sop_uid"].isin(sop_uids)].reset_index(drop=True)
        logger.info("cache at %s is stale/incomplete, rebuilding", cache_path)

    records = []
    for i, sop_uid in enumerate(sop_uids):
        dcm_path = dicom_dir / f"{sop_uid}.dcm"
        if not dcm_path.exists():
            raise FileNotFoundError(
                f"expected DICOM file not found: {dcm_path}. Check that "
                f"`dicom_dir` points at the stage_2_train/ directory.")
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
        records.append({"sop_uid": sop_uid, "patient_id": str(ds.PatientID)})
        if (i + 1) % 20000 == 0:
            logger.info("read %d/%d DICOM headers", i + 1, len(sop_uids))

    index = pd.DataFrame.from_records(records)
    index.to_parquet(cache_path)
    logger.info("cached patient index (%d rows) to %s", len(index), cache_path)
    return index


def read_hu(dcm_path: Path) -> np.ndarray:
    """Read a DICOM file and return the raw Hounsfield-unit pixel array."""
    if not _HAS_PYDICOM:
        raise ImportError("pydicom is required; `pip install pydicom`")
    ds = pydicom.dcmread(dcm_path)
    pixels = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixels * slope + intercept


def build_samples(root: Path,
                   labels_csv_name: str = "stage_2_train.csv",
                   dicom_subdir: str = "stage_2_train",
                   max_samples: int | None = None,
                   exclude_patient_ids: set[str] | None = None) -> list[Sample]:
    """Construct the full list[Sample] for RSNA, ready to hand to
    patient_grouped_kfold (via the `patient_id` field) and to
    MultiSiteICHDataset.

    `max_samples` is provided purely for fast local smoke-testing on a
    subset (e.g. during development of this pipeline) -- the full
    experimental protocol (manuscript Section IV-A) uses the complete
    training pool.

    `exclude_patient_ids`, if given, is a set of RAW RSNA PatientID
    strings (i.e. WITHOUT the "RSNA_" prefix this function adds below)
    to drop from the pool entirely before any sampling/truncation. This
    exists to remove the RSNA patients confirmed to overlap with BHSD's
    192 labeled volumes (see code/data/bhsd_rsna_overlap_patient_ids.json
    and ich_gen.train.build_folds) -- BHSD is used as external validation
    elsewhere in this pipeline, so any RSNA training patient that is also
    a BHSD patient would leak external-eval identity into training.
    Exclusion is applied to the FULL (untruncated) merged patient index,
    BEFORE `max_samples` truncation, so that (a) excluded patients never
    reach label_matrix/Sample construction under any circumstance, and
    (b) the `max_samples` sample count target is unaffected by exclusion
    (excluding ~191/18,938 RSNA patients is a small fraction of the
    ~752k-slice pool, so this does not require reducing max_samples --
    if it ever did, this function does NOT silently backfill past
    max_samples to compensate; that would be a protocol change requiring
    explicit sign-off).
    """
    root = Path(root)
    labels = load_labels_wide(root / labels_csv_name)
    dicom_dir = root / dicom_subdir

    if exclude_patient_ids:
        # Exclusion must see every RSNA patient before any max_samples
        # truncation is applied (see docstring above) -- so when exclusion
        # is requested, build the patient index for the FULL label set,
        # not a max_samples-truncated slice of it.
        patient_index = build_patient_index(dicom_dir, labels["sop_uid"].tolist())
        merged = labels.merge(patient_index, on="sop_uid", how="left")
        if merged["patient_id"].isna().any():
            missing = int(merged["patient_id"].isna().sum())
            raise ValueError(f"{missing} slice(s) could not be matched to "
                              f"a patient_id; check DICOM directory contents.")

        n_before = len(merged)
        n_patients_before = merged["patient_id"].nunique()
        merged = merged[~merged["patient_id"].isin(exclude_patient_ids)].reset_index(drop=True)
        n_excluded_slices = n_before - len(merged)
        n_excluded_patients = n_patients_before - merged["patient_id"].nunique()
        logger.info(
            "excluded %d slice(s) belonging to %d RSNA patient(s) "
            "confirmed to overlap with BHSD (of %d requested exclusion "
            "ids); %d slices remain", n_excluded_slices, n_excluded_patients,
            len(exclude_patient_ids), len(merged))

        if max_samples is not None:
            merged = merged.iloc[:max_samples].reset_index(drop=True)
    else:
        # Original (pre-exclusion-support) fast path: truncate first so
        # build_patient_index only has to read DICOM headers for the
        # requested subset -- important for smoke-testing on a fresh
        # cache-less environment.
        if max_samples is not None:
            labels = labels.iloc[:max_samples].reset_index(drop=True)
        patient_index = build_patient_index(dicom_dir, labels["sop_uid"].tolist())
        merged = labels.merge(patient_index, on="sop_uid", how="left")
        if merged["patient_id"].isna().any():
            missing = int(merged["patient_id"].isna().sum())
            raise ValueError(f"{missing} slice(s) could not be matched to "
                              f"a patient_id; check DICOM directory contents.")

    label_matrix = merged[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    samples = []
    for i, row in merged.iterrows():
        sop_uid = row["sop_uid"]
        dcm_path = dicom_dir / f"{sop_uid}.dcm"
        samples.append(Sample(
            sample_id=sop_uid,
            patient_id=f"RSNA_{row['patient_id']}",
            site="rsna",
            hu_loader=lambda p=dcm_path: read_hu(p),
            labels=label_matrix[i],
            lesion_size_mm3=None,  # RSNA has no voxel-level mask; size
                                     # stratification for RSNA-internal
                                     # results is therefore unavailable and
                                     # is reported as such (manuscript
                                     # Section IV-E applies size
                                     # stratification at evaluation time on
                                     # BHSD-derived data, where masks exist)
        ))
    return samples
