"""PhysioNet-ICH (Hssayeni et al.) external test-site loader (manuscript
ref [29]; used as the sole data source in manuscript refs [17], [22]).

ACCESS REQUIREMENT -- START THIS EARLY, IT IS NOT AN INSTANT DOWNLOAD
=======================================================================
Unlike RSNA/CQ500/BHSD (plain Kaggle downloads), this dataset is
RESTRICTED-ACCESS on PhysioNet: you must (1) register a PhysioNet
account and (2) sign PhysioNet's Restricted Health Data Use Agreement
(v1.5.0), and only then can you download the files -- this is a review/
approval step, not an instant click-through, so start it well before you
plan to actually train on this site. See
https://physionet.org/content/ct-ich/1.3.1/ for the current version and
its access page.

SCHEMA -- confirmed structure and column names/casing, independently
cross-checked against a real downloaded copy (v1.3.1)
==============================================================================
    <root>/
      hemorrhage_diagnosis_raw_ct.csv   # per-slice diagnosis labels
      Patient_demographics.csv          # note capital P; "Patient Number"
                                          # (with a space) + age, gender,
                                          # + patient-level diagnostic labels
                                          # under a two-row merged header
                                          # (not currently consumed by this
                                          # loader; available if a
                                          # demographic-subgroup analysis is
                                          # later added)
      ct_scans/
        <PatientNumber:03d>.nii[.gz]    # one 3D volume per patient, filename
                                          # zero-padded to 3 digits (e.g.
                                          # "049.nii", not "49.nii"); 75 of
                                          # the original 82 patients are
                                          # publicly released; patients
                                          # #59-65 are confirmed missing
      masks/
        <PatientNumber:03d>.nii[.gz]    # binary hemorrhage mask (0/~255,
                                          # not strictly {0,1}); same
                                          # zero-padded filename as ct_scans/,
                                          # no "_HGE_Seg" suffix; present for
                                          # all 75 released patients

and expects `hemorrhage_diagnosis_raw_ct.csv` to contain columns:
    PatientNumber, SliceNumber, Intraventricular, Intraparenchymal,
    Subarachnoid, Epidural, Subdural, No_Hemorrhage, Fracture_Yes_No

The folder/file layout and CSV column names/casing above have been
independently verified against a real downloaded copy (v1.3.1, 156/156
files SHA256-checked); see physionet_ich_structure_verification.log.
Treat a require_columns() failure here as an instruction to inspect your
actual CSV and adjust COLUMN_MAP below, not as a bug elsewhere in the
pipeline.

Manuscript Section IV-C treats this as a SECONDARY external stress test
(82 patients total): results from this site should be reported with wide
bootstrap confidence intervals and never as the sole basis for a
generalization claim, consistent with how Chen et al. [8] treat the same
dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ich_gen.datasets.common import LABEL_COLUMNS, Sample, require_columns

try:
    import nibabel as nib
    _HAS_NIBABEL = True
except ImportError:  # pragma: no cover
    _HAS_NIBABEL = False

# Maps this dataset's raw column names (5 subtypes; no "any" column is
# provided -- it is derived as the OR of the five subtype flags) onto the
# shared LABEL_COLUMNS schema (Section: ich_gen.datasets.common).
COLUMN_MAP = {
    "epidural": "Epidural",
    "intraparenchymal": "Intraparenchymal",
    "intraventricular": "Intraventricular",
    "subarachnoid": "Subarachnoid",
    "subdural": "Subdural",
}


def _read_volume_slice(nii_path: Path, slice_index_0based: int) -> np.ndarray:
    if not _HAS_NIBABEL:
        raise ImportError("nibabel is required; `pip install nibabel`")
    vol = nib.load(str(nii_path))
    data = np.asarray(vol.dataobj)
    # PhysioNet-ICH volumes are already stored in Hounsfield units (no
    # separate rescale step, unlike raw DICOM) -- verify this holds for
    # your downloaded copy (e.g. spot-check that skull HU ~ 700-3000)
    # before trusting downstream windowing.
    return data[..., slice_index_0based].astype(np.float32)


def build_samples(root: Path,
                   diagnosis_csv_name: str = "hemorrhage_diagnosis_raw_ct.csv",
                   ct_subdir: str = "ct_scans") -> list[Sample]:
    root = Path(root)
    df = pd.read_csv(root / diagnosis_csv_name)
    required = ["PatientNumber", "SliceNumber"] + list(COLUMN_MAP.values())
    require_columns(df.columns, required, "PhysioNet-ICH diagnosis CSV")

    subtype_flags = df[list(COLUMN_MAP.values())].to_numpy(dtype=np.float32)
    any_flag = (subtype_flags.sum(axis=1) > 0).astype(np.float32)
    label_matrix = np.concatenate([any_flag[:, None], subtype_flags], axis=1)
    # sanity-check column order matches LABEL_COLUMNS = ("any",) + SUBTYPES
    assert list(COLUMN_MAP.keys()) == list(LABEL_COLUMNS[1:]), (
        "COLUMN_MAP key order must match ich_gen.datasets.common.SUBTYPES")

    ct_dir = root / ct_subdir
    samples = []
    for i, row in df.iterrows():
        # Real released files are zero-padded to 3 digits (e.g. "049.nii",
        # not "49.nii") -- confirmed against the actual v1.3.1 download;
        # see physionet_ich_structure_verification.log item 2.
        patient = f"{int(row['PatientNumber']):03d}"
        slice_idx0 = int(row["SliceNumber"]) - 1  # CSV is commonly 1-indexed;
                                                    # verify against your copy
        nii_candidates = [ct_dir / f"{patient}.nii", ct_dir / f"{patient}.nii.gz"]
        nii_path = next((p for p in nii_candidates if p.exists()), None)
        if nii_path is None:
            raise FileNotFoundError(
                f"no volume found for patient {patient} in {ct_dir} "
                f"(tried {[str(p) for p in nii_candidates]})")
        samples.append(Sample(
            sample_id=f"PhysioNetICH_{patient}_{row['SliceNumber']}",
            patient_id=f"PhysioNetICH_{patient}",
            site="physionet_ich",
            hu_loader=lambda p=nii_path, s=slice_idx0: _read_volume_slice(p, s),
            labels=label_matrix[i],
            lesion_size_mm3=None,  # masks/ directory provides pixel-level
                                     # hemorrhage extent per Hssayeni et al.,
                                     # but converting that to a physical
                                     # mm^3 lesion volume for size
                                     # stratification is not implemented in
                                     # this loader (this site is used only
                                     # as a secondary, small-n stress test
                                     # per Section IV-C, not for the primary
                                     # size-stratified analysis, which uses
                                     # BHSD -- see bhsd.py)
        ))
    return samples
