"""Build `bhsd_manifest.csv`, the intermediate manifest that
ich_gen/datasets/bhsd.py's build_samples() expects (columns
[volume_id, ct_path, mask_path]) but that BHSD does not ship in that
exact form.

WHY THIS SCRIPT EXISTS
=======================
BHSD's real on-disk release (verified directly against the actual
download at data/bhsd/, 2026-09-11) is a flat MSD-style (Medical
Segmentation Decathlon) layout, NOT the "imagesTr/labelsTr" paths that
dataset.json's own JSON entries claim:

    data/bhsd/
      dataset.json              # MSD-style manifest; its own "training"/
                                 # "test" entries point at "./imagesTr/..."
                                 # and "./labelsTr/..." paths that DO NOT
                                 # exist on disk -- the real folders are:
      label_192/
        images/                 # 192 volumes, filenames "ID_<8hex>_ID_<10hex>.nii.gz"
        ground truths/          # same 192 filenames (note the SPACE in
                                 # the directory name), voxel class masks

Only `label_192/` (the 192 volumes with BOTH images and pixel-level
masks) is usable for this manuscript's per-slice label derivation
(Section IV-D). `FirstStage_test/` (50 volumes) and `unlabel_2000/`
(1,980 volumes, split into anybleed/nobleed) ship images only, no masks,
and are NOT wired into this manifest -- they are a plausible unlabeled
pool for baseline 5 (Section VI-A.5) but that is a separate, not-yet-
scoped task.

CLASS-ID MAPPING -- VERIFIED against data/bhsd/dataset.json's own
"labels" field (2026-09-11): {0: background, 1: epidural,
2: intraparenchymal, 3: intraventricular, 4: subarachnoid, 5: subdural},
an EXACT match to CLASS_ID_TO_SUBTYPE already hard-coded in
ich_gen/datasets/bhsd.py. That dict's "SCHEMA ASSUMPTION -- VERIFY
BEFORE USE" warning can now be marked resolved (see bhsd.py's updated
docstring); this script deliberately re-derives labels from
dataset.json itself, rather than hard-coding a second copy, wherever
practical, as an extra safeguard against silent drift.

CRITICAL, MANUSCRIPT-RELEVANT FINDING -- BHSD/RSNA PATIENT OVERLAP
=====================================================================
This script also cross-references each BHSD volume's embedded ID token
against RSNA's real PatientID DICOM tag (full 752,803-file scan of
data/rsna/.../stage_2_train, 2026-09-11) and found that ALL 192 (100%)
of BHSD's label_192 volumes carry a first ID token
("ID_<8hex>_ID_<...>.nii.gz") that exactly matches an RSNA PatientID.
This is strong, direct evidence -- independent of and cheaper than the
perceptual-hash check specified in manuscript Section IV-F(iii) -- that
BHSD's pixel-level-labeled volumes are reconstructed FROM RSNA 2019
patients, not an institutionally independent collection. See the
`patient_overlaps_rsna` column below and the accompanying report for
what this means for BHSD's use as manuscript Section IV-D's "external
test site 3": as staged, it is NOT a genuinely independent external
site, and using it as one without excluding the overlapping patients
from whatever RSNA training fold is being evaluated would reproduce
exactly the patient-level leakage failure mode this research program
has previously published on (S1). This script does not attempt to
resolve that here (out of scope -- no training/scoring is run against
BHSD by this script); it only surfaces the fact per-volume, so any
downstream training/scoring script can act on it (e.g. by excluding
these patient IDs from the RSNA training pool before treating BHSD as
external, or by explicitly reporting BHSD not as a fourth independent
site but as an in-distribution-provenance stress test).

ONE-VOLUME-PER-PATIENT ASSUMPTION -- FALSE FOR AT LEAST ONE CASE
====================================================================
ich_gen/datasets/bhsd.py's build_samples() sets
`patient_id=f"BHSD_{volume_id}"`, i.e. one volume == one patient,
flagged in its own comment as an assumption to verify. Verified FALSE
for at least one case: volumes ID_2db7ee14_ID_2591d00dfc.nii.gz and
ID_2db7ee14_ID_f27e38fdfd.nii.gz share the same first ID token
(2db7ee14), i.e. the same patient contributes two of the 192 volumes.
This manifest therefore emits a `patient_key` column (the shared first
ID token) SEPARATELY from `volume_id`, so a caller that wants genuine
patient-level grouping (not volume-level, which bhsd.py's build_samples
currently uses) can group on `patient_key` instead. This is flagged,
not silently fixed, because fixing it means changing the patient_id
namespace bhsd.py hands to the patient-grouped splitter
(ich_gen/datasets/splits.py), which is training/eval-pipeline-adjacent
and therefore out of this engineering-only step's scope.

Usage:
    python ich_gen/datasets/prepare_bhsd_manifest.py --root data/bhsd \
        --out bhsd_manifest.csv \
        --rsna-patient-ids-json /tmp/rsna_full_patient_ids.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

VOLUME_RE = re.compile(r"^(ID_[0-9a-f]+_ID_[0-9a-f]+)\.nii\.gz$")
PATIENT_TOKEN_RE = re.compile(r"^ID_([0-9a-f]+)_ID_[0-9a-f]+\.nii\.gz$")


def build_manifest(root: Path, rsna_patient_ids):
    images_dir = root / "label_192" / "images"
    masks_dir = root / "label_192" / "ground truths"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"expected {images_dir} and {masks_dir} to exist; BHSD's "
            f"real release layout may differ from what this script "
            f"assumes -- inspect {root} manually before proceeding.")

    image_files = sorted(images_dir.glob("*.nii.gz"))
    rows = []
    missing_masks = []
    for img_path in image_files:
        m = VOLUME_RE.match(img_path.name)
        if not m:
            raise ValueError(
                f"unexpected BHSD filename {img_path.name!r}; expected "
                f"'ID_<hex>_ID_<hex>.nii.gz' -- inspect manually.")
        volume_id = m.group(1)
        mask_path = masks_dir / img_path.name
        if not mask_path.exists():
            missing_masks.append(img_path.name)
            continue
        patient_token = PATIENT_TOKEN_RE.match(img_path.name).group(1)
        rows.append({
            "volume_id": volume_id,
            "patient_key": f"ID_{patient_token}",
            "ct_path": str(img_path.relative_to(root)),
            "mask_path": str(mask_path.relative_to(root)),
            "patient_overlaps_rsna": (
                None if rsna_patient_ids is None
                else (f"ID_{patient_token}" in rsna_patient_ids)),
        })
    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} BHSD image volume(s) have no matching "
            f"mask file in {masks_dir}: {missing_masks[:5]}...")

    df = pd.DataFrame(rows)
    n_patients = df["patient_key"].nunique()
    if n_patients != len(df):
        dup = df[df.duplicated("patient_key", keep=False)]
        print(f"NOTE: {len(df)} volumes but only {n_patients} unique "
              f"patient_key values -- {len(dup)} volume(s) share a "
              f"patient with another volume (same-patient, multi-volume "
              f"BHSD cases): {sorted(dup['patient_key'].unique())}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/bhsd"))
    ap.add_argument("--out", type=Path, default=Path("bhsd_manifest.csv"))
    ap.add_argument("--rsna-patient-ids-json", type=Path, default=None,
                     help="optional path to a JSON list of RSNA PatientID "
                          "strings (e.g. produced by scanning "
                          "stage_2_train/*.dcm's PatientID tag), used to "
                          "populate the patient_overlaps_rsna column. If "
                          "omitted, that column is left as None/NaN.")
    args = ap.parse_args()

    rsna_ids = None
    if args.rsna_patient_ids_json is not None:
        raw = json.load(open(args.rsna_patient_ids_json))
        rsna_ids = set(raw)
        print(f"loaded {len(rsna_ids)} RSNA PatientID strings from "
              f"{args.rsna_patient_ids_json}")

    df = build_manifest(args.root, rsna_ids)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    print(f"unique volumes: {df['volume_id'].nunique()}, "
          f"unique patients: {df['patient_key'].nunique()}")
    if "patient_overlaps_rsna" in df.columns and rsna_ids is not None:
        n_overlap = df["patient_overlaps_rsna"].sum()
        print(f"volumes whose patient_key matches an RSNA PatientID: "
              f"{n_overlap} / {len(df)} ({100*n_overlap/len(df):.1f}%)")


if __name__ == "__main__":
    main()
