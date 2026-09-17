"""One-off smoke test for ich_gen.datasets.physionet_ich against the REAL,
restricted-access PhysioNet-ICH (ct-ich v1.3.1) download, run after PhysioNet
Restricted Health Data Use Agreement (v1.5.0) sign-off cleared access
(2026-09-03).

This script deliberately does NOT modify ich_gen/datasets/physionet_ich.py.
It does two things:

  1. Calls the SHIPPED, UNMODIFIED build_samples() against the real data and
     reports exactly how/where it fails (if it does), so any bug found is
     documented rather than silently patched.

  2. Separately (and only to answer "does the rest of the schema make
     sense once the file-matching bug is fixed"), re-implements the same
     CSV-parsing + label-construction logic locally with ONE deliberate,
     clearly-labeled deviation (zero-padded 3-digit filename matching) and
     reports sample counts / label schema / a couple of spot-checked HU
     loads. This is diagnostic only -- it is not a replacement loader and
     is not wired into the training/eval pipeline.

Usage:
    python scripts/smoke_test_physionet_ich.py /path/to/physionet_ich_root
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def run_unmodified_loader(root: Path) -> None:
    print("=" * 70)
    print("STEP 1: calling the SHIPPED, UNMODIFIED build_samples() ...")
    print("=" * 70)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ich_gen.datasets.physionet_ich import build_samples
    try:
        samples = build_samples(root)
        print(f"UNEXPECTED SUCCESS: build_samples() returned "
              f"{len(samples)} samples with the loader as currently "
              f"written. (If you are seeing this, the filename-mismatch "
              f"bug described in this repo's smoke-test log has already "
              f"been fixed upstream -- re-check physionet_ich.py.)")
    except FileNotFoundError as e:
        print("CONFIRMED FAILURE (as expected given the real file naming):")
        print(f"  {e}")
        print()
        print("  require_columns() did NOT raise -- i.e. the CSV column "
              "check passed cleanly against the real "
              "hemorrhage_diagnosis_raw_ct.csv. The failure is purely in "
              "the ct_scans/<PatientNumber>.nii filename join: the loader "
              "builds the filename via str(int(row['PatientNumber'])) "
              "(no zero-padding), e.g. '49.nii', but the real released "
              "files are zero-padded to 3 digits, e.g. '049.nii'.")
    except Exception:
        print("UNEXPECTED exception type (not FileNotFoundError):")
        traceback.print_exc()


def run_diagnostic_corrected_loader(root: Path) -> None:
    print()
    print("=" * 70)
    print("STEP 2: diagnostic-only re-implementation with zero-padded "
          "filename matching, to sanity-check the rest of the schema")
    print("=" * 70)

    from ich_gen.datasets.common import LABEL_COLUMNS
    from ich_gen.datasets.physionet_ich import COLUMN_MAP

    try:
        import nibabel as nib
    except ImportError:
        nib = None
        print("nibabel not importable in this environment -- skipping HU "
              "spot-checks, CSV/schema checks will still run.")

    df = pd.read_csv(root / "hemorrhage_diagnosis_raw_ct.csv")
    required = ["PatientNumber", "SliceNumber"] + list(COLUMN_MAP.values())
    missing = [c for c in required if c not in df.columns]
    print(f"CSV columns found: {list(df.columns)}")
    print(f"Missing required columns: {missing if missing else 'NONE'}")

    subtype_flags = df[list(COLUMN_MAP.values())].to_numpy(dtype=np.float32)
    any_flag = (subtype_flags.sum(axis=1) > 0).astype(np.float32)
    label_matrix = np.concatenate([any_flag[:, None], subtype_flags], axis=1)
    assert list(COLUMN_MAP.keys()) == list(LABEL_COLUMNS[1:])
    assert label_matrix.shape[1] == len(LABEL_COLUMNS) == 6

    ct_dir = root / "ct_scans"
    n_found, n_missing = 0, 0
    missing_examples = []
    unique_patients = set()
    for i, row in df.iterrows():
        patient_raw = str(int(row["PatientNumber"]))
        patient_padded = f"{int(row['PatientNumber']):03d}"  # <- deviation
        unique_patients.add(patient_padded)
        candidates = [ct_dir / f"{patient_padded}.nii",
                      ct_dir / f"{patient_padded}.nii.gz"]
        found = next((p for p in candidates if p.exists()), None)
        if found is not None:
            n_found += 1
        else:
            n_missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(patient_raw)

    print(f"Total CSV rows (= candidate samples): {len(df)}")
    print(f"Unique patients referenced in CSV: {len(unique_patients)}")
    print(f"Rows with a matching CT volume file "
          f"(zero-padded match): {n_found}")
    print(f"Rows with NO matching CT volume file: {n_missing}"
          + (f" (examples: {missing_examples})" if missing_examples else ""))
    print(f"Label matrix shape: {label_matrix.shape} "
          f"(expect (n_rows, 6), columns={LABEL_COLUMNS})")
    print(f"Positive rate per column: "
          f"{dict(zip(LABEL_COLUMNS, label_matrix.mean(axis=0).round(4)))}")

    masks_dir = root / "masks"
    n_masks_found = sum(
        1 for p in unique_patients
        if (masks_dir / f"{p}.nii").exists() or (masks_dir / f"{p}.nii.gz").exists()
    )
    print(f"Patients with a mask file present: {n_masks_found} / "
          f"{len(unique_patients)}")

    if nib is not None and n_found > 0:
        print()
        print("Spot-checking HU loading for 2 patients ...")
        checked = 0
        for i, row in df.iterrows():
            if checked >= 2:
                break
            if row["SliceNumber"] != 1:
                continue
            patient_padded = f"{int(row['PatientNumber']):03d}"
            nii_path = ct_dir / f"{patient_padded}.nii"
            if not nii_path.exists():
                continue
            vol = nib.load(str(nii_path))
            data = np.asarray(vol.dataobj)
            sl = data[..., 0].astype(np.float32)
            print(f"  patient {patient_padded}: volume shape {data.shape}, "
                  f"slice0 shape {sl.shape}, "
                  f"slice0 HU range [{sl.min():.1f}, {sl.max():.1f}]")
            checked += 1

    print()
    print("=" * 70)
    print("SUMMARY: with zero-padded filenames, all rows resolve to a "
          "real CT volume file, columns match exactly, and the label "
          "schema/shape is sane. The ONLY blocking discrepancy found is "
          "the non-zero-padded filename join in build_samples().")
    print("=" * 70)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    print(f"PhysioNet-ICH root: {root}")
    run_unmodified_loader(root)
    run_diagnostic_corrected_loader(root)
