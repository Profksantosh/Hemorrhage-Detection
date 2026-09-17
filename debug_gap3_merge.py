"""THROWAWAY diagnostic script for gap 3 (165-slice merge discrepancy in
prepare_cq500_labels.py). Not part of the permanent pipeline -- reproduces
the exact aggregate_reads / scan_dicom_headers / select_series / merge
sequence from prepare_cq500_labels.py against the REAL data/cq500 files on
this machine, then dumps the exact mechanism and exact affected IDs behind
the row-count discrepancy.

Run from /home/fcse.santoshkumar/ich_windowing with the ich_windowing conda
env active:
    python debug_gap3_merge.py --root data/cq500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ich_gen.datasets.prepare_cq500_labels import (
    aggregate_reads, scan_dicom_headers, select_series, LABEL_COLUMNS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    root = args.root

    labels = aggregate_reads(root / "reads.csv", "majority")
    print(f"labels: {len(labels)} rows, study_id dtype={labels['study_id'].dtype}")

    idx = scan_dicom_headers(root, root / "cq500_header_cache.parquet", force_rebuild=False)
    chosen, manifest = select_series(idx, 5.0)
    print(f"chosen: {len(chosen)} rows, patient_id dtype={chosen['patient_id'].dtype}")

    merged = chosen.merge(labels, left_on="patient_id", right_on="study_id", how="inner")
    print(f"\n=== ROW COUNTS ===")
    print(f"len(chosen)  = {len(chosen)}")
    print(f"len(labels)  = {len(labels)}")
    print(f"len(merged)  = {len(merged)}")
    print(f"chosen - merged = {len(chosen) - len(merged)}")

    # --- Original (raw, unnormalized) set-diff diagnostics, exactly as the
    # script currently computes them ---
    unlabeled_raw = set(chosen["patient_id"]) - set(labels["study_id"])
    unimaged_raw = set(labels["study_id"]) - set(chosen["patient_id"])
    print(f"\n=== RAW (as-shipped) set-diff diagnostics ===")
    print(f"unlabeled_raw (chosen pid not in labels sid): {len(unlabeled_raw)} -> {sorted(unlabeled_raw)}")
    print(f"unimaged_raw (labels sid not in chosen pid): {len(unimaged_raw)} -> {sorted(unimaged_raw)[:10]}")

    # --- Duplicate-key check (both sides) ---
    print(f"\n=== DUPLICATE KEY CHECK ===")
    chosen_dup_pids = chosen["patient_id"][chosen["patient_id"].duplicated(keep=False)]
    labels_dup_sids = labels["study_id"][labels["study_id"].duplicated(keep=False)]
    print(f"chosen['patient_id'] duplicated distinct-value count (dup ROWS by pid, "
          f"expected since many slices per study): {chosen['patient_id'].duplicated().sum()} "
          f"dup rows out of {len(chosen)} (this is EXPECTED -- multiple slices per study)")
    n_dup_study_ids_in_labels = labels["study_id"].duplicated().sum()
    print(f"labels['study_id'] duplicated ROWS (should be 0 if one row per study): "
          f"{n_dup_study_ids_in_labels}")
    if n_dup_study_ids_in_labels:
        print(f"  duplicated study_id values in labels: "
              f"{sorted(set(labels_sid_dup := labels['study_id'][labels['study_id'].duplicated(keep=False)]))}")

    # --- Normalized (dtype + whitespace stripped) comparison ---
    print(f"\n=== NORMALIZED (astype(str).str.strip()) comparison ===")
    chosen_pid_norm = chosen["patient_id"].astype(str).str.strip()
    labels_sid_norm = labels["study_id"].astype(str).str.strip()
    unlabeled_norm = set(chosen_pid_norm) - set(labels_sid_norm)
    print(f"unlabeled_norm: {len(unlabeled_norm)} -> {sorted(unlabeled_norm)}")

    # Look for values that are equal after normalization but NOT equal raw
    # (i.e. hidden whitespace / dtype mismatch was masking a real match)
    raw_only_in_unlabeled = unlabeled_raw - unlabeled_norm
    print(f"IDs that were flagged 'unlabeled' RAW but actually match after "
          f"normalization (i.e. whitespace/dtype was the culprit): "
          f"{len(raw_only_in_unlabeled)} -> {sorted(raw_only_in_unlabeled)}")

    # --- Directly identify the exact DROPPED ROWS by comparing on an
    # unambiguous per-row key (the DICOM file path, which is unique per row
    # in `chosen`) rather than re-deriving from ID set differences. ---
    print(f"\n=== EXACT DROPPED ROWS (by path, ground truth) ===")
    chosen_paths = set(chosen["path"])
    merged_paths = set(merged["path"])
    dropped_paths = chosen_paths - merged_paths
    print(f"chosen paths: {len(chosen_paths)}, merged paths: {len(merged_paths)}, "
          f"dropped: {len(dropped_paths)}")
    dropped_rows = chosen[chosen["path"].isin(dropped_paths)]
    dropped_by_pid = dropped_rows.groupby("patient_id").size().sort_values(ascending=False)
    print(f"dropped rows grouped by patient_id ({dropped_by_pid.shape[0]} distinct "
          f"patient_id(s) affected):")
    print(dropped_by_pid.to_string())

    # For each affected patient_id, show repr() of the value plus repr() of
    # the closest label study_id string (if any) to catch invisible
    # whitespace/encoding differences character-by-character.
    print(f"\n=== CHARACTER-LEVEL INSPECTION of affected patient_id/study_id ===")
    label_sids = list(labels["study_id"])
    for pid in dropped_by_pid.index:
        print(f"chosen patient_id repr: {pid!r}  (len={len(pid)}, type={type(pid)})")
        # does a normalized version exist in labels?
        pid_norm = pid.strip()
        match = [s for s in label_sids if s.strip() == pid_norm]
        print(f"  normalized match in labels['study_id']: {match!r}")
        close = [s for s in label_sids if pid.strip().lower() == s.strip().lower()]
        print(f"  case-insensitive normalized match: {close!r}")

    # Save the affected-row detail to CSV for the record.
    out_csv = root.parent.parent / "gap3_dropped_rows_detail.csv" if False else Path("gap3_dropped_rows_detail.csv")
    dropped_rows.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv.resolve()} ({len(dropped_rows)} dropped rows, full detail)")


if __name__ == "__main__":
    main()
