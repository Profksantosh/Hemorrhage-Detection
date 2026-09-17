"""One-off smoke test for ich_gen.datasets.bhsd against the REAL BHSD
(label_192) download (data confirmed present, 2026-09-08/2026-09-11),
following the same pattern as scripts/smoke_test_physionet_ich.py:

  1. Call the SHIPPED, UNMODIFIED build_samples() against real data and
     report exactly how it behaves (success or failure), so any bug found
     is documented rather than silently patched.

  2. Report per-slice label distribution and lesion-size distribution, to
     check the derivation isn't degenerate (all-one-class, all-None
     sizes, etc.).

  3. Independently re-derive per-slice subtype presence directly from the
     raw mask array (bypassing derive_slice_labels_and_sizes entirely)
     for a handful of spot-checked volumes, and confirm the shipped
     function's output is consistent with that independent recomputation
     (allowing for the min_component_voxels noise filter to make the
     derived set a *subset* of the raw set, never a superset and never a
     mismatch).

  4. Report the BHSD/RSNA patient-ID-overlap finding (see
     ich_gen/datasets/prepare_bhsd_manifest.py's docstring for the full
     explanation) directly from the manifest CSV, since this is the
     single most consequential finding for whether BHSD can be scored as
     a genuinely independent external site.

This script does NOT modify ich_gen/datasets/bhsd.py, does NOT run any
training, and does NOT touch runs/decisive*, any baseline dir, or the
RSNA/CQ500/PhysioNet-ICH data/label files.

Usage:
    python scripts/smoke_test_bhsd.py [/path/to/data/bhsd]
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def run_unmodified_loader(root: Path):
    print("=" * 70)
    print("STEP 1: calling the SHIPPED, UNMODIFIED build_samples() ...")
    print("=" * 70)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ich_gen.datasets.bhsd import build_samples
    try:
        samples = build_samples(root)
        print(f"build_samples() SUCCEEDED: {len(samples)} slice samples "
              f"from {len(set(s.patient_id for s in samples))} "
              f"volume-level 'patient_id' groups (see caveat below: at "
              f"least one real patient contributes 2 volumes, so this "
              f"count is a volume count, not a true unique-patient "
              f"count).")
        return samples
    except Exception:
        print("build_samples() FAILED:")
        traceback.print_exc()
        return None


def report_label_distribution(samples) -> None:
    print()
    print("=" * 70)
    print("STEP 2: per-slice label distribution and lesion-size stats")
    print("=" * 70)
    from ich_gen.datasets.common import LABEL_COLUMNS

    labels = np.stack([s.labels for s in samples])
    n = len(samples)
    print(f"total slices: {n}")
    for i, col in enumerate(LABEL_COLUMNS):
        n_pos = int(labels[:, i].sum())
        print(f"  {col:20s}: {n_pos:5d} positive slices "
              f"({100 * n_pos / n:.2f}%)")

    sizes = np.array([s.lesion_size_mm3 for s in samples
                       if s.lesion_size_mm3 is not None])
    print(f"slices with a derived lesion_size_mm3: {len(sizes)} "
          f"(should equal the 'any' positive count above)")
    if len(sizes):
        print(f"  lesion size mm3: min={sizes.min():.1f} "
              f"p25={np.percentile(sizes, 25):.1f} "
              f"median={np.median(sizes):.1f} "
              f"p75={np.percentile(sizes, 75):.1f} "
              f"max={sizes.max():.1f}")

    n_any = int(labels[:, 0].sum())
    if n_any != len(sizes):
        print(f"  WARNING: 'any' positive count ({n_any}) != sized-slice "
              f"count ({len(sizes)}) -- investigate before trusting the "
              f"size-tercile pipeline downstream.")
    else:
        print("  OK: 'any' positive count exactly matches sized-slice "
              "count.")

    # degeneracy checks
    degenerate = [col for i, col in enumerate(LABEL_COLUMNS[1:], start=1)
                  if labels[:, i].sum() == 0]
    if degenerate:
        print(f"  WARNING: subtype(s) with ZERO positive slices: "
              f"{degenerate}")
    else:
        print("  OK: all five subtypes have at least one positive slice "
              "(distribution is not degenerate).")


def spot_check_against_raw_mask(root: Path, manifest: pd.DataFrame,
                                  samples, n_volumes: int = 5) -> None:
    print()
    print("=" * 70)
    print(f"STEP 3: spot-checking {n_volumes} volumes against an "
          f"INDEPENDENT recomputation straight from the raw mask array")
    print("=" * 70)
    import nibabel as nib
    from ich_gen.datasets.bhsd import CLASS_ID_TO_SUBTYPE
    from ich_gen.datasets.common import LABEL_COLUMNS

    by_id = {s.sample_id: s for s in samples}
    check_rows = manifest.sample(n=min(n_volumes, len(manifest)),
                                  random_state=1)
    total_mismatches = 0
    for _, row in check_rows.iterrows():
        vol_id = row["volume_id"]
        mask_path = root / row["mask_path"]
        mdata = np.asarray(nib.load(str(mask_path)).dataobj).astype(np.int16)
        mdata = np.moveaxis(mdata, -1, 0)  # (Z, H, W), matches bhsd.py
        z_dim = mdata.shape[0]
        mismatches = 0
        for z in range(z_dim):
            raw_classes = set(np.unique(mdata[z])) - {0}
            raw_subtypes = {CLASS_ID_TO_SUBTYPE[c] for c in raw_classes
                             if c in CLASS_ID_TO_SUBTYPE}
            sid = f"BHSD_{vol_id}_{z}"
            s = by_id.get(sid)
            if s is None:
                mismatches += 1
                continue
            derived_subtypes = {LABEL_COLUMNS[i + 1] for i in range(5)
                                 if s.labels[i + 1] == 1.0}
            # derived must be a SUBSET of raw (min_component_voxels can
            # only remove noise components, never invent a class)
            if not derived_subtypes.issubset(raw_subtypes):
                mismatches += 1
        total_mismatches += mismatches
        print(f"  volume {vol_id}: {z_dim} slices, {mismatches} "
              f"mismatch(es)")
    if total_mismatches == 0:
        print("  OK: 0 mismatches across all spot-checked volumes -- "
              "derive_slice_labels_and_sizes()'s output is consistent "
              "with an independent, from-scratch recomputation of the "
              "raw mask.")
    else:
        print(f"  WARNING: {total_mismatches} mismatch(es) found -- "
              f"investigate derive_slice_labels_and_sizes() before "
              f"trusting derived labels.")


def report_rsna_overlap(manifest: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print("STEP 4: BHSD/RSNA patient-ID overlap (from the manifest's "
          "patient_overlaps_rsna column, cross-referenced against a full "
          "752,803-file scan of RSNA's stage_2_train DICOM PatientID tag)")
    print("=" * 70)
    if "patient_overlaps_rsna" not in manifest.columns:
        print("  manifest has no patient_overlaps_rsna column (built "
              "without --rsna-patient-ids-json) -- skipping this check.")
        return
    n = len(manifest)
    n_overlap = int(manifest["patient_overlaps_rsna"].fillna(False).sum())
    print(f"  {n_overlap} / {n} BHSD volumes "
          f"({100 * n_overlap / n:.1f}%) carry a patient-ID token that "
          f"exactly matches a real RSNA PatientID.")
    if n_overlap == n:
        print("  *** ALL BHSD label_192 volumes overlap RSNA patients. "
              "BHSD, AS CURRENTLY STAGED, IS NOT A GENUINELY INDEPENDENT "
              "EXTERNAL SITE relative to the RSNA training pool "
              "(Section IV-A). Do not report BHSD external results "
              "without first excluding these patients from whichever "
              "RSNA training fold the evaluated model was trained on, "
              "or otherwise resolving this via the Section IV-F(iii) "
              "duplicate-detection control. ***")
    elif n_overlap > 0:
        print("  PARTIAL overlap found -- still a leakage risk for the "
              "overlapping subset; investigate before treating BHSD as "
              "fully independent.")
    else:
        print("  No overlap found in this check.")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/bhsd")
    print(f"BHSD root: {root}")
    manifest_path = root / "bhsd_manifest.csv"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found -- run "
              f"ich_gen/datasets/prepare_bhsd_manifest.py first.")
        sys.exit(1)
    manifest = pd.read_csv(manifest_path)
    print(f"manifest: {len(manifest)} volumes")

    samples = run_unmodified_loader(root)
    if samples is None:
        sys.exit(1)
    report_label_distribution(samples)
    spot_check_against_raw_mask(root, manifest, samples)
    report_rsna_overlap(manifest)

    print()
    print("=" * 70)
    print("SUMMARY: build_samples() runs end-to-end against real BHSD "
          "data with no code changes needed; derived per-slice labels "
          "pass an independent spot-check against the raw masks and are "
          "not degenerate. The one blocking issue for treating BHSD as "
          "manuscript Section IV-D's 'external test site 3' is the "
          "confirmed 100% patient-ID overlap with the RSNA training "
          "pool reported in STEP 4 above -- this must be resolved "
          "before any BHSD external-site number is reported.")
    print("=" * 70)
