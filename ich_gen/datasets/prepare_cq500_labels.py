"""Build `cq500_labels_aggregated.csv`, the intermediate file that
ich_gen/datasets/cq500.py expects but that CQ500 does not ship.

WHY THIS SCRIPT EXISTS
======================
CQ500 (Chilamkurthy et al., The Lancet 2018; manuscript ref [28]) ships:
  * `reads.csv` -- STUDY-level labels, three independent radiologist
    reads per study (R1:/R2:/R3: column prefixes), and
  * a DICOM tree with MULTIPLE SERIES per study,
whereas `ich_gen/datasets/cq500.py` consumes one row per SLICE with
columns [study_id, slice_path] + LABEL_COLUMNS. This script bridges the
two. Every non-obvious decision it makes is listed below, because each
one materially affects the external-validation numbers this dataset
produces and therefore belongs in the manuscript's methods section, not
buried in a script.

DECISIONS MADE HERE (all overridable via CLI flags)
===================================================
1. READER AGGREGATION -- majority vote (>=2 of 3 readers) per finding,
   mapping CQ500's abbreviations to this project's LABEL_COLUMNS:
       ICH -> any,  IPH -> intraparenchymal,  IVH -> intraventricular,
       SDH -> subdural,  EDH -> epidural,     SAH -> subarachnoid
   Majority vote is the standard reduction for CQ500's 3-reader design.
   `--reader-rule any` (>=1 reader) is offered as a sensitivity analysis.

2. SERIES SELECTION -- one series per study, chosen from DICOM METADATA
   rather than by string-matching directory names (the tree contains
   names like 'CT 4cc sec 150cc D3D on-2' that no stable rule can parse).
   EXCLUDED:
     * bone-reconstruction kernels (ConvolutionKernel contains 'BONE',
       or SeriesDescription says BONE) -- a bone kernel deliberately
       sharpens/alters the HU rendering of soft tissue, which is exactly
       the signal a WINDOWING paper measures; including it would confound
       site effect with reconstruction-kernel effect.
     * contrast-enhanced series (ContrastBolusAgent present, or
       description matching POST CONTRAST / angio-style '4cc sec' /
       'CT C' / 'ORAL IV') -- IV contrast shifts vessel/parenchyma HU
       values, again confounding the HU-window manipulation under study.
   PREFERRED among what remains: the series whose SliceThickness is
   closest to --target-thickness (default 5.0mm), because the RSNA
   training source is predominantly standard-thickness clinical axial
   CT; matching acquisition thickness keeps the zero-shot external test
   a test of SITE shift rather than of slice-thickness shift. Ties break
   toward the series with more slices.

3. SPLIT STUDIES -- the Kaggle mirror chunked its archive into batch
   folders (qct01..qct19) MID-STUDY: 18 study IDs have their series
   spread across two adjacent batch folders, sometimes with the SAME
   series partially present in both. This script therefore groups by
   DICOM PatientID (not by directory) and de-duplicates on
   SOPInstanceUID, so a split study is reassembled rather than either
   half-dropped or double-counted.

4. STUDY-LEVEL LABELS ON SLICES -- CQ500 provides no slice-level
   annotation, so each selected slice inherits its study's label. This
   is the only option the data permits, but it means CQ500 slice-level
   metrics are NOISY BY CONSTRUCTION (a hemorrhage-positive study's
   normal slices are labeled positive). Report CQ500 at the STUDY level
   where possible, and state this propagation explicitly wherever
   slice-level CQ500 numbers appear. (`cq500.py` sets
   patient_id = CQ500_<study_id>, so patient-level aggregation and
   patient-level bootstrap CIs both work correctly downstream.)

Usage
-----
    python prepare_cq500_labels.py --root data/cq500
    # writes data/cq500/cq500_labels_aggregated.csv (+ a _series_manifest.csv
    # audit trail of which series was chosen per study, and why)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

try:
    import pydicom
except ImportError:  # pragma: no cover
    raise SystemExit("pydicom is required; `pip install pydicom`")

# CQ500 abbreviation -> this project's LABEL_COLUMNS name
FINDING_MAP = {
    "ICH": "any",
    "IPH": "intraparenchymal",
    "IVH": "intraventricular",
    "SDH": "subdural",
    "EDH": "epidural",
    "SAH": "subarachnoid",
}
LABEL_COLUMNS = ("any", "epidural", "intraparenchymal", "intraventricular",
                 "subarachnoid", "subdural")
READERS = ("R1", "R2", "R3")

# Series-level exclusions -- see DECISION 2 in the module docstring.
BONE_PATTERNS = re.compile(r"BONE", re.IGNORECASE)
# NOTE on `\+\s*C`: radiology shorthand for "with contrast" -- CQ500 labels
# such series '+C', '+C THIN', '1.25 MM+C'. An earlier version of this
# script omitted it and six studies (e.g. CQ500-CT-26, -267, -292) silently
# selected their contrast-enhanced series even though an identical-thickness
# 'Plain' series was present, because the thickness tie broke arbitrarily.
# Do not remove this pattern without re-checking cq500_series_manifest.csv.
CONTRAST_PATTERNS = re.compile(
    r"POST\s*CONTRAST|\bCT\s*C\b|\+\s*C|ORAL\s*&?\s*IV|\d+\s*cc\s*sec|D3D",
    re.IGNORECASE)
# Positive evidence a series is genuinely non-contrast, used only to break
# ties between otherwise equally-suitable series (belt-and-braces alongside
# the exclusion above).
PLAIN_PATTERNS = re.compile(r"PLAIN|PRE\s*CONTRAST", re.IGNORECASE)


def aggregate_reads(reads_csv: Path, rule: str = "majority") -> pd.DataFrame:
    """DECISION 1: reduce CQ500's three independent reads to one label
    set per study."""
    df = pd.read_csv(reads_csv)
    if "name" not in df.columns:
        raise ValueError(f"{reads_csv} has no 'name' column; got {list(df.columns)[:8]}")

    out = pd.DataFrame({"study_id": df["name"].astype(str)})
    for abbrev, canonical in FINDING_MAP.items():
        cols = [f"{r}:{abbrev}" for r in READERS]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{reads_csv} is missing expected reader column(s) {missing}. "
                f"CQ500's published schema may differ from this download -- "
                f"inspect the header before trusting this aggregation.")
        votes = df[cols].sum(axis=1)
        if rule == "majority":
            out[canonical] = (votes >= 2).astype("float32")
        elif rule == "any":
            out[canonical] = (votes >= 1).astype("float32")
        else:
            raise ValueError(f"unknown --reader-rule {rule!r}")

    # Consistency guard: 'any' should cover the union of the 5 subtypes.
    # Readers occasionally mark a subtype without marking ICH (or vice
    # versa); report it rather than silently reconciling, since which way
    # you reconcile changes the positive rate.
    subtype_union = out[[c for c in LABEL_COLUMNS if c != "any"]].max(axis=1)
    disagree = int((out["any"] != subtype_union).sum())
    if disagree:
        print(f"[note] {disagree}/{len(out)} studies where aggregated 'any' != "
              f"union(subtypes). Left AS-IS (not reconciled); "
              f"'any' is the readers' own ICH call.")
    return out


def scan_dicom_headers(root: Path, cache_path: Path,
                        force_rebuild: bool = False) -> pd.DataFrame:
    """Read every DICOM header under `root` once, caching the result --
    the same slow-step-with-cache pattern as rsna.build_patient_index."""
    if cache_path.exists() and not force_rebuild:
        print(f"[cache] using {cache_path}")
        return pd.read_parquet(cache_path)

    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".dcm", "")]
    print(f"scanning {len(files)} candidate DICOM files under {root} ...")
    records = []
    skipped = 0
    for i, path in enumerate(files):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception:
            skipped += 1
            continue
        if not hasattr(ds, "SOPInstanceUID"):
            skipped += 1
            continue
        records.append({
            "path": str(path),
            "patient_id": str(getattr(ds, "PatientID", "")),
            "sop_uid": str(ds.SOPInstanceUID),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            "series_desc": str(getattr(ds, "SeriesDescription", "")),
            "kernel": str(getattr(ds, "ConvolutionKernel", "")),
            "contrast": str(getattr(ds, "ContrastBolusAgent", "")),
            "thickness": float(getattr(ds, "SliceThickness", 0) or 0),
        })
        if (i + 1) % 20000 == 0:
            print(f"  ...{i + 1}/{len(files)} headers read")

    idx = pd.DataFrame.from_records(records)
    if skipped:
        print(f"[note] skipped {skipped} unreadable/non-DICOM file(s)")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(cache_path)
    print(f"[cache] wrote {len(idx)} slice headers to {cache_path}")
    return idx


def select_series(idx: pd.DataFrame, target_thickness: float = 5.0
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DECISION 2 + 3: pick exactly one usable series per study.

    Returns (selected_slices, manifest) where `manifest` records the
    choice made for every study -- including studies rejected outright --
    so the selection is auditable rather than implicit.
    """
    idx = idx.drop_duplicates(subset="sop_uid").copy()  # DECISION 3

    text = idx["series_desc"].fillna("") + " " + idx["kernel"].fillna("")
    idx["is_bone"] = text.str.contains(BONE_PATTERNS)
    idx["is_contrast"] = (text.str.contains(CONTRAST_PATTERNS)
                           | idx["contrast"].fillna("").str.strip().ne(""))
    usable = idx[~idx["is_bone"] & ~idx["is_contrast"]]

    manifest_rows = []
    keep_series = {}
    for patient_id, group in idx.groupby("patient_id"):
        cand = usable[usable["patient_id"] == patient_id]
        if len(cand) == 0:
            manifest_rows.append({
                "study_id": patient_id, "chosen_series_uid": None,
                "reason": "no non-bone, non-contrast series available",
                "n_slices": 0, "thickness": None, "series_desc": None})
            continue
        stats = (cand.groupby("series_uid")
                     .agg(n_slices=("sop_uid", "size"),
                          thickness=("thickness", "median"),
                          series_desc=("series_desc", "first"))
                     .reset_index())
        # prefer thickness closest to target, then a series whose description
        # positively says plain/pre-contrast, then more slices
        stats["thickness_gap"] = (stats["thickness"] - target_thickness).abs()
        stats["is_plain"] = stats["series_desc"].fillna("").str.contains(PLAIN_PATTERNS)
        stats = stats.sort_values(["thickness_gap", "is_plain", "n_slices"],
                                   ascending=[True, False, False])
        best = stats.iloc[0]
        keep_series[patient_id] = best["series_uid"]
        manifest_rows.append({
            "study_id": patient_id,
            "chosen_series_uid": best["series_uid"],
            "reason": (f"closest to {target_thickness}mm among "
                        f"{len(stats)} usable series"),
            "n_slices": int(best["n_slices"]),
            "thickness": float(best["thickness"]),
            "series_desc": best["series_desc"]})

    manifest = pd.DataFrame(manifest_rows)
    chosen = usable[usable.apply(
        lambda r: keep_series.get(r["patient_id"]) == r["series_uid"], axis=1)]
    return chosen, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                     help="CQ500 root (contains reads.csv and qct*/ folders)")
    ap.add_argument("--out", default="cq500_labels_aggregated.csv")
    ap.add_argument("--reader-rule", default="majority", choices=("majority", "any"))
    ap.add_argument("--target-thickness", type=float, default=5.0)
    ap.add_argument("--force-rescan", action="store_true")
    args = ap.parse_args()

    labels = aggregate_reads(args.root / "reads.csv", args.reader_rule)
    print(f"aggregated labels for {len(labels)} studies "
          f"(rule={args.reader_rule}); positive 'any' rate = "
          f"{labels['any'].mean():.3f}")

    idx = scan_dicom_headers(args.root, args.root / "cq500_header_cache.parquet",
                              args.force_rescan)
    chosen, manifest = select_series(idx, args.target_thickness)
    n_rejected = int(manifest["chosen_series_uid"].isna().sum())
    print(f"selected 1 series for {len(manifest) - n_rejected} studies "
          f"({n_rejected} had no usable non-bone/non-contrast series); "
          f"{len(chosen)} slices total")

    merged = chosen.merge(labels, left_on="patient_id", right_on="study_id",
                           how="inner")
    unlabeled = set(chosen["patient_id"]) - set(labels["study_id"])
    unimaged = set(labels["study_id"]) - set(chosen["patient_id"])
    if unlabeled:
        print(f"[note] {len(unlabeled)} imaged study/studies had no row in "
               f"reads.csv and were DROPPED: {sorted(unlabeled)[:5]}")
    if unimaged:
        print(f"[note] {len(unimaged)} labeled study/studies had no usable "
               f"imaging and were DROPPED: {sorted(unimaged)[:5]}")

    out = pd.DataFrame({
        "study_id": merged["study_id"],
        # cq500.py does `root / row["slice_path"]`, so store paths RELATIVE
        # to --root to keep the CSV portable across machines
        "slice_path": [str(Path(p).relative_to(args.root)) for p in merged["path"]],
    })
    for col in LABEL_COLUMNS:
        out[col] = merged[col].astype("float32")

    out_path = args.root / args.out
    out.to_csv(out_path, index=False)
    manifest.to_csv(args.root / "cq500_series_manifest.csv", index=False)

    summary = {
        "n_slices": len(out),
        "n_studies": int(out["study_id"].nunique()),
        "reader_rule": args.reader_rule,
        "target_thickness": args.target_thickness,
        "slice_positive_rate": {c: float(out[c].mean()) for c in LABEL_COLUMNS},
        "study_positive_rate": {
            c: float(out.groupby("study_id")[c].first().mean()) for c in LABEL_COLUMNS},
    }
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    print(f"wrote {args.root / 'cq500_series_manifest.csv'} (audit trail)")


if __name__ == "__main__":
    main()
