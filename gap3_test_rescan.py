"""THROWAWAY diagnostic: test whether select_series() gives a different
result on a FRESHLY-SCANNED (pre-parquet-round-trip) idx DataFrame vs. the
CACHED (post-parquet-round-trip) idx DataFrame, to determine whether the
165-slice discrepancy between the 2026-08-19 log (22,474 slices) and the
on-disk CSV (22,309 rows) is caused by select_series() non-determinism
across a parquet round trip, rather than by the inner-merge step.

Writes the fresh scan to a SEPARATE cache path so the real
cq500_header_cache.parquet is untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ich_gen.datasets.prepare_cq500_labels import scan_dicom_headers, select_series

root = Path("data/cq500")
fresh_cache = root / "cq500_header_cache_RESCAN_TEST.parquet"

print("=== FRESH SCAN (force_rebuild, new cache path, mimics run 1's "
      "in-memory idx before any parquet round trip) ===")
idx_fresh = scan_dicom_headers(root, fresh_cache, force_rebuild=True)
print(f"idx_fresh: {len(idx_fresh)} rows, dtypes:\n{idx_fresh.dtypes}")

chosen_fresh, manifest_fresh = select_series(idx_fresh, 5.0)
print(f"\nchosen_fresh: {len(chosen_fresh)} rows, "
      f"{manifest_fresh['chosen_series_uid'].notna().sum()} studies")

print("\n=== SAME fresh idx, but round-tripped through parquet (simulates "
      "what scan_dicom_headers returns on a CACHED load) ===")
import pandas as pd
idx_roundtrip = pd.read_parquet(fresh_cache)
print(f"idx_roundtrip: {len(idx_roundtrip)} rows, dtypes:\n{idx_roundtrip.dtypes}")
chosen_rt, manifest_rt = select_series(idx_roundtrip, 5.0)
print(f"\nchosen_rt: {len(chosen_rt)} rows, "
      f"{manifest_rt['chosen_series_uid'].notna().sum()} studies")

print("\n=== Compare fresh-idx selection vs round-tripped-idx selection "
      "vs the EXISTING production cache's selection ===")
idx_prod_cache = pd.read_parquet(root / "cq500_header_cache.parquet")
chosen_prod, manifest_prod = select_series(idx_prod_cache, 5.0)
print(f"chosen_prod (existing cache): {len(chosen_prod)} rows")

# Compare manifests to find exactly which studies picked a DIFFERENT
# series_uid or got a DIFFERENT n_slices between fresh (pre-roundtrip) and
# the production cache (post-roundtrip).
mf = manifest_fresh.set_index("study_id")
mp = manifest_prod.set_index("study_id")
common = mf.index.intersection(mp.index)
diffs = []
for sid in common:
    a, b = mf.loc[sid], mp.loc[sid]
    if a["chosen_series_uid"] != b["chosen_series_uid"] or a["n_slices"] != b["n_slices"]:
        diffs.append({
            "study_id": sid,
            "fresh_series_uid": a["chosen_series_uid"], "fresh_n_slices": a["n_slices"],
            "fresh_thickness": a["thickness"], "fresh_desc": a["series_desc"],
            "prod_series_uid": b["chosen_series_uid"], "prod_n_slices": b["n_slices"],
            "prod_thickness": b["thickness"], "prod_desc": b["series_desc"],
        })
diffs_df = pd.DataFrame(diffs)
print(f"\n{len(diffs_df)} studies differ in chosen series between "
      f"fresh-scan selection and production-cache selection")
if len(diffs_df):
    print(diffs_df.to_string())
    print(f"\ntotal fresh n_slices sum for differing studies: {diffs_df['fresh_n_slices'].sum()}")
    print(f"total prod  n_slices sum for differing studies: {diffs_df['prod_n_slices'].sum()}")
    print(f"delta: {diffs_df['fresh_n_slices'].sum() - diffs_df['prod_n_slices'].sum()}")
    diffs_df.to_csv("gap3_series_selection_diffs.csv", index=False)
    print("wrote gap3_series_selection_diffs.csv")

# Also directly check: does idx_fresh (raw) vs idx_roundtrip (same data,
# through parquet) differ in dtype/content in any column relevant to
# selection (thickness, is_bone/is_contrast text)?
print("\n=== dtype comparison: fresh (pre-roundtrip) vs roundtrip ===")
for col in idx_fresh.columns:
    print(f"  {col}: fresh={idx_fresh[col].dtype}  roundtrip={idx_roundtrip[col].dtype}")

# row-order check
same_order = (idx_fresh["sop_uid"].tolist() == idx_roundtrip["sop_uid"].tolist())
print(f"\nrow order identical between fresh and roundtrip: {same_order}")

same_order_vs_prod = (idx_fresh["sop_uid"].reset_index(drop=True).equals(
    idx_prod_cache["sop_uid"].reset_index(drop=True)))
print(f"row order identical between THIS fresh scan and EXISTING production "
      f"cache (tests filesystem scan-order determinism across separate "
      f"scans): {same_order_vs_prod}")
