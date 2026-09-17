"""THROWAWAY diagnostic: reconstruct the PRE-FIX CONTRAST_PATTERNS regex
(the "earlier version" referenced in prepare_cq500_labels.py's own comment,
which omitted the `\+\s*C` alternative) and re-run select_series() against
the SAME current header cache to see exactly which studies -- and how many
slices -- it affects, confirming that the 165-slice gap between
prepare_cq500.log (22,474 slices, captured with the pre-fix regex) and the
on-disk CSV (22,309 rows, produced by the post-fix regex, since the script
file's mtime -- 22:16:17 -- falls chronologically between the log's last
write at 22:13:36 and the CSV's write at 22:16:37) is fully explained by
that regex fix, not by any merge-key dtype/whitespace bug.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ich_gen.datasets import prepare_cq500_labels as m

root = Path("data/cq500")
idx = pd.read_parquet(root / "cq500_header_cache.parquet")

# CURRENT (post-fix) regex, as shipped today
print("current CONTRAST_PATTERNS:", m.CONTRAST_PATTERNS.pattern)
chosen_now, manifest_now = m.select_series(idx, 5.0)
print(f"current (post-fix) chosen: {len(chosen_now)} rows, "
      f"{manifest_now['chosen_series_uid'].notna().sum()} studies")

# PRE-FIX regex: same as current but with the `\+\s*C` alternative removed
# (this is the exact alternative the module docstring says was ADDED to
# fix the six-study bug -- see prepare_cq500_labels.py lines ~99-107)
PRE_FIX_CONTRAST_PATTERNS = re.compile(
    r"POST\s*CONTRAST|\bCT\s*C\b|ORAL\s*&?\s*IV|\d+\s*cc\s*sec|D3D",
    re.IGNORECASE)

# Monkeypatch select_series's module-level CONTRAST_PATTERNS temporarily
orig = m.CONTRAST_PATTERNS
m.CONTRAST_PATTERNS = PRE_FIX_CONTRAST_PATTERNS
try:
    chosen_pre, manifest_pre = m.select_series(idx, 5.0)
finally:
    m.CONTRAST_PATTERNS = orig

print(f"\nreconstructed PRE-FIX chosen: {len(chosen_pre)} rows, "
      f"{manifest_pre['chosen_series_uid'].notna().sum()} studies")
print(f"delta (pre_fix - current): {len(chosen_pre) - len(chosen_now)}")
print(f"(log's stale figure was 22,474; current on-disk CSV is 22,309; "
      f"gap = 165)")

# Diff the manifests to find exactly which studies differ
mn = manifest_now.set_index("study_id")
mp = manifest_pre.set_index("study_id")
common = mn.index.intersection(mp.index)
rows = []
for sid in common:
    a, b = mn.loc[sid], mp.loc[sid]  # a=current(post-fix) b=pre-fix
    if a["chosen_series_uid"] != b["chosen_series_uid"]:
        rows.append({
            "study_id": sid,
            "post_fix_series_uid": a["chosen_series_uid"],
            "post_fix_n_slices": a["n_slices"],
            "post_fix_desc": a["series_desc"],
            "pre_fix_series_uid": b["chosen_series_uid"],
            "pre_fix_n_slices": b["n_slices"],
            "pre_fix_desc": b["series_desc"],
        })
diff = pd.DataFrame(rows)
print(f"\n{len(diff)} studies whose SELECTED SERIES differs between "
      f"pre-fix and post-fix regex:")
print(diff.to_string())
if len(diff):
    print(f"\nsum(post_fix_n_slices) = {diff['post_fix_n_slices'].sum()}")
    print(f"sum(pre_fix_n_slices)  = {diff['pre_fix_n_slices'].sum()}")
    print(f"slice delta explained by these studies = "
          f"{diff['pre_fix_n_slices'].sum() - diff['post_fix_n_slices'].sum()}")
    diff.to_csv("gap3_regex_fix_affected_studies.csv", index=False)
    print("wrote gap3_regex_fix_affected_studies.csv")
