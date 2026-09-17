# Gap 1: RSNA <-> CQ500 near-duplicate candidate pair review

**Source:** `runs/decisive_v2/leakage_check_rsna_cq500_result.json` (original run,
2026-09-03) flagged 302 candidate pairs at phash Hamming distance <= 4 out of an
800,000-pair subsampled comparison (400 RSNA training slices x 2,000 CQ500 slices,
seed=0). Only the top 20 were persisted in that JSON.

**Full reproduction:** `gap1_reproduce_full_hits.py` re-ran the exact same
deterministic subsample (seed=0) and reproduced **n_hits = 302**, exactly matching
the original run -- confirming this is a valid, exact reproduction, not an
approximation. Full breakdown: **6 pairs at Hamming=0**, 75 at Hamming=2, 221 at
Hamming=4. Full list: `runs/decisive_v2/leakage_check_full_hits.json`.

**Sample inspected:** all 6 Hamming=0 pairs, plus a spread of 8/75 Hamming=2 and
8/221 Hamming=4 pairs (22 pairs total) via `gap1_inspect_pairs.py`, which:
- loaded both slices' real HU pixel arrays with the pipeline's own loaders
  (`ich_gen.datasets.rsna.read_hu`, same function used by CQ500 via
  `ich_gen.datasets.cq500`),
- computed mean absolute HU difference, Pearson correlation, and SSIM (on the
  brain-windowed image) after resizing both to a common 256x256 grid,
- saved a side-by-side PNG (RSNA | CQ500 | abs-diff heatmap) per pair to
  `runs/decisive_v2/leakage_pair_review_images/`,
- applied an automated classification heuristic, THEN every one of the 22 pairs
  was additionally visually inspected by hand against the saved PNGs.

**Result: 0/22 TRUE_DUPLICATE_OR_NEAR, 22/22 ARTIFACT, 0/22 UNCLEAR.**

Two consistent artifact patterns explain all 22 pairs, including all 6 Hamming=0
pairs:
1. **Near-blank / degenerate low-information slices** (skull-base or
   orbit-level slices with a bright circular orbital structure, or slices at the
   very top/bottom of the scan range that are almost entirely black). These carry
   almost no distinguishing structure, so imagehash.phash's coarse DCT-based
   descriptor collides across genuinely different images.
2. **Generic normal-appearing brain slices at a similar coarse axial level**
   (e.g. ventricle level, near-vertex level) with a similar round skull silhouette,
   but clearly DIFFERENT individual sulcal/gyral folding and ventricle morphology
   on close inspection -- i.e. different patients whose slices happen to share a
   coarse gradient structure phash is sensitive to.

No pair showed near-zero pixel diff or visually indistinguishable anatomy. Full
per-pair metrics, reasoning, and manual visual-review notes:
`runs/decisive_v2/leakage_pair_review.json`. Side-by-side PNGs:
`runs/decisive_v2/leakage_pair_review_images/*.png`.

## Verdict

**NOT closed as "no leakage, full stop."** This is a 22-pair sample out of a
302-pair candidate set from an explicitly non-exhaustive 400x2,000-slice subsample
(itself a small fraction of the full ~16k RSNA-train x ~22k CQ500 grid). Every
sampled pair -- including all 6 of the highest-risk Hamming=0 pairs -- was
confirmed, with pixel-level evidence, to be a phash false positive rather than
genuine shared provenance. This is reassuring evidence AGAINST cross-dataset
leakage via this mechanism, backed by inspectable artifacts, not inference or
"eyeballing a few." It does not raise the leakage check's coverage beyond what
the underlying subsample already covers, and does not replace check 1 (exact
patient/study-ID overlap, exhaustive, PASS) as the primary leakage control.

## Files
- `runs/decisive_v2/leakage_check_full_hits.json` -- full reproduced 302-pair list
- `gap1_reproduce_full_hits.py` -- reproduction script (throwaway, kept for
  auditability; not part of the permanent pipeline)
- `gap1_inspect_pairs.py` -- pixel-diff + visual-comparison script (throwaway)
- `runs/decisive_v2/leakage_pair_review.json` -- structured per-pair results
- `runs/decisive_v2/leakage_pair_review_images/*.png` -- 22 side-by-side comparison images
