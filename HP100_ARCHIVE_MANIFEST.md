# hp100 Evidence Archive Manifest

**Archived:** 2026-09-16, via `ssh hp100` (172.16.0.77, user `fcse.santoshkumar`) from
`/home/fcse.santoshkumar/ich_windowing/` on the remote host.

**Reason:** Paper_Status_Tracker.md flagged a repro-auditor FAIL-PENDING gate: all
Baseline-5, fold-1 H1 replication, and WICL component-ablation evidence (configs,
`history.json` training curves, and result JSONs) existed only on hp100, a host with
a documented history of going offline (see `~/.ssh/config` note under the
`github-jobdikhao` entry: "hp100 ... is now also unreachable"). This archive closes
that gate by copying the evidence itself, not the raw model checkpoints, into the
local project.

## What was pulled

All files under `ich_windowing/` matching `*.json *.log *.py *.sh *.yaml *.yml *.txt
*.md`, excluding `__pycache__/`. Explicitly **excluded**: model weight files
(`*.pt`, `*.pth`, `*.ckpt`) and other large binaries — these were not needed to
verify the reported numbers (every number in the manuscript traces to a `history.json`
training curve or a `*_result.json` scoring output, not to the raw weights), and
pulling them would have added tens of GB with no evidentiary value beyond what the
JSON/log evidence already provides.

## New directories added under `code/runs/`

- `runs/baselines_v1/` — first BHSD-*inclusive* baseline sweep (superseded by
  `baselines_bhsd_excluded` for all reported manuscript numbers, kept for audit trail)
- `runs/baselines_bhsd_excluded/` — **the run set backing every reported number**:
  baselines 1/2/2-fold1/3a/3b/4a/4b/4c/5, WICL, WICL prediction-only and
  embedding-only ablation variants, plus every corresponding `*_result.json` (H1
  fold-0, H1 fold-1 replication, H1 same-metric check, H1 BHSD-external, H3
  calibration, H4 HU stress test, H4 WICL ablation, Baseline-5 internal/CQ500/
  PhysioNet, and the external ablation H2-style check)
- `runs/audit_rederive/derive_overlap.py` — the RSNA/BHSD patient-overlap derivation
  script referenced by §4.4 / Supplementary Table S2

80 files, ~1.3 MB compressed on the wire. Full file list: see git history of this
manifest's companion listing, or re-run:
`find code/runs/baselines_v1 code/runs/baselines_bhsd_excluded code/runs/audit_rederive -type f`

## One reconciliation note

`runs/decisive_v2/external_cq500_result.json` on hp100 contained additional
CQ500 per-subtype/WICL/baseline-4c stratified entries (`cq500_stratified_wicl`,
`cq500_stratified_baseline4c`, `cq500_stratified_subtype_summary`) appended after
the content already present in the local copy — a strict superset, not a conflicting
value. The local file was updated to the fuller version; the prior local-only copy is
preserved as `external_cq500_result.LOCAL_PRE_ARCHIVE_BACKUP.json` in the same
directory.

## What this does not resolve

This archives evidence files (configs, logs, result JSONs), not raw model
checkpoints. Full weight-level reproducibility still requires re-running training
from the scripts in `code/`, which remain the authoritative source. This archive is
sufficient to independently verify every numerical claim in the manuscript traces to
a real logged run, which was the specific gap Paper_Status_Tracker.md identified.
