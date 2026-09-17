# ich_gen — reference implementation

Code for *Windowing-Invariant Representation Learning for Cross-Site
Generalization in Deep Learning-Based Intracranial Hemorrhage Detection
and Subtyping* (project "G1"). See the companion manuscripts for the
actual reported findings and their interpretation — this repository is
the implementation and evidence trail, not a substitute for reading them:

- `Manuscript_G1_SCI_Full/Manuscript_G1_SCI_Full.md` — the full nine-condition study
- `Manuscript_G1_ClinicalImaging/Manuscript_G1_ClinicalImaging.md` — the three-condition clinical-imaging-venue version

## What this is

This is the training / evaluation / statistics implementation for the G1
study, **run for real** against real RSNA 2019, CQ500, PhysioNet-ICH, and
BHSD data on a laboratory compute server ("hp100") — not a synthetic-data
placeholder. `pytest tests/` (11 test files, unit-level) verifies each
module's logic against synthetic arrays with known ground truth;
separately, and on top of that, the full protocol has actually been
trained and scored, and the real result files that back the numbers
reported in the manuscripts are included in `runs/` (see below).

## What's in `runs/`

Each run directory (e.g. `runs/baselines_bhsd_excluded/baseline1_fixed_single/`)
holds:

- `config.json` — the exact hyperparameters/arguments the run was launched with
- `history.json` — per-epoch training-curve log (loss/metrics over time)
- `*_result.json` files (at the `runs/<set>/` level, e.g.
  `leakage_check_rsna_cq500_result.json`, `external_cq500_result.json`,
  `h1_*_result.json`) — scoring/evaluation output: AUROC, sensitivity,
  size-stratified and subtype-stratified metrics, calibration, bootstrap
  CIs, and statistical test results, depending on the script

**Not included:** raw model checkpoints (`*.pt`) and raw imaging data.
Every number in the manuscripts traces to a `history.json` curve or a
`*_result.json` scoring output, not to the model weights — see each
manuscript's Data and Code Availability statement, which states this
explicitly ("raw model checkpoints were not copied and remain
server-only"). `.gitignore` in this repo enforces the same rule.

Run sets present:

| Directory | What it is |
|---|---|
| `runs/decisive/`, `runs/decisive_v2/` | The decisive WICL-vs-baseline-4c check (Research Decision Report's highest-priority comparison), run twice as the protocol was refined; `decisive_v2/` also holds the RSNA/CQ500 patient-overlap leakage check (`leakage_check_*`, `leakage_pair_review.json`) |
| `runs/baselines_v1/` | First BHSD-*inclusive* baseline sweep — superseded for all reported manuscript numbers by `baselines_bhsd_excluded`, kept for the audit trail |
| `runs/baselines_bhsd_excluded/` | **The run set backing every reported manuscript number.** Baselines 1/2/2-fold1/3a/3b/4a/4b/4c/5, WICL, and the WICL prediction-only/embedding-only ablation variants, after RSNA training patients overlapping with BHSD were excluded (see `data/` below) |
| `runs/audit_rederive/` | `derive_overlap.py` — the RSNA/BHSD patient-overlap derivation script referenced by the manuscript's Section 4.4 / Supplementary Table S2 |

`HP100_ARCHIVE_MANIFEST.md` documents exactly what was pulled from the
compute server and why (a reproducibility gate: the evidence needed to
exist somewhere other than a single lab machine with a history of going
offline).

The top-level `baseline*_train.log`, `score_*.log`, `h1_*.py` /
`h3_*.py` / `h4_*.py`, and `run_*.sh` files are the actual driver
scripts and stdout logs from those training/scoring/hypothesis-test runs
— kept for the audit trail, not because they're meant to be a polished
CLI (see "Running it" below for the intended entry points).

## Patient-identifier redaction

Two evidence artifacts that would otherwise contain raw RSNA PatientIDs
are redacted for this public release, consistent with the truncated
SHA-256 method described in the manuscript's Section 4.4 /
Supplementary Table S2:

- `data/bhsd_rsna_overlap_patient_ids_hashed.json` (public) replaces
  `data/bhsd_rsna_overlap_patient_ids.json` (raw IDs, git-ignored, never
  published — RSNA 2019 is a restricted-use Kaggle-competition dataset
  and publishing raw PatientIDs would likely violate its data-use terms).
  `data/Supplementary_Table_S2_RSNA_BHSD_Overlap_Hashed.csv` is the same
  mapping in the exact form referenced by the manuscript.
- `runs/decisive_v2/leakage_check_full_hits.json`,
  `leakage_check_rsna_cq500_result.json`, and `leakage_pair_review.json`
  (the RSNA/CQ500 near-duplicate leakage check) had their raw
  `rsna_id` / `cq500_id` values and local `.dcm` file paths replaced with
  `RSNA_<hash>` / `CQ500_<hash>` tokens (same truncated-SHA-256 method,
  applied consistently within each file) before publication. The
  aggregate findings (counts, PASS/FAIL, hamming-distance stats, and the
  manual-review conclusion that all sampled near-duplicates were
  artifacts, not genuine cross-dataset leakage) are unchanged.

## Getting the data (you must do this yourself)

None of RSNA 2019, CQ500, PhysioNet-ICH, or BHSD are included in this
repository. Each has its own license/access terms and must be obtained
directly:

| Dataset | Where | Access |
|---|---|---|
| **RSNA 2019** (training source) | [Kaggle competition](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection) | Accept competition rules on kaggle.com, then instant download |
| **CQ500** (external test) | [Kaggle mirror](https://www.kaggle.com/datasets/crawford/qureai-headct) of Qure.ai's release (Chilamkurthy et al., *The Lancet* 2018) | Instant |
| **PhysioNet-ICH** (external test) | [physionet.org/content/ct-ich/1.3.1](https://physionet.org/content/ct-ich/1.3.1/) | Restricted — register and sign PhysioNet's Restricted Health Data Use Agreement first |
| **BHSD** (external test, size-stratification source) | [Kaggle mirror](https://www.kaggle.com/datasets/stevezeyuzhang/bhsd-dataset); original at [GitHub](https://github.com/White65534/BHSD) or [Hugging Face](https://huggingface.co/datasets/Wendy-Fly/BHSD) | Instant |

`scripts/download_data.py` automates RSNA/CQ500/BHSD via the Kaggle CLI
(`pip install kaggle`, place your API token at `~/.kaggle/kaggle.json`)
and prints PhysioNet-ICH instructions since that one can't be automated.
Each loader in `ich_gen/datasets/` validates the schema of what you
actually downloaded and fails with a specific `ValueError` naming the
missing/renamed columns rather than silently mis-parsing — read the
"SCHEMA ASSUMPTION" note at the top of the relevant loader's docstring
before trusting its output.

## Setup

```bash
pip install -r requirements.txt
pytest tests/          # unit tests against synthetic data, no GPU/data required
```

## Running it

### Step 1 — the decisive check

```bash
python -m ich_gen.run_pipeline decisive-check \
    --rsna-root /path/to/rsna \
    --max-samples 5000 --epochs 5 --out-dir runs/decisive
```

Trains WICL and the strongest baseline comparator (4c) on a fast
subset/short schedule; score both with `ich_gen.evaluate` and compare
with `ich_gen.stats.delong_paired_auc_test`.

### Step 2 — the full protocol

```bash
python -m ich_gen.run_pipeline full-grid \
    --rsna-root /path/to/rsna \
    --epochs 30 --out-dir runs/full
```

Trains all 9 conditions (`fixed_single_window`, `fixed_three_window`,
`adaptive_hrt`, `adaptive_wem`, `dg_augment`, `dg_coral`,
`random_window_single`, `semi_supervised`, `wicl`) across 5
patient-grouped folds. Re-tune `--coral-weight` on the validation fold
before trusting the `dg_coral` comparison — its raw covariance-alignment
term is numerically tiny by construction, so the default weight
under-scales it (see the CLI help text).

### Step 3 — external evaluation, ablations, calibration, stress test

`ich_gen.evaluate.evaluate_pooled_and_stratified` for CQ500 /
PhysioNet-ICH / BHSD external evaluation, `ich_gen.stress_test` for the
HU-recalibration robustness grid, `ich_gen.failure_analysis` for the
Grad-CAM + windowing-outlier-score mechanism-attribution analysis — all
consume the checkpoint format `train.py` produces (`checkpoint_last.pt`,
not distributed with this repo — retrain to regenerate) and the same
`EvalInputs` / `Sample` structures throughout.

## Design principle followed throughout

Every baseline condition and WICL share the *same* dataset class
(`MultiSiteICHDataset`), the *same* backbone constructor
(`models/backbone.py`), and the *same* evaluation function
(`evaluate.evaluate_pooled_and_stratified`) — only the `view_mode` /
`condition` argument differs. This removes an entire class of "the
comparison wasn't actually fair" objection by construction, since no
condition-specific preprocessing or scoring code path can silently
diverge from the others.

## Module map

| Module | Purpose |
|---|---|
| `ich_gen/windowing.py` | HU windowing utilities, including the torch-differentiable op baseline 3b needs for end-to-end training |
| `ich_gen/datasets/splits.py` | Patient-grouped cross-validation and leakage checks — the module the paper's entire methodological claim rests on |
| `ich_gen/datasets/rsna.py`, `cq500.py`, `physionet_ich.py`, `bhsd.py` | Per-dataset loaders with explicit schema validation |
| `ich_gen/models/backbone.py`, `wicl.py` | The shared backbone and the WICL method |
| `ich_gen/models/adaptive_window.py` | Baselines 3a (HRT) and 3b (WEM, joint end-to-end training) |
| `ich_gen/models/augmentation_baselines.py` | Baselines 4a/4b (CORAL)/4c |
| `ich_gen/models/semi_supervised.py` | Baseline 5 (teacher/pseudo-label/student self-training) |
| `ich_gen/train.py` | All 9 training conditions |
| `ich_gen/evaluate.py` | Pooled + subtype + size-stratified metrics, calibration, bootstrap CIs |
| `ich_gen/stats.py` | DeLong's paired AUC test and other statistical comparisons |
| `ich_gen/stress_test.py`, `failure_analysis.py` | HU-recalibration robustness grid; Grad-CAM/mechanism attribution |
| `ich_gen/run_pipeline.py` | Orchestrates the above (`decisive-check`, `full-grid`) |
