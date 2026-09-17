# AUTO-GENERATED bhsd_excluded variant of h4_hu_recalibration_stress_test.py,
# created 2026-09-11 for the BHSD/RSNA patient-overlap remediation task.
# Functional change from the original: select_holdout_samples() now builds
# its training-pool patient set (used to filter the held-out subset for
# patient-disjointness) with the same BHSD-overlap exclude_patient_ids
# applied to baseline2_bhsd_excluded/WICL_bhsd_excluded's actual training
# pool -- see inline comment at select_holdout_samples() for why this
# matters (using the unexcluded pool definition here would silently check
# disjointness against the WRONG pool). Point --runs-dir / --wicl-dir at
# the bhsd_excluded run directories via CLI flags when invoking. See the
# original h4_hu_recalibration_stress_test.py for full methodology docs.
"""H4 test (manuscript Section III-C "H4 (robustness, exploratory)" /
Section VI-F "Synthetic HU-recalibration robustness test" / Table VII):
applies ich_gen.stress_test's controlled affine HU perturbation grid
(x' = scale*x + offset) to a held-out RSNA subset NEVER used in training
or in any other evaluation in this project (verified below to be both
index-disjoint from, and patient-ID-disjoint from, the RSNA pool
baseline2/WICL were trained+validated on), and compares baseline 2
(fixed_three_window) against WICL (runs/decisive_v2/decisive_wicl) as the
perturbation severity increases.

This is a SYNTHETIC stress test, explicitly distinct from the three REAL
external sites (CQ500/PhysioNet-ICH/BHSD) -- per Section VI-F, the point
is to isolate robustness to HU/calibration shift alone, holding anatomy,
patient population, and annotation protocol fixed (all identical to the
unperturbed RSNA data; only the affine HU transform differs).

DESIGN SOURCE, stated explicitly: the affine perturbation mechanism
(x' = scale*x + offset), the "held-out RSNA subset never used elsewhere"
requirement, and the default 5x5 (scale x offset) grid are NOT invented
by this script -- they are exactly what ich_gen.stress_test.py (already
in this repo, Status: "Done, tested" per README.md) implements and
documents. This script is a driver that (a) selects the held-out RSNA
subset, (b) wraps the two audited checkpoints (baseline2, WICL) as
`model_predict_fn`s, and (c) reports the resulting degradation curve --
it adds no new perturbation design.

DISCLOSED AMBIGUITY (do not treat as silently resolved): manuscript
Section VI-F states the (scale, offset) bounds will be "sampled within
physically plausible calibration-drift bounds documented in CT
quality-assurance literature, to be cited in the completed manuscript
once the specific QA reference is selected and verified" -- i.e. the
manuscript's own text marks the EXACT numeric bounds as not yet decided
(confirmed again at the manuscript's closing "remaining design
parameters" note: "the ... QA-literature citation for physically
plausible HU-drift bounds in Section VI-F ... require[s] a concrete
decision ... before experiments begin"). ich_gen.stress_test.py's own
docstring independently flags its default grid (scale in [0.85, 1.15],
offset in [-60, 60] HU) as "a starting point ... treat the numbers below
as provisional pending that citation, not as an established clinical
tolerance." This script uses that same provisional default grid
UNCHANGED (not a new invention, but also not yet citation-backed) and
reports results explicitly labeled as provisional pending the
manuscript's own still-open QA-citation decision -- this is flagged here
rather than silently presented as final.

Also disclosed: the held-out subset SIZE (2,000 slices, after patient-
overlap filtering) is this script's own choice, since the manuscript does
not specify a sample size for this synthetic test; chosen for inference
tractability (2 models x 25 grid points x ~2,000 slices = 100,000 scored
slice-perturbation pairs) while keeping bootstrap CIs reasonably tight.

Inference-only: both checkpoints loaded read-only; no file under
runs/decisive*, runs/baselines_v1/baseline2_fixed_three, or any
CQ500/RSNA label/leakage-audit file is written to.

Usage
-----
    python h4_hu_recalibration_stress_test.py --runs-dir runs/baselines_v1 \
        --wicl-dir runs/decisive_v2/decisive_wicl --rsna-root data/rsna/...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from ich_gen.datasets.common import MultiSiteICHDataset, LABEL_COLUMNS, SUBTYPES
from ich_gen.datasets.rsna import build_samples as build_rsna_samples
from ich_gen.models.backbone import build_backbone
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stress_test import make_perturbation_grid, HUPerturbation
from ich_gen.datasets.common import Sample

ANY_IDX = LABEL_COLUMNS.index("any")
N_BOOTSTRAP = 1000
SEED = 0
TRAIN_POOL_MAX_SAMPLES = 20000   # matches baseline2/WICL config.json's max_samples
HOLDOUT_RAW_N = 3000             # raw slices pulled beyond the training pool,
                                   # before patient-overlap filtering
HOLDOUT_FINAL_N = 2000            # final held-out size after filtering


def safe_auroc(y_true_bin, y_score):
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_bin, y_score))


def sensitivity(y_true_bin, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    pos = y_true_bin == 1
    if pos.sum() == 0:
        return float("nan")
    tp = ((y_true_bin == 1) & (pred == 1)).sum()
    return float(tp / pos.sum())


def accuracy(y_true_bin, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    return float((pred == y_true_bin).mean())


def rel_deg(reference_val, perturbed_val):
    """(reference - perturbed) / reference; positive = perturbation made
    it worse (lower AUROC/sensitivity), matching h1/h3's rel_deg sign
    convention for performance metrics where higher = better."""
    if reference_val is None or np.isnan(reference_val) or reference_val == 0:
        return float("nan")
    return float((reference_val - perturbed_val) / reference_val)


def load_baseline2(runs_dir: Path, device: str):
    b2_dir = runs_dir / "baseline2_fixed_three"
    with open(b2_dir / "config.json") as f:
        config = json.load(f)
    ckpt = torch.load(b2_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    forward_fn = lambda m, x: m(x)
    return model, config, forward_fn


def load_wicl(wicl_dir: Path, device: str):
    with open(wicl_dir / "config.json") as f:
        config = json.load(f)
    ckpt = torch.load(wicl_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    model = build_wicl_model(config["backbone"], pretrained=False,
                              lambda_consistency=config["lambda_consistency"],
                              lambda_embedding=config["lambda_embedding"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    forward_fn = lambda m, x: m.forward_single(x)  # Section V-E: canonical
                                                      # window, single view
    return model, config, forward_fn


def select_holdout_samples(rsna_root: Path, seed_config: dict):
    """Selects a held-out RSNA subset that is (a) INDEX-disjoint from the
    first TRAIN_POOL_MAX_SAMPLES rows used to build baseline2/WICL's
    5-fold split, and (b) PATIENT-ID-disjoint from every patient in that
    pool (since a patient's slices are not necessarily contiguous in the
    label CSV, index-disjointness alone would not guarantee patient-level
    separation) -- both checked and enforced explicitly, following this
    project's leakage-audit discipline (manuscript Section IV-F)."""
    # BHSD/RSNA overlap remediation (2026-09-11): baseline2_bhsd_excluded
    # and WICL_bhsd_excluded were trained on a pool that excludes the 191
    # BHSD-overlapping RSNA patients BEFORE max_samples truncation (see
    # ich_gen.datasets.rsna.build_samples docstring) -- so "the same
    # prefix baseline2/WICL were trained+validated on" now means this
    # exclusion-then-truncate pool, not a raw CSV-row prefix. Passing the
    # same exclude_patient_ids here is required for pool_patients (and
    # therefore the disjointness filter below) to actually match what
    # those checkpoints were trained on; otherwise this function would
    # silently compute disjointness against the WRONG (pre-remediation)
    # pool definition.
    from ich_gen.datasets.rsna import load_bhsd_overlap_exclusion_ids
    from ich_gen.train import DEFAULT_BHSD_EXCLUSION_JSON
    exclude_ids = load_bhsd_overlap_exclusion_ids(DEFAULT_BHSD_EXCLUSION_JSON)
    print(f"loaded {len(exclude_ids)} BHSD-overlap RSNA patient id(s) to "
          f"exclude, matching the bhsd_excluded training protocol")

    print(f"loading RSNA pool (first {TRAIN_POOL_MAX_SAMPLES} rows AFTER "
          f"BHSD-overlap exclusion, the same pool baseline2/WICL were "
          f"trained+validated on) to determine its patient set ...")
    pool_samples = build_rsna_samples(rsna_root, max_samples=TRAIN_POOL_MAX_SAMPLES,
                                       exclude_patient_ids=exclude_ids)
    pool_patients = {s.patient_id for s in pool_samples}
    print(f"  pool: {len(pool_samples)} slices, {len(pool_patients)} patients")

    print(f"loading RSNA rows [{TRAIN_POOL_MAX_SAMPLES}:"
          f"{TRAIN_POOL_MAX_SAMPLES + HOLDOUT_RAW_N}] (post-exclusion "
          f"ordering) as the candidate held-out subset ...")
    raw_all = build_rsna_samples(rsna_root, max_samples=TRAIN_POOL_MAX_SAMPLES + HOLDOUT_RAW_N,
                                  exclude_patient_ids=exclude_ids)
    candidate = raw_all[TRAIN_POOL_MAX_SAMPLES:]
    assert len(candidate) == HOLDOUT_RAW_N

    filtered = [s for s in candidate if s.patient_id not in pool_patients]
    n_removed = len(candidate) - len(filtered)
    print(f"  candidate held-out: {len(candidate)} slices; "
          f"{n_removed} removed for patient-ID overlap with the training "
          f"pool; {len(filtered)} remain patient-disjoint")

    held_out = filtered[:HOLDOUT_FINAL_N]
    held_out_patients = {s.patient_id for s in held_out}
    assert held_out_patients.isdisjoint(pool_patients), \
        "held-out set is NOT patient-disjoint from the training pool -- bug"
    print(f"  final held-out subset: {len(held_out)} slices, "
          f"{len(held_out_patients)} patients (patient-disjoint from the "
          f"RSNA pool used to train/validate baseline2 and WICL, and never "
          f"used in h1/h3's RSNA-internal-fold evaluation, which only used "
          f"the fold-0 val split of the first {TRAIN_POOL_MAX_SAMPLES} rows)")
    return held_out, n_removed


@torch.no_grad()
def score_grid(model, forward_fn, held_out, seed, perturbations, device,
                batch_size=64, num_workers=4):
    """For each perturbation, apply it to held_out's raw HU (via
    ich_gen.stress_test's Sample-wrapping convention) and score. Returns
    dict[(scale, offset)] -> dict(y_true, y_prob, patients), all sharing
    the SAME sample order/patient_ids across perturbations and models
    (verified by the caller), so bootstrap resampling can reuse a single
    set of patient-index draws across every (model, perturbation) cell."""
    out = {}
    for pert in perturbations:
        key = (pert.scale, pert.offset)

        def _wrap(original_loader, p=pert):
            return lambda: p.apply(original_loader())

        perturbed = [
            Sample(sample_id=s.sample_id, patient_id=s.patient_id, site=s.site,
                   hu_loader=_wrap(s.hu_loader), labels=s.labels,
                   lesion_size_mm3=s.lesion_size_mm3)
            for s in held_out
        ]
        ds = MultiSiteICHDataset(perturbed, view_mode="fixed_three_window", seed=seed)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
        probs, labels, patients = [], [], []
        for batch in loader:
            logits = forward_fn(model, batch["image"].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch["labels"].numpy())
            patients.extend(batch["patient_id"])
        out[key] = {"y_true": np.concatenate(labels), "y_prob": np.concatenate(probs),
                     "patients": np.array(patients)}
        print(f"    scored scale={pert.scale:.3f} offset={pert.offset:+.1f}  "
              f"n={len(out[key]['y_true'])}")
    return out


def bootstrap_resample_indices(patient_ids, rng):
    unique_patients = np.unique(patient_ids)
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    return np.concatenate([np.where(patient_ids == p)[0] for p in sampled])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/baselines_v1"))
    ap.add_argument("--wicl-dir", type=Path, default=Path("runs/decisive_v2/decisive_wicl"))
    ap.add_argument("--rsna-root", type=Path,
                     default=Path("data/rsna/rsna-intracranial-hemorrhage-detection"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading baseline2 and WICL checkpoints (read-only) ...")
    b2_model, b2_config, b2_fwd = load_baseline2(args.runs_dir, device)
    wicl_model, wicl_config, wicl_fwd = load_wicl(args.wicl_dir, device)

    held_out, n_removed_overlap = select_holdout_samples(args.rsna_root, b2_config)

    perturbations = make_perturbation_grid()  # provisional default grid,
                                                 # see module docstring
    print(f"\nperturbation grid: {len(perturbations)} points "
          f"(scale in [0.85,1.15] x5, offset in [-60,60]HU x5) -- "
          "PROVISIONAL bounds, see module docstring")

    print("\nscoring baseline2 across the perturbation grid ...")
    b2_scores = score_grid(b2_model, b2_fwd, held_out, b2_config["seed"],
                             perturbations, device)
    print("\nscoring WICL across the perturbation grid ...")
    wicl_scores = score_grid(wicl_model, wicl_fwd, held_out, wicl_config["seed"],
                               perturbations, device)

    # sanity: identical labels/patients across every grid cell and both models
    ref_key = (perturbations[0].scale, perturbations[0].offset)
    ref_labels = b2_scores[ref_key]["y_true"]
    ref_patients = b2_scores[ref_key]["patients"]
    for scores in (b2_scores, wicl_scores):
        for key, v in scores.items():
            assert np.array_equal(v["patients"], ref_patients), \
                f"patient ordering mismatch at {key}"
            assert np.allclose(v["y_true"], ref_labels), \
                f"label mismatch at {key} (perturbation must not change labels)"

    identity_key = (1.0, 0.0)
    assert identity_key in b2_scores, "grid must include the identity (scale=1,offset=0) point"

    # --- Table VII: full grid results, both conditions ---
    grid_results = []
    for pert in perturbations:
        key = (pert.scale, pert.offset)
        row = {"scale": pert.scale, "offset": pert.offset}
        for cond_name, scores in (("baseline2", b2_scores), ("wicl", wicl_scores)):
            y, p = scores[key]["y_true"], scores[key]["y_prob"]
            row[f"{cond_name}_auroc_any"] = safe_auroc(y[:, ANY_IDX], p[:, ANY_IDX])
            row[f"{cond_name}_accuracy_any"] = accuracy(y[:, ANY_IDX], p[:, ANY_IDX])
            row[f"{cond_name}_sensitivity_any"] = sensitivity(y[:, ANY_IDX], p[:, ANY_IDX])
            row[f"{cond_name}_sensitivity_edh"] = sensitivity(
                y[:, LABEL_COLUMNS.index("epidural")], p[:, LABEL_COLUMNS.index("epidural")])
        grid_results.append(row)

    print("\n=== Table VII grid (scale, offset, AUROC/sensitivity, both "
          "conditions) ===")
    print(f"{'scale':>7}{'offset':>8}{'b2_AUROC':>11}{'wicl_AUROC':>12}"
          f"{'b2_sens':>10}{'wicl_sens':>11}{'b2_EDHsens':>12}{'wicl_EDHsens':>13}")
    for row in grid_results:
        print(f"{row['scale']:>7.3f}{row['offset']:>8.1f}"
              f"{row['baseline2_auroc_any']:>11.4f}{row['wicl_auroc_any']:>12.4f}"
              f"{row['baseline2_sensitivity_any']:>10.4f}{row['wicl_sensitivity_any']:>11.4f}"
              f"{row['baseline2_sensitivity_edh']:>12.4f}{row['wicl_sensitivity_edh']:>13.4f}")

    # --- H4 test: degradation at the 4 corner (most severe) perturbations,
    # relative to the identity point, baseline2 vs WICL ---
    corner_keys = [(s, o) for s in (0.85, 1.15) for o in (-60.0, 60.0)]
    print(f"\n=== H4 TEST: mean AUROC degradation at the 4 most severe "
          f"corner perturbations {corner_keys}, relative to identity "
          f"(scale=1.0, offset=0.0), baseline2 vs WICL ===")

    def corner_mean_degradation(scores, idx_id, idx_corner_by_key):
        y_id, p_id = scores[identity_key]["y_true"][idx_id], scores[identity_key]["y_prob"][idx_id]
        auroc_id = safe_auroc(y_id[:, ANY_IDX], p_id[:, ANY_IDX])
        degs = []
        for key in corner_keys:
            idx_c = idx_corner_by_key[key]
            y_c, p_c = scores[key]["y_true"][idx_c], scores[key]["y_prob"][idx_c]
            auroc_c = safe_auroc(y_c[:, ANY_IDX], p_c[:, ANY_IDX])
            degs.append(rel_deg(auroc_id, auroc_c))
        finite = [d for d in degs if not np.isnan(d)]
        return float(np.mean(finite)) if finite else float("nan"), auroc_id

    n = len(ref_patients)
    full_idx = np.arange(n)
    obs_deg_b2, obs_auroc_id_b2 = corner_mean_degradation(
        b2_scores, full_idx, {k: full_idx for k in corner_keys})
    obs_deg_wicl, obs_auroc_id_wicl = corner_mean_degradation(
        wicl_scores, full_idx, {k: full_idx for k in corner_keys})
    observed_diff = obs_deg_b2 - obs_deg_wicl  # >0 => baseline2 degrades more

    print(f"baseline2: identity AUROC={obs_auroc_id_b2:.4f}  "
          f"mean corner degradation={obs_deg_b2:+.4f}")
    print(f"wicl:      identity AUROC={obs_auroc_id_wicl:.4f}  "
          f"mean corner degradation={obs_deg_wicl:+.4f}")
    print(f"observed diff (baseline2_deg - wicl_deg), >0 => baseline2 "
          f"degrades more (consistent with H4): {observed_diff:+.4f}")

    print(f"\nrunning {N_BOOTSTRAP}-resample patient-level bootstrap "
          "(single shared resample per replicate across both conditions "
          "and all grid points, since all cells share the identical "
          "underlying held-out sample set -- a genuinely PAIRED "
          "comparison, unlike H1/H3's cross-site independent-sample "
          "bootstrap) ...")
    rng = np.random.RandomState(SEED)
    diffs = np.empty(N_BOOTSTRAP)
    b2_deg_boot = np.empty(N_BOOTSTRAP)
    wicl_deg_boot = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = bootstrap_resample_indices(ref_patients, rng)
        idx_map = {k: idx for k in corner_keys}
        d_b2, _ = corner_mean_degradation(b2_scores, idx, idx_map)
        d_w, _ = corner_mean_degradation(wicl_scores, idx, idx_map)
        b2_deg_boot[b] = d_b2
        wicl_deg_boot[b] = d_w
        diffs[b] = d_b2 - d_w

    diffs_finite = diffs[~np.isnan(diffs)]
    n_dropped = int(np.isnan(diffs).sum())
    if observed_diff >= 0:
        p_value = 2 * min((diffs_finite <= 0).mean(), 0.5)
    else:
        p_value = 2 * min((diffs_finite >= 0).mean(), 0.5)
    p_value = float(min(p_value, 1.0))

    b2_deg_ci = tuple(np.nanpercentile(b2_deg_boot, [2.5, 97.5]).tolist())
    wicl_deg_ci = tuple(np.nanpercentile(wicl_deg_boot, [2.5, 97.5]).tolist())

    print(f"\nbaseline2 mean corner degradation: {obs_deg_b2:+.4f} 95% CI {b2_deg_ci}")
    print(f"wicl mean corner degradation:      {obs_deg_wicl:+.4f} 95% CI {wicl_deg_ci}")
    print(f"two-sided bootstrap p-value: {p_value:.4f} "
          f"({n_dropped}/{N_BOOTSTRAP} degenerate replicates dropped)")
    if p_value < 0.05 and observed_diff > 0:
        verdict = ("baseline2 degrades SIGNIFICANTLY MORE than WICL under "
                    "the synthetic HU-recalibration stress test -- "
                    "consistent with H4.")
    elif p_value < 0.05:
        verdict = ("statistically significant but in the OPPOSITE "
                    "direction (WICL degrades more than baseline2) -- H4 "
                    "is NOT supported by this test.")
    else:
        verdict = "NOT statistically significant (p >= 0.05) -- H4 is NOT supported by this test."
    print(f"RESULT: {verdict}")

    results = {
        "grid_results": grid_results,
        "held_out_subset": {
            "n_slices": len(held_out),
            "n_patients": len(set(ref_patients)),
            "n_removed_for_patient_overlap_with_training_pool": n_removed_overlap,
            "index_range_in_rsna_label_csv": [TRAIN_POOL_MAX_SAMPLES,
                                                TRAIN_POOL_MAX_SAMPLES + HOLDOUT_RAW_N],
        },
        "h4_test_corner_degradation": {
            "corner_keys": corner_keys,
            "baseline2_identity_auroc": obs_auroc_id_b2,
            "wicl_identity_auroc": obs_auroc_id_wicl,
            "baseline2_mean_corner_degradation": obs_deg_b2,
            "wicl_mean_corner_degradation": obs_deg_wicl,
            "observed_diff_b2_minus_wicl": observed_diff,
            "n_bootstrap": N_BOOTSTRAP,
            "p_value": p_value,
            "n_degenerate_dropped": n_dropped,
            "baseline2_degradation_ci95": b2_deg_ci,
            "wicl_degradation_ci95": wicl_deg_ci,
            "verdict": verdict,
        },
        "provisional_bounds_disclosure": (
            "The (scale, offset) grid bounds ([0.85,1.15], [-60,60] HU) "
            "are ich_gen.stress_test.make_perturbation_grid()'s own "
            "default, itself documented as provisional pending a specific "
            "CT QA-literature citation the manuscript has not yet selected "
            "(manuscript Section VI-F and the manuscript's own closing "
            "'remaining design parameters' note both state this "
            "explicitly). This result should be reported as provisional "
            "for the same reason, not as a citation-backed final number."
        ),
        "held_out_subset_size_disclosure": (
            f"HOLDOUT_FINAL_N={HOLDOUT_FINAL_N} is this script's own "
            "choice (inference-tractability), not a manuscript-specified "
            "sample size."
        ),
        "configs": {"baseline2": b2_config, "wicl": wicl_config},
    }
    out_path = args.runs_dir / "h4_hu_recalibration_stress_test_result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
