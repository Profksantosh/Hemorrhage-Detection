"""H3 test (manuscript Section III-C "H3 (calibration)" / Section VI-E
"Calibration analysis (testing H3)" / Table VI): Expected Calibration
Error (ECE, 15-bin) and reliability-diagram bin data, computed for
baseline 2 (fixed_three_window, runs/baselines_v1/baseline2_fixed_three)
and WICL (runs/decisive_v2/decisive_wicl), on RSNA-internal (fold 0
held-out validation split, matching every other post-hoc script in this
repo) and CQ500-external, for the pooled "any ICH" prediction AND
separately for each of the five subtypes (manuscript Section VI-E's
explicit "and, separately, for each subtype" clause) -- both BEFORE and
AFTER temperature scaling fit on the RSNA-internal validation split only
(never re-fit on CQ500, per Section VI-E: "not re-fit per external site,
since re-fitting on external data would not reflect a realistic zero-shot
deployment scenario").

Inference-only: both checkpoints are loaded read-only
(`weights_only=False` torch.load, no optimizer state touched, no
`.train()`/backward pass anywhere in this script). No file under
runs/decisive*, runs/baselines_v1/baseline2_fixed_three, or any
CQ500/RSNA label/leakage-audit file is written to.

SCOPE, stated explicitly (do not treat as the full Section VI-E design):
  - Only CQ500 is used as the external site here, matching the scope of
    every other external-scoring script already audited in this repo
    (h1_baseline2_degradation_test.py, score_baselines3ab_external.py,
    etc.) -- PhysioNet-ICH and BHSD data ARE present under data/ but have
    never been wired into any external-scoring script in this project to
    date; extending H3 to those two sites is future work, not silently
    done here.
  - The manuscript's H3 sentence ("Expected calibration error (ECE) for
    the H1 baseline will increase ... and WICL training will reduce this
    calibration degradation relative to the H1 baseline") and Section
    VI-E's explicit "reported for both baseline 2 and WICL" both name
    exactly this two-condition comparison as the core test -- so this is
    not an arbitrary narrowing of Section VI-E's more general "per
    condition" framing, it is the specific comparison H3 itself names.
  - Reliability "diagrams" are saved as the underlying per-bin
    (confidence, accuracy, count) arrays (JSON), not rendered as image
    plots -- matplotlib is not in requirements.txt and this script
    intentionally avoids adding a new dependency for a post-hoc,
    inference-only analysis; the bin data is sufficient to render the
    diagram separately.
  - JUDGMENT CALL, disclosed: the manuscript does not specify whether
    temperature scaling fits one scalar T per label or one T shared
    across all six output columns ("applied uniformly at test time" is
    stated but is ambiguous between "same T at every site" and "same T
    across every output dimension"). This script fits ONE scalar T per
    condition on the pooled "any ICH" internal-validation logits only
    (the manuscript's stated primary calibration target, and the
    standard single-scalar formulation of Guo et al. 2017 temperature
    scaling), and applies that same T to rescale logits for the pooled
    prediction AND all five subtype columns alike, at both sites. This
    operationalization is stated here rather than assumed silently.

Usage
-----
    python h3_calibration_ece_test.py --runs-dir runs/baselines_v1 \
        --wicl-dir runs/decisive_v2/decisive_wicl --cq500-root data/cq500
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
from ich_gen.datasets.cq500 import build_samples as build_cq500
from ich_gen.models.backbone import build_backbone
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stats import expected_calibration_error
from ich_gen import train as train_mod

ANY_IDX = LABEL_COLUMNS.index("any")
N_BINS = 15
N_BOOTSTRAP = 1000
SEED = 0


# ---------------------------------------------------------------------------
# data / scoring (mirrors h1_baseline2_degradation_test.py's conventions)
# ---------------------------------------------------------------------------

def load_internal_val_split(config: dict):
    ns = argparse.Namespace(
        rsna_root=Path(config["rsna_root"]),
        max_samples=config["max_samples"],
        n_folds=config["n_folds"],
        seed=config["seed"],
    )
    samples, folds = train_mod.build_folds(ns)
    fold = folds[config["fold"]]
    return [samples[i] for i in fold.val_idx]


@torch.no_grad()
def score_logits(model, view_mode, forward_fn, samples, config, device,
                  batch_size=64, num_workers=4):
    """Returns (logits, labels, patient_ids), logits BEFORE sigmoid --
    needed for temperature scaling, unlike the plain-probability score()
    helpers used by other post-hoc scripts in this repo."""
    ds = MultiSiteICHDataset(samples, view_mode=view_mode, seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)
    logits_all, labels_all, patients = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        logits = forward_fn(model, x)
        logits_all.append(logits.cpu().numpy())
        labels_all.append(batch["labels"].numpy())
        patients.extend(batch["patient_id"])
    return (np.concatenate(logits_all), np.concatenate(labels_all),
            np.array(patients))


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
    return model, config, "fixed_three_window", forward_fn


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
    # manuscript Section V-E: WICL is evaluated with the fixed canonical
    # three-window composite at test time, single view, forward_single().
    forward_fn = lambda m, x: m.forward_single(x)
    return model, config, "fixed_three_window", forward_fn


# ---------------------------------------------------------------------------
# temperature scaling (Guo et al. 2017, single scalar T; see module
# docstring for the disclosed "fit on pooled 'any' only, apply uniformly
# to all 6 columns" judgment call)
# ---------------------------------------------------------------------------

def _bce_with_logits_nll(z: np.ndarray, y: np.ndarray) -> float:
    z = z.astype(np.float64)
    y = y.astype(np.float64)
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature(logits_any: np.ndarray, labels_any: np.ndarray) -> dict:
    """Grid search over T (deterministic, no optimizer-convergence
    ambiguity) minimizing NLL of sigmoid(logit / T) against labels_any,
    on the RSNA-internal validation split only."""
    t_grid = np.concatenate([np.linspace(0.05, 1.0, 96)[:-1],
                              np.linspace(1.0, 5.0, 161)])
    nlls = np.array([_bce_with_logits_nll(logits_any / t, labels_any) for t in t_grid])
    best_idx = int(np.argmin(nlls))
    nll_at_1 = _bce_with_logits_nll(logits_any, labels_any)
    return {"T": float(t_grid[best_idx]), "nll_at_T": float(nlls[best_idx]),
            "nll_at_T1_uncalibrated": nll_at_1}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def safe_auroc(y_true_bin, y_score):
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_bin, y_score))


def rel_deg(internal_val, external_val):
    """Same convention as h1_baseline2_degradation_test.py's rel_deg:
    (internal - external) / internal, positive = external worse."""
    if internal_val is None or np.isnan(internal_val) or internal_val == 0:
        return float("nan")
    return float((internal_val - external_val) / internal_val)


def ece_rel_deg(ece_internal, ece_external):
    """ECE-specific degradation convention (NOTE: opposite numerator sign
    from rel_deg's AUROC convention, since for ECE *higher* = *worse*,
    while for AUROC *lower* = *worse*): (external - internal) / internal,
    positive = calibration got WORSE (ECE increased) externally."""
    if ece_internal is None or np.isnan(ece_internal) or ece_internal == 0:
        return float("nan")
    return float((ece_external - ece_internal) / ece_internal)


def per_column_ece(y_true, y_prob, n_bins=N_BINS):
    """dict: 'any' + each subtype -> CalibrationResult-derived dict."""
    out = {}
    for name in LABEL_COLUMNS:
        idx = LABEL_COLUMNS.index(name)
        cal = expected_calibration_error(y_true[:, idx], y_prob[:, idx], n_bins=n_bins)
        out[name] = {"ece": cal.ece, "bin_accuracy": cal.bin_accuracy.tolist(),
                      "bin_confidence": cal.bin_confidence.tolist(),
                      "bin_count": cal.bin_count.tolist()}
    return out


def bootstrap_resample_indices(patient_ids, rng):
    unique_patients = np.unique(patient_ids)
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    return np.concatenate([np.where(patient_ids == p)[0] for p in sampled])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/baselines_v1"))
    ap.add_argument("--wicl-dir", type=Path, default=Path("runs/decisive_v2/decisive_wicl"))
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading baseline2 (fixed_three_window) and WICL checkpoints "
          "(read-only) ...")
    b2_model, b2_config, b2_view, b2_fwd = load_baseline2(args.runs_dir, device)
    wicl_model, wicl_config, wicl_view, wicl_fwd = load_wicl(args.wicl_dir, device)
    assert b2_config["fold"] == wicl_config["fold"], \
        "baseline2 and WICL must share the same RSNA fold for a fair internal comparison"

    print("loading RSNA-internal held-out fold (fold %d) ..." % b2_config["fold"])
    internal_samples = load_internal_val_split(b2_config)
    print(f"  {len(internal_samples)} slices, "
          f"{len({s.patient_id for s in internal_samples})} patients")

    print("building CQ500 external test set ...")
    cq_samples = build_cq500(args.cq500_root)
    print(f"  {len(cq_samples)} slices, "
          f"{len({s.patient_id for s in cq_samples})} studies")

    conditions = {
        "baseline2": (b2_model, b2_view, b2_fwd, b2_config),
        "wicl": (wicl_model, wicl_view, wicl_fwd, wicl_config),
    }

    scored = {}  # scored[cond][site] = dict(logits, labels, patients)
    for cond, (model, view_mode, fwd, config) in conditions.items():
        scored[cond] = {}
        for site, samples in (("rsna_internal", internal_samples),
                               ("cq500_external", cq_samples)):
            print(f"scoring {cond} on {site} ...")
            logits, labels, patients = score_logits(model, view_mode, fwd,
                                                      samples, config, device)
            scored[cond][site] = {"logits": logits, "labels": labels,
                                    "patients": patients}

    # sample-ordering/label sanity check between conditions on each site
    # (both models scored on the identical held-out samples, same order)
    for site in ("rsna_internal", "cq500_external"):
        assert np.allclose(scored["baseline2"][site]["labels"],
                            scored["wicl"][site]["labels"]), \
            f"label mismatch between baseline2 and wicl on {site}"

    # --- temperature scaling: fit on RSNA-internal pooled 'any' logits only ---
    temperature = {}
    for cond in ("baseline2", "wicl"):
        li = scored[cond]["rsna_internal"]["logits"][:, ANY_IDX]
        yi = scored[cond]["rsna_internal"]["labels"][:, ANY_IDX]
        temperature[cond] = fit_temperature(li, yi)
        print(f"\ntemperature scaling fit ({cond}, RSNA-internal, pooled "
              f"'any' only): T={temperature[cond]['T']:.3f}  "
              f"NLL(T)={temperature[cond]['nll_at_T']:.4f}  "
              f"NLL(T=1, uncalibrated)={temperature[cond]['nll_at_T1_uncalibrated']:.4f}")

    # --- ECE, pre- and post-temperature-scaling, pooled + per-subtype ---
    results = {"temperature_scaling": temperature, "per_site_per_condition": {}}
    ece_summary = {}  # for the degradation/statistical tests below
    for cond in ("baseline2", "wicl"):
        results["per_site_per_condition"][cond] = {}
        ece_summary[cond] = {}
        T = temperature[cond]["T"]
        for site in ("rsna_internal", "cq500_external"):
            logits = scored[cond][site]["logits"]
            labels = scored[cond][site]["labels"]
            probs_raw = sigmoid(logits)
            probs_T = sigmoid(logits / T)

            auroc_any = safe_auroc(labels[:, ANY_IDX], probs_raw[:, ANY_IDX])
            ece_raw = per_column_ece(labels, probs_raw)
            ece_post_T = per_column_ece(labels, probs_T)

            results["per_site_per_condition"][cond][site] = {
                "n_samples": int(len(labels)),
                "n_patients": int(len(np.unique(scored[cond][site]["patients"]))),
                "pooled_auroc_any": auroc_any,
                "ece_pre_temperature_scaling": ece_raw,
                "ece_post_temperature_scaling": ece_post_T,
            }
            ece_summary[cond][site] = {
                "auroc_any": auroc_any,
                "ece_any_pre_T": ece_raw["any"]["ece"],
                "ece_any_post_T": ece_post_T["any"]["ece"],
            }

            print(f"\n[{cond} | {site}] n={len(labels)}  AUROC(any)={auroc_any:.4f}  "
                  f"ECE(any, pre-T)={ece_raw['any']['ece']:.4f}  "
                  f"ECE(any, post-T)={ece_post_T['any']['ece']:.4f}")
            for subtype in SUBTYPES:
                print(f"    {subtype:<18} ECE(pre-T)={ece_raw[subtype]['ece']:.4f}  "
                      f"ECE(post-T)={ece_post_T[subtype]['ece']:.4f}")

    # --- H3 test (a): does baseline2's pooled ECE degrade internal->external
    # MORE than AUROC degradation would predict? (pre-temperature-scaling,
    # since this tests the RAW baseline's calibration behavior under site
    # transfer, matching H3's "for the H1 baseline" wording literally) ---
    print("\nrunning two-independent-sample bootstrap "
          f"({N_BOOTSTRAP} resamples, patients resampled independently "
          "within RSNA-internal and CQ500, mirroring "
          "h1_baseline2_degradation_test.py's methodology) ...")

    def compute_replicate(cond, idx_int, idx_ext):
        l_int = scored[cond]["rsna_internal"]
        l_ext = scored[cond]["cq500_external"]
        y_int, p_int = l_int["labels"][idx_int], sigmoid(l_int["logits"][idx_int])
        y_ext, p_ext = l_ext["labels"][idx_ext], sigmoid(l_ext["logits"][idx_ext])
        auroc_int = safe_auroc(y_int[:, ANY_IDX], p_int[:, ANY_IDX])
        auroc_ext = safe_auroc(y_ext[:, ANY_IDX], p_ext[:, ANY_IDX])
        ece_int = expected_calibration_error(y_int[:, ANY_IDX], p_int[:, ANY_IDX], N_BINS).ece
        ece_ext = expected_calibration_error(y_ext[:, ANY_IDX], p_ext[:, ANY_IDX], N_BINS).ece
        return rel_deg(auroc_int, auroc_ext), ece_rel_deg(ece_int, ece_ext), ece_int, ece_ext

    rng = np.random.RandomState(SEED)
    n_int = len(scored["baseline2"]["rsna_internal"]["patients"])
    n_ext = len(scored["baseline2"]["cq500_external"]["patients"])
    pat_int_b2 = scored["baseline2"]["rsna_internal"]["patients"]
    pat_ext_b2 = scored["baseline2"]["cq500_external"]["patients"]

    # observed (point) values
    obs_auroc_deg_b2, obs_ece_deg_b2, obs_ece_int_b2, obs_ece_ext_b2 = compute_replicate(
        "baseline2", np.arange(n_int), np.arange(n_ext))
    obs_auroc_deg_w, obs_ece_deg_w, obs_ece_int_w, obs_ece_ext_w = compute_replicate(
        "wicl", np.arange(n_int), np.arange(n_ext))

    diff_ece_vs_auroc_b2 = np.empty(N_BOOTSTRAP)         # H3a: ece_deg - auroc_deg, baseline2
    diff_wicl_minus_b2_ece_deg = np.empty(N_BOOTSTRAP)   # H3b: b2_ece_deg - wicl_ece_deg (>0 => WICL degrades less)
    b2_ext_ece_boot = np.empty(N_BOOTSTRAP)
    wicl_ext_ece_boot = np.empty(N_BOOTSTRAP)
    b2_int_ece_boot = np.empty(N_BOOTSTRAP)
    wicl_int_ece_boot = np.empty(N_BOOTSTRAP)

    for b in range(N_BOOTSTRAP):
        idx_int = bootstrap_resample_indices(pat_int_b2, rng)
        idx_ext = bootstrap_resample_indices(pat_ext_b2, rng)
        a_deg_b2, e_deg_b2, e_int_b2, e_ext_b2 = compute_replicate("baseline2", idx_int, idx_ext)
        a_deg_w, e_deg_w, e_int_w, e_ext_w = compute_replicate("wicl", idx_int, idx_ext)
        diff_ece_vs_auroc_b2[b] = e_deg_b2 - a_deg_b2
        diff_wicl_minus_b2_ece_deg[b] = e_deg_b2 - e_deg_w
        b2_ext_ece_boot[b] = e_ext_b2
        wicl_ext_ece_boot[b] = e_ext_w
        b2_int_ece_boot[b] = e_int_b2
        wicl_int_ece_boot[b] = e_int_w

    def two_sided_p(diffs, observed_diff):
        diffs = diffs[~np.isnan(diffs)]
        if len(diffs) == 0:
            return float("nan"), 0
        if observed_diff >= 0:
            p = 2 * min((diffs <= 0).mean(), 0.5)
        else:
            p = 2 * min((diffs >= 0).mean(), 0.5)
        n_dropped = int(np.isnan(diffs).sum())
        return float(min(p, 1.0)), n_dropped

    observed_diff_a = obs_ece_deg_b2 - obs_auroc_deg_b2
    p_h3a, n_drop_a = two_sided_p(diff_ece_vs_auroc_b2, observed_diff_a)

    observed_diff_b = obs_ece_deg_b2 - obs_ece_deg_w
    p_h3b, n_drop_b = two_sided_p(diff_wicl_minus_b2_ece_deg, observed_diff_b)

    ci_b2_ext = tuple(np.nanpercentile(b2_ext_ece_boot, [2.5, 97.5]).tolist())
    ci_wicl_ext = tuple(np.nanpercentile(wicl_ext_ece_boot, [2.5, 97.5]).tolist())
    ci_b2_int = tuple(np.nanpercentile(b2_int_ece_boot, [2.5, 97.5]).tolist())
    ci_wicl_int = tuple(np.nanpercentile(wicl_int_ece_boot, [2.5, 97.5]).tolist())

    print("\n=== H3 TEST (a): baseline2 pooled-ECE relative degradation "
          "(internal->CQ500) vs. pooled-AUROC relative degradation ===")
    print(f"AUROC relative degradation:            {obs_auroc_deg_b2:+.4f}")
    print(f"ECE relative degradation (higher=worse): {obs_ece_deg_b2:+.4f}")
    print(f"difference (ECE_deg - AUROC_deg):       {observed_diff_a:+.4f}")
    print(f"two-sided bootstrap p-value: {p_h3a:.4f} "
          f"({n_drop_a}/{N_BOOTSTRAP} degenerate replicates dropped)")
    if p_h3a < 0.05 and observed_diff_a > 0:
        verdict_a = ("ECE degrades SIGNIFICANTLY MORE than AUROC degradation "
                      "would predict -- consistent with H3's first clause.")
    elif p_h3a < 0.05:
        verdict_a = ("statistically significant but in the OPPOSITE direction "
                      "(ECE degradation smaller than AUROC-predicted) -- H3's "
                      "first clause is NOT supported.")
    else:
        verdict_a = "NOT statistically significant (p >= 0.05)."
    print(f"RESULT: {verdict_a}")

    print("\n=== H3 TEST (b): does WICL reduce external calibration "
          "degradation relative to baseline2? (pre-temperature-scaling) ===")
    print(f"baseline2 ECE relative degradation: {obs_ece_deg_b2:+.4f}")
    print(f"WICL ECE relative degradation:      {obs_ece_deg_w:+.4f}")
    print(f"difference (baseline2_deg - wicl_deg), >0 => WICL degrades less: "
          f"{observed_diff_b:+.4f}")
    print(f"two-sided bootstrap p-value: {p_h3b:.4f} "
          f"({n_drop_b}/{N_BOOTSTRAP} degenerate replicates dropped)")
    if p_h3b < 0.05 and observed_diff_b > 0:
        verdict_b = ("WICL's external ECE degradation IS significantly "
                      "smaller than baseline2's -- consistent with H3's "
                      "second clause.")
    elif p_h3b < 0.05:
        verdict_b = ("statistically significant but in the OPPOSITE "
                      "direction (WICL degrades MORE than baseline2) -- "
                      "H3's second clause is NOT supported.")
    else:
        verdict_b = "NOT statistically significant (p >= 0.05)."
    print(f"RESULT: {verdict_b}")

    print("\n=== supplementary: post-temperature-scaling external ECE, "
          "baseline2 vs WICL (tests whether WICL adds calibration benefit "
          "beyond post-hoc temperature scaling alone, per Section VI-E) ===")
    print(f"baseline2: pre-T={obs_ece_ext_b2:.4f}  post-T={ece_summary['baseline2']['cq500_external']['ece_any_post_T']:.4f}")
    print(f"wicl:      pre-T={obs_ece_ext_w:.4f}  post-T={ece_summary['wicl']['cq500_external']['ece_any_post_T']:.4f}")

    results["h3_test_a_ece_deg_vs_auroc_deg"] = {
        "condition": "baseline2", "auroc_relative_degradation": obs_auroc_deg_b2,
        "ece_relative_degradation": obs_ece_deg_b2, "observed_diff": observed_diff_a,
        "n_bootstrap": N_BOOTSTRAP, "p_value": p_h3a, "n_degenerate_dropped": n_drop_a,
        "verdict": verdict_a,
    }
    results["h3_test_b_wicl_vs_baseline2_ece_degradation"] = {
        "baseline2_ece_relative_degradation": obs_ece_deg_b2,
        "wicl_ece_relative_degradation": obs_ece_deg_w,
        "observed_diff_b2_minus_wicl": observed_diff_b,
        "n_bootstrap": N_BOOTSTRAP, "p_value": p_h3b, "n_degenerate_dropped": n_drop_b,
        "verdict": verdict_b,
    }
    results["ece_bootstrap_cis"] = {
        "baseline2_rsna_internal_ece_any": {"point": obs_ece_int_b2, "ci95": ci_b2_int},
        "baseline2_cq500_external_ece_any": {"point": obs_ece_ext_b2, "ci95": ci_b2_ext},
        "wicl_rsna_internal_ece_any": {"point": obs_ece_int_w, "ci95": ci_wicl_int},
        "wicl_cq500_external_ece_any": {"point": obs_ece_ext_w, "ci95": ci_wicl_ext},
    }
    results["scope_limitation"] = (
        "Only CQ500 (one of the three Section IV-B external sites) was "
        "used as the external site, matching the scope of every other "
        "post-hoc external-scoring script already audited in this repo "
        "(PhysioNet-ICH and BHSD are present under data/ but have never "
        "been wired into any external-scoring script for any condition in "
        "this project). Only baseline2 and WICL are compared, matching "
        "H3's own and Section VI-E's explicit 'reported for both baseline "
        "2 and WICL' framing -- Section VI-E's broader 'per condition' "
        "commitment (i.e. computing ECE for baselines 1, 3a/3b, 4a/4b/4c, "
        "5 too) is NOT covered by this script."
    )
    results["temperature_scaling_judgment_call"] = (
        "Manuscript Section VI-E does not specify whether temperature "
        "scaling fits one scalar per label or one scalar shared across "
        "all six output columns. This script fits ONE scalar T per "
        "condition on the pooled 'any ICH' internal-validation logits "
        "only (Guo et al. 2017's standard single-scalar formulation, "
        "applied to the manuscript's stated primary calibration target), "
        "then applies that same T to rescale logits for the pooled AND "
        "all five subtype columns alike, at both sites. This is disclosed "
        "here as an operationalization choice, not silently assumed."
    )
    results["configs"] = {"baseline2": b2_config, "wicl": wicl_config}

    out_path = args.runs_dir / "h3_calibration_ece_result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
