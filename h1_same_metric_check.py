"""Same-metric confirmatory analysis for H1 (manuscript revision, 2026-09-14):
compares POOLED sensitivity relative degradation (baseline 2, RSNA-internal
-> CQ500-external, "any" ICH label) against the already-audited
SUBTYPE-AVERAGED sensitivity relative degradation (0.5778, from
h1_baseline2_degradation_result.json), i.e. both arms of the comparison now
use the identical metric type (sensitivity) and the identical relative-
degradation transform rel_deg(internal,external)=(internal-external)/internal,
removing the AUROC/accuracy-vs-sensitivity metric-type heterogeneity flagged
in the manuscript's Section 3.4 note.

Reuses h1_baseline2_degradation_test_bhsd_excluded.py's exact model-loading,
scoring, resampling, and bootstrap methodology (same checkpoint, same CQ500
build, same seed=0, same N_BOOTSTRAP=1000, same patient-level independent
two-sample resampling) -- only the metric computed is different (adds
pooled "any"-label sensitivity to compute_metrics()).

Usage
-----
    python h1_same_metric_check.py --runs-dir runs/baselines_bhsd_excluded --cq500-root data/cq500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ich_gen.datasets.common import MultiSiteICHDataset, LABEL_COLUMNS, SUBTYPES
from ich_gen.datasets.cq500 import build_samples as build_cq500
from ich_gen.models.backbone import build_backbone
from ich_gen import train as train_mod

ANY_IDX = LABEL_COLUMNS.index("any")
VIEW_MODE_B2 = "fixed_three_window"
N_BOOTSTRAP = 1000
SEED = 0


def load_internal_val_split(config: dict):
    ns = argparse.Namespace(
        rsna_root=Path(config["rsna_root"]),
        max_samples=config["max_samples"],
        n_folds=config["n_folds"],
        seed=config["seed"],
        bhsd_exclusion=True,
    )
    samples, folds = train_mod.build_folds(ns)
    fold = folds[config["fold"]]
    return [samples[i] for i in fold.val_idx]


@torch.no_grad()
def score(model, samples, config, device, batch_size=64):
    ds = MultiSiteICHDataset(samples, view_mode=VIEW_MODE_B2, seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    probs, labels, patients = [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
        patients.extend(batch["patient_id"])
    return np.concatenate(labels), np.concatenate(probs), np.array(patients)


def sensitivity(y_true_bin, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    pos = y_true_bin == 1
    if pos.sum() == 0:
        return float("nan")
    tp = ((y_true_bin == 1) & (pred == 1)).sum()
    return float(tp / pos.sum())


def rel_deg(internal_val, external_val):
    if internal_val is None or np.isnan(internal_val) or internal_val == 0:
        return float("nan")
    return float((internal_val - external_val) / internal_val)


def compute_pooled_and_subtype_sens(y_true, y_prob):
    out = {"sens_any": sensitivity(y_true[:, ANY_IDX], y_prob[:, ANY_IDX])}
    for subtype in SUBTYPES:
        idx = LABEL_COLUMNS.index(subtype)
        out[f"sens_{subtype}"] = sensitivity(y_true[:, idx], y_prob[:, idx])
    return out


def bootstrap_resample_indices(patient_ids, rng):
    unique_patients = np.unique(patient_ids)
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    return np.concatenate([np.where(patient_ids == p)[0] for p in sampled])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b2_dir = args.runs_dir / "baseline2_fixed_three"
    with open(b2_dir / "config.json") as f:
        config = json.load(f)

    ckpt = torch.load(b2_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print("loading RSNA internal held-out fold ...")
    internal_samples = load_internal_val_split(config)
    y_int, p_int, pat_int = score(model, internal_samples, config, device)
    print(f"internal (RSNA fold {config['fold']}): {len(y_int)} slices, "
          f"{len(set(pat_int.tolist()))} patients")

    print("building CQ500 external test set ...")
    cq_samples = build_cq500(args.cq500_root)
    y_ext, p_ext, pat_ext = score(model, cq_samples, config, device)
    print(f"external (CQ500): {len(y_ext)} slices, {len(set(pat_ext.tolist()))} studies")

    m_int = compute_pooled_and_subtype_sens(y_int, p_int)
    m_ext = compute_pooled_and_subtype_sens(y_ext, p_ext)

    pooled_sens_deg = rel_deg(m_int["sens_any"], m_ext["sens_any"])
    subtype_degs = {s: rel_deg(m_int[f"sens_{s}"], m_ext[f"sens_{s}"]) for s in SUBTYPES}
    finite_subtype_degs = [v for v in subtype_degs.values() if not np.isnan(v)]
    stratified_sens_deg = float(np.mean(finite_subtype_degs)) if finite_subtype_degs else float("nan")
    observed_diff = stratified_sens_deg - pooled_sens_deg

    print(f"\n=== SAME-METRIC H1 CHECK (baseline 2, sensitivity only, both arms) ===")
    print(f"internal pooled 'any' sensitivity = {m_int['sens_any']:.4f}  "
          f"external = {m_ext['sens_any']:.4f}  "
          f"pooled sensitivity relative degradation = {pooled_sens_deg:+.4f}")
    print(f"subtype-averaged sensitivity relative degradation (same as audited "
          f"0.5778 figure) = {stratified_sens_deg:+.4f}")
    print(f"difference (stratified - pooled, SAME metric) = {observed_diff:+.4f}")

    print(f"\nrunning {N_BOOTSTRAP}-resample independent two-sample bootstrap ...")
    rng = np.random.RandomState(SEED)
    diffs = np.empty(N_BOOTSTRAP)
    pooled_boot = np.empty(N_BOOTSTRAP)
    stratified_boot = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx_int = bootstrap_resample_indices(pat_int, rng)
        idx_ext = bootstrap_resample_indices(pat_ext, rng)
        mi = compute_pooled_and_subtype_sens(y_int[idx_int], p_int[idx_int])
        me = compute_pooled_and_subtype_sens(y_ext[idx_ext], p_ext[idx_ext])
        pooled_boot[b] = rel_deg(mi["sens_any"], me["sens_any"])
        sub_degs_b = [rel_deg(mi[f"sens_{s}"], me[f"sens_{s}"]) for s in SUBTYPES]
        sub_degs_b = [v for v in sub_degs_b if not np.isnan(v)]
        stratified_boot[b] = float(np.mean(sub_degs_b)) if sub_degs_b else float("nan")
        diffs[b] = stratified_boot[b] - pooled_boot[b]

    diffs_finite = diffs[~np.isnan(diffs)]
    n_dropped = int(np.isnan(diffs).sum())
    if observed_diff >= 0:
        p_value = 2 * min((diffs_finite <= 0).mean(), 0.5)
    else:
        p_value = 2 * min((diffs_finite >= 0).mean(), 0.5)
    p_value = float(min(p_value, 1.0))

    pooled_ci = tuple(np.nanpercentile(pooled_boot, [2.5, 97.5]).tolist())
    stratified_ci = tuple(np.nanpercentile(stratified_boot, [2.5, 97.5]).tolist())

    print(f"\n=== SAME-METRIC H1 RESULT ===")
    print(f"pooled sensitivity degradation:     {pooled_sens_deg:+.4f} 95% CI {pooled_ci}")
    print(f"stratified sensitivity degradation: {stratified_sens_deg:+.4f} 95% CI {stratified_ci}")
    print(f"difference (stratified - pooled):   {observed_diff:+.4f}")
    print(f"two-sided bootstrap p-value: {p_value:.4f} ({n_dropped}/{N_BOOTSTRAP} degenerate dropped)")

    out = {
        "internal_sens": m_int, "external_sens": m_ext,
        "pooled_sensitivity_degradation": pooled_sens_deg,
        "subtype_degradations": subtype_degs,
        "stratified_sensitivity_degradation": stratified_sens_deg,
        "observed_diff_stratified_minus_pooled": observed_diff,
        "bootstrap": {
            "n_bootstrap": N_BOOTSTRAP, "seed": SEED, "n_degenerate_dropped": n_dropped,
            "pooled_degradation_ci": pooled_ci, "stratified_degradation_ci": stratified_ci,
            "p_value": p_value,
        },
        "note": ("Same-metric confirmatory analysis: both pooled and "
                 "stratified degradation computed on SENSITIVITY only "
                 "(the 'any' label for pooled, mean of 5 subtypes for "
                 "stratified), using the identical rel_deg=(internal-"
                 "external)/internal transform and the identical patient-"
                 "level independent two-sample bootstrap as "
                 "h1_baseline2_degradation_test_bhsd_excluded.py, removing "
                 "the AUROC/accuracy-vs-sensitivity metric-type "
                 "heterogeneity of the original H1 pooled-degradation "
                 "composite."),
    }
    out_path = args.runs_dir / "h1_same_metric_sensitivity_only_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
