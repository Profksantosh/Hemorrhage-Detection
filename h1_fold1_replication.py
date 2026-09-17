"""H1 second-fold replication (manuscript revision, 2026-09-14): re-runs
h1_baseline2_degradation_test_bhsd_excluded.py's exact methodology (same
rel_deg transform, same patient-level independent two-sample bootstrap,
same N_BOOTSTRAP=1000, seed=0) against the fold-1 baseline-2 checkpoint
(runs/baselines_bhsd_excluded/baseline2_fixed_three_fold1, trained with
--fold 1 instead of fold 0, same protocol otherwise) to check whether the
primary H1 CQ500 result is specific to fold 0.

Only functional difference from the audited script: the baseline-2
checkpoint subdirectory is a CLI argument (--b2-subdir) instead of the
hardcoded "baseline2_fixed_three", and load_internal_val_split() uses
config["fold"] (read from the checkpoint's own config.json, which will
correctly say fold=1) exactly as the original script already does -- no
other methodological change.

Usage
-----
    python h1_fold1_replication.py --runs-dir runs/baselines_bhsd_excluded \
        --b2-subdir baseline2_fixed_three_fold1 --cq500-root data/cq500
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


def safe_auroc(y_true_bin, y_score):
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_bin, y_score))


def accuracy(y_true_bin, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    return float((pred == y_true_bin).mean())


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


def compute_metrics(y_true, y_prob):
    out = {
        "auroc": safe_auroc(y_true[:, ANY_IDX], y_prob[:, ANY_IDX]),
        "accuracy": accuracy(y_true[:, ANY_IDX], y_prob[:, ANY_IDX]),
    }
    for subtype in SUBTYPES:
        idx = LABEL_COLUMNS.index(subtype)
        out[f"sens_{subtype}"] = sensitivity(y_true[:, idx], y_prob[:, idx])
    return out


def degradation_summary(internal_metrics, external_metrics):
    pooled_auroc_deg = rel_deg(internal_metrics["auroc"], external_metrics["auroc"])
    pooled_acc_deg = rel_deg(internal_metrics["accuracy"], external_metrics["accuracy"])
    pooled_components = [d for d in (pooled_auroc_deg, pooled_acc_deg) if not np.isnan(d)]
    pooled_deg = float(np.mean(pooled_components)) if pooled_components else float("nan")

    subtype_degs = {}
    for subtype in SUBTYPES:
        key = f"sens_{subtype}"
        subtype_degs[subtype] = rel_deg(internal_metrics[key], external_metrics[key])
    finite_subtype_degs = [v for v in subtype_degs.values() if not np.isnan(v)]
    stratified_deg = float(np.mean(finite_subtype_degs)) if finite_subtype_degs else float("nan")

    return {
        "pooled_auroc_degradation": pooled_auroc_deg,
        "pooled_accuracy_degradation": pooled_acc_deg,
        "pooled_degradation": pooled_deg,
        "subtype_degradations": subtype_degs,
        "stratified_degradation_mean_over_subtypes": stratified_deg,
    }


def bootstrap_resample_indices(patient_ids, rng):
    unique_patients = np.unique(patient_ids)
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    return np.concatenate([np.where(patient_ids == p)[0] for p in sampled])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--b2-subdir", type=str, default="baseline2_fixed_three_fold1")
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b2_dir = args.runs_dir / args.b2_subdir
    with open(b2_dir / "config.json") as f:
        config = json.load(f)
    print(f"fold-1 replication: loading checkpoint from {b2_dir} (config fold={config['fold']})")
    assert config["fold"] == 1, f"expected fold=1 checkpoint, got fold={config['fold']}"

    ckpt = torch.load(b2_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    print("loading RSNA internal held-out fold 1 ...")
    internal_samples = load_internal_val_split(config)
    y_int, p_int, pat_int = score(model, internal_samples, config, device)
    print(f"internal (RSNA fold {config['fold']}): {len(y_int)} slices, "
          f"{len(set(pat_int.tolist()))} patients")

    print("building CQ500 external test set ...")
    cq_samples = build_cq500(args.cq500_root)
    y_ext, p_ext, pat_ext = score(model, cq_samples, config, device)
    print(f"external (CQ500): {len(y_ext)} slices, {len(set(pat_ext.tolist()))} studies")

    internal_metrics = compute_metrics(y_int, p_int)
    external_metrics = compute_metrics(y_ext, p_ext)
    observed = degradation_summary(internal_metrics, external_metrics)
    observed_diff = (observed["stratified_degradation_mean_over_subtypes"]
                      - observed["pooled_degradation"])

    print("\n=== baseline 2 FOLD 1 (fixed_three_window): internal RSNA vs external CQ500 ===")
    print(f"internal AUROC={internal_metrics['auroc']:.4f}  "
          f"external AUROC={external_metrics['auroc']:.4f}  "
          f"relative degradation={observed['pooled_auroc_degradation']:+.4f}")
    print(f"internal accuracy={internal_metrics['accuracy']:.4f}  "
          f"external accuracy={external_metrics['accuracy']:.4f}  "
          f"relative degradation={observed['pooled_accuracy_degradation']:+.4f}")
    print(f"pooled degradation (mean of AUROC/accuracy relative degradation) = "
          f"{observed['pooled_degradation']:+.4f}")
    print("\nper-subtype sensitivity relative degradation:")
    for subtype in SUBTYPES:
        d = observed["subtype_degradations"][subtype]
        print(f"  {subtype:<18} internal_sens={internal_metrics[f'sens_{subtype}']:.4f}  "
              f"external_sens={external_metrics[f'sens_{subtype}']:.4f}  "
              f"rel_degradation={d:+.4f}" if not np.isnan(d) else
              f"  {subtype:<18} (degenerate -- cannot compute)")
    print(f"\nstratified degradation (mean over 5 subtypes) = "
          f"{observed['stratified_degradation_mean_over_subtypes']:+.4f}")
    print(f"EDH-specific relative degradation = "
          f"{observed['subtype_degradations']['epidural']:+.4f} (H1/H2 priority subtype)")
    print(f"\nobserved H1 test statistic (stratified - pooled degradation) = "
          f"{observed_diff:+.4f}")

    print(f"\nrunning {N_BOOTSTRAP}-resample independent two-sample bootstrap ...")
    rng = np.random.RandomState(SEED)
    diffs = np.empty(N_BOOTSTRAP)
    pooled_boot = np.empty(N_BOOTSTRAP)
    stratified_boot = np.empty(N_BOOTSTRAP)
    edh_boot = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx_int = bootstrap_resample_indices(pat_int, rng)
        idx_ext = bootstrap_resample_indices(pat_ext, rng)
        m_int_b = compute_metrics(y_int[idx_int], p_int[idx_int])
        m_ext_b = compute_metrics(y_ext[idx_ext], p_ext[idx_ext])
        deg_b = degradation_summary(m_int_b, m_ext_b)
        pooled_boot[b] = deg_b["pooled_degradation"]
        stratified_boot[b] = deg_b["stratified_degradation_mean_over_subtypes"]
        edh_boot[b] = deg_b["subtype_degradations"]["epidural"]
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
    edh_ci = tuple(np.nanpercentile(edh_boot, [2.5, 97.5]).tolist())

    print(f"\n=== H1 FOLD-1 REPLICATION RESULT (baseline 2, RSNA-internal -> CQ500-external) ===")
    print(f"pooled degradation:      {observed['pooled_degradation']:+.4f} 95% CI {pooled_ci}")
    print(f"stratified degradation:  {observed['stratified_degradation_mean_over_subtypes']:+.4f} "
          f"95% CI {stratified_ci}")
    print(f"EDH-specific degradation:{observed['subtype_degradations']['epidural']:+.4f} 95% CI {edh_ci}")
    print(f"difference (stratified - pooled): {observed_diff:+.4f}")
    print(f"two-sided bootstrap p-value: {p_value:.4f} "
          f"({n_dropped}/{N_BOOTSTRAP} degenerate replicates dropped)")

    out = {
        "fold": config["fold"],
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "observed_degradation": observed,
        "observed_diff_stratified_minus_pooled": observed_diff,
        "bootstrap": {
            "n_bootstrap": N_BOOTSTRAP, "seed": SEED, "n_degenerate_dropped": n_dropped,
            "pooled_degradation_ci": pooled_ci, "stratified_degradation_ci": stratified_ci,
            "edh_degradation_ci": edh_ci, "p_value": p_value,
        },
        "config": {"backbone": config["backbone"], "epochs": config["epochs"],
                    "batch_size": config["batch_size"], "lr": config["lr"],
                    "weight_decay": config["weight_decay"], "fold": config["fold"],
                    "n_folds": config["n_folds"], "max_samples": config["max_samples"],
                    "seed": config["seed"]},
    }
    out_path = args.runs_dir / "h1_baseline2_degradation_fold1_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
