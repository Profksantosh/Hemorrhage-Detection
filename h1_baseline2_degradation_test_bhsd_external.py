"""H1 BHSD-external-site degradation test for baseline 2 (fixed three-window
compositing, "the strongest conventional baseline").

Uses the BHSD-excluded checkpoint (runs/baselines_bhsd_excluded/baseline2_fixed_three/)
in which RSNA patients whose PatientID tokens overlap with any of the 192
BHSD label_192 volumes were excluded from training fold 0. This makes all
192 BHSD volumes genuinely held-out external test data -- no BHSD patient
contributed to the weight optimisation of this checkpoint.

H1 test (manuscript Section III-C / VI-C / VI-G):
    Test whether subtype-stratified sensitivity degradation (internal RSNA
    held-out fold -> external BHSD) is SIGNIFICANTLY LARGER than pooled
    AUROC/accuracy degradation, via a two-independent-sample paired bootstrap
    (see methodology note below).

METHODOLOGY NOTE -- two-independent-sample bootstrap:
    Identical rationale and implementation as h1_baseline2_degradation_test_bhsd_excluded.py
    (RSNA-internal -> CQ500). See that file's module docstring for the full
    justification of the two-independent-sample design (RSNA and BHSD patients
    resampled independently, 1,000 replicates, same patient-level resampling
    unit as ich_gen.stats). Degradation is relative: (internal - external) / internal.

SIZE-TERCILE STRATIFICATION STATUS (Section IV-E):
    BHSD provides lesion_size_mm3 per slice (derived from volumetric masks,
    see ich_gen/datasets/bhsd.py). However, size-tercile THRESHOLDS are
    supposed to be computed from the RSNA/BHSD training-pool distribution
    and held fixed across sites (bhsd.py docstring, common.assign_size_tercile).
    RSNA training data has NO lesion_size_mm3 (masks unavailable; rsna.py
    returns None for all samples), so training-pool tercile boundaries
    CANNOT be established. Using BHSD test data alone to define thresholds
    would constitute test-set contamination (the tercile boundaries would be
    adapted to the test distribution, not fixed from training as the
    manuscript requires). Therefore: size-tercile stratification is SKIPPED
    for this run and the failure mode is reported explicitly in the output
    JSON. Per-subtype subtype-stratified sensitivity degradation IS computed
    (main H1 clause). Size-tercile raw statistics (min, max, median from
    BHSD) are reported for manuscript author review.

PATIENT-LEVEL BOOTSTRAP NOTE:
    BHSD patient_ids in build_samples() are BHSD_{volume_id} (one volume =
    one "patient" in the bootstrap), except that one real patient contributed
    two volumes (see bhsd.py ONE-VOLUME-PER-PATIENT comment). The bootstrap
    therefore treats 192 volumes as 192 independent patients; this
    overcounts unique patients by 1 but has negligible effect on the
    bootstrap distribution (191 vs 192 resampling units, same direction
    as conservative). Flagged per the "document deviations rather than
    silently assume equivalence" discipline.

Usage
-----
    python h1_baseline2_degradation_test_bhsd_external.py \
        --runs-dir runs/baselines_bhsd_excluded \
        --bhsd-root data/bhsd
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
from ich_gen.datasets.bhsd import build_samples as build_bhsd
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
        bhsd_exclusion=True,  # BHSD/RSNA overlap remediation (2026-09-11) -- must be
                              # explicit for a hand-built Namespace (not from
                              # build_argparser()), which defaults to False.
    )
    samples, folds = train_mod.build_folds(ns)
    fold = folds[config["fold"]]
    return [samples[i] for i in fold.val_idx]


@torch.no_grad()
def score(model, samples, config, device, batch_size=32):
    ds = MultiSiteICHDataset(samples, view_mode=VIEW_MODE_B2, seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        prefetch_factor=2)
    probs, labels, patients, sizes = [], [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
        patients.extend(batch["patient_id"])
        sizes.append(batch["lesion_size_mm3"].numpy())
    return (np.concatenate(labels),
            np.concatenate(probs),
            np.array(patients),
            np.concatenate(sizes))


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
    """Relative degradation: (internal - external) / internal.
    Positive = external performs worse (more degraded)."""
    if internal_val is None or np.isnan(internal_val) or internal_val == 0:
        return float("nan")
    return float((internal_val - external_val) / internal_val)


def compute_metrics(y_true, y_prob):
    """y_true, y_prob: (N, 6) arrays -> dict of pooled AUROC, pooled
    accuracy, and per-subtype sensitivity."""
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
    ap.add_argument("--runs-dir", type=Path, required=True,
                    help="runs/baselines_bhsd_excluded")
    ap.add_argument("--bhsd-root", type=Path, default=Path("data/bhsd"),
                    help="root dir containing bhsd_manifest.csv and label_192/")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b2_dir = args.runs_dir / "baseline2_fixed_three"
    with open(b2_dir / "config.json") as f:
        config = json.load(f)

    # Sanity-check: checkpoint must be from the BHSD-excluded training run.
    assert config.get("bhsd_exclusion", False), (
        f"Checkpoint config in {b2_dir} does not set bhsd_exclusion=True -- "
        "this checkpoint was NOT trained with BHSD patients excluded from RSNA. "
        "Aborting to prevent patient-level leakage."
    )
    print(f"checkpoint: {b2_dir / 'checkpoint_last.pt'}")
    print(f"config bhsd_exclusion={config['bhsd_exclusion']}  "
          f"backbone={config['backbone']}  epochs={config['epochs']}  "
          f"fold={config['fold']}/{config['n_folds']}")

    ckpt = torch.load(b2_dir / "checkpoint_last.pt", map_location=device,
                      weights_only=False)
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # ---- Internal: RSNA held-out fold (bhsd_exclusion=True) ----
    print("\nloading RSNA internal held-out fold (bhsd_exclusion=True) ...")
    internal_samples = load_internal_val_split(config)
    y_int, p_int, pat_int, sz_int = score(model, internal_samples, config, device)
    print(f"internal (RSNA fold {config['fold']}): {len(y_int)} slices, "
          f"{len(set(pat_int.tolist()))} patients")

    # ---- External: BHSD ----
    print(f"\nbuilding BHSD external test set from {args.bhsd_root} ...")
    bhsd_samples = build_bhsd(args.bhsd_root)
    n_bhsd_volumes = len({s.patient_id for s in bhsd_samples})
    print(f"BHSD: {len(bhsd_samples)} slices from {n_bhsd_volumes} volumes "
          f"(note: 191 unique real patients -- one patient contributed 2 volumes)")
    y_ext, p_ext, pat_ext, sz_ext = score(model, bhsd_samples, config, device)
    print(f"scored BHSD: {len(y_ext)} slices, {len(set(pat_ext.tolist()))} patient_ids")

    # ---- Size-tercile assessment ----
    bhsd_sizes_positive = sz_ext[~np.isnan(sz_ext) & (y_ext[:, ANY_IDX] == 1)]
    size_tercile_status = (
        "SKIPPED -- RSNA training pool has no lesion_size_mm3 (masks unavailable; "
        "rsna.py returns None for all samples), so training-distribution tercile "
        "thresholds cannot be established. Using BHSD test data alone to define "
        "thresholds would be test-set contamination. Size-tercile results are absent "
        "from this output. Raw BHSD positive-slice size statistics are reported below."
    )
    bhsd_size_stats = {}
    if len(bhsd_sizes_positive) > 0:
        bhsd_size_stats = {
            "n_positive_slices_with_size": int(len(bhsd_sizes_positive)),
            "min_mm3": float(np.min(bhsd_sizes_positive)),
            "p25_mm3": float(np.percentile(bhsd_sizes_positive, 25)),
            "median_mm3": float(np.median(bhsd_sizes_positive)),
            "p75_mm3": float(np.percentile(bhsd_sizes_positive, 75)),
            "max_mm3": float(np.max(bhsd_sizes_positive)),
            "mean_mm3": float(np.mean(bhsd_sizes_positive)),
        }
    print(f"\n[size-tercile] {size_tercile_status}")
    if bhsd_size_stats:
        print(f"  BHSD positive-slice size (mm3): "
              f"n={bhsd_size_stats['n_positive_slices_with_size']} "
              f"median={bhsd_size_stats['median_mm3']:.1f} "
              f"[{bhsd_size_stats['p25_mm3']:.1f}, {bhsd_size_stats['p75_mm3']:.1f}]")

    # ---- Point-estimate metrics and degradation ----
    internal_metrics = compute_metrics(y_int, p_int)
    external_metrics = compute_metrics(y_ext, p_ext)
    observed = degradation_summary(internal_metrics, external_metrics)
    observed_diff = (observed["stratified_degradation_mean_over_subtypes"]
                     - observed["pooled_degradation"])

    print("\n=== baseline 2 (fixed_three_window): internal RSNA vs external BHSD ===")
    print(f"internal AUROC={internal_metrics['auroc']:.4f}  "
          f"external AUROC={external_metrics['auroc']:.4f}  "
          f"relative degradation={observed['pooled_auroc_degradation']:+.4f}")
    print(f"internal accuracy={internal_metrics['accuracy']:.4f}  "
          f"external accuracy={external_metrics['accuracy']:.4f}  "
          f"relative degradation={observed['pooled_accuracy_degradation']:+.4f}")
    print(f"pooled degradation (mean of AUROC/accuracy relative degradation) = "
          f"{observed['pooled_degradation']:+.4f}")
    print("\nper-subtype sensitivity relative degradation (RSNA -> BHSD):")
    for subtype in SUBTYPES:
        d = observed["subtype_degradations"][subtype]
        if np.isnan(d):
            print(f"  {subtype:<22} (degenerate -- cannot compute)")
        else:
            print(f"  {subtype:<22} internal_sens={internal_metrics[f'sens_{subtype}']:.4f}  "
                  f"external_sens={external_metrics[f'sens_{subtype}']:.4f}  "
                  f"rel_degradation={d:+.4f}")
    print(f"\nstratified degradation (mean over {len(SUBTYPES)} subtypes) = "
          f"{observed['stratified_degradation_mean_over_subtypes']:+.4f}")
    print(f"EDH-specific relative degradation = "
          f"{observed['subtype_degradations']['epidural']:+.4f} "
          f"(H1/H2 priority subtype)")
    print(f"\nobserved H1 test statistic (stratified - pooled degradation) = "
          f"{observed_diff:+.4f}")

    # ---- Two-independent-sample bootstrap ----
    print(f"\nrunning {N_BOOTSTRAP}-resample independent two-sample bootstrap "
          f"(patients resampled within each site independently) ...")
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

    print(f"\n=== H1 TEST RESULT (baseline 2, RSNA-internal -> BHSD-external) ===")
    print(f"pooled degradation:      {observed['pooled_degradation']:+.4f} "
          f"95% CI [{pooled_ci[0]:+.4f}, {pooled_ci[1]:+.4f}]")
    print(f"stratified degradation:  {observed['stratified_degradation_mean_over_subtypes']:+.4f} "
          f"95% CI [{stratified_ci[0]:+.4f}, {stratified_ci[1]:+.4f}]")
    print(f"EDH-specific degradation:{observed['subtype_degradations']['epidural']:+.4f} "
          f"95% CI [{edh_ci[0]:+.4f}, {edh_ci[1]:+.4f}]")
    print(f"difference (stratified - pooled): {observed_diff:+.4f}")
    print(f"two-sided bootstrap p-value: {p_value:.4f} "
          f"({n_dropped}/{N_BOOTSTRAP} degenerate replicates dropped)")
    if p_value < 0.05 and observed_diff > 0:
        print("RESULT: stratified sensitivity degradation IS statistically "
              "significantly LARGER than pooled AUROC/accuracy degradation "
              "-- consistent with H1 (subtype clause; size-tercile clause skipped "
              "-- see size_tercile_status in output JSON).")
    elif p_value < 0.05 and observed_diff < 0:
        print("RESULT: statistically significant, but in the OPPOSITE direction "
              "from H1's prediction (stratified degradation SMALLER than pooled) "
              "-- H1 is NOT supported by this test.")
    else:
        print("RESULT: NOT statistically significant (p >= 0.05) -- H1 is "
              "NOT supported by this BHSD-external-site, subtype-only test.")

    out = {
        "test": "H1 BHSD-external-site degradation test",
        "external_site": "BHSD",
        "checkpoint": str(b2_dir / "checkpoint_last.pt"),
        "bhsd_exclusion_confirmed": True,
        "n_bhsd_volumes": n_bhsd_volumes,
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "observed_degradation": observed,
        "observed_diff_stratified_minus_pooled": observed_diff,
        "bootstrap": {
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "n_degenerate_dropped": n_dropped,
            "pooled_degradation_ci": pooled_ci,
            "stratified_degradation_ci": stratified_ci,
            "edh_degradation_ci": edh_ci,
            "p_value": p_value,
        },
        "size_tercile_status": size_tercile_status,
        "bhsd_size_stats_positive_slices": bhsd_size_stats,
        "h1_result": (
            "CONFIRMED (p < 0.05, stratified > pooled)" if (p_value < 0.05 and observed_diff > 0)
            else ("OPPOSITE DIRECTION (p < 0.05 but stratified < pooled)" if p_value < 0.05
                  else "NOT CONFIRMED (p >= 0.05)")
        ),
        "config": {
            "backbone": config["backbone"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "fold": config["fold"],
            "n_folds": config["n_folds"],
            "max_samples": config["max_samples"],
            "seed": config["seed"],
        },
    }
    out_path = args.runs_dir / "h1_bhsd_external_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
