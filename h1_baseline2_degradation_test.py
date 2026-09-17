"""H1 test (manuscript Section III-C / VI-C / VI-G): for baseline 2 (fixed
three-window compositing, "the strongest conventional baseline") only, test
whether subtype-stratified sensitivity degradation (internal RSNA held-out
fold -> external CQ500) is SIGNIFICANTLY LARGER than pooled AUROC/accuracy
degradation, via a paired bootstrap.

Section VI-C's exact wording: "H1 is tested by computing, for baseline 2
... only, the relative degradation (internal -> each external site)
separately for pooled AUROC/accuracy versus subtype- and size-stratified
sensitivity, and testing whether the stratified degradation is
significantly larger via a paired bootstrap (Section VI-G)."

SCOPE LIMITATION (explicitly flagged, not silently omitted): this run only
has CQ500 staged among the three external sites specified in Section IV-B,
and CQ500 carries no lesion_size_mm3 (study-level labels only -- see
score_baselines_external.py's docstring; BHSD is this manuscript's
size-stratification source per Section IV-E and is out of scope for this
task). This script therefore tests H1's SUBTYPE-stratified clause only,
against ONE external site (CQ500), not the full "three external sites x
five subtypes x three size terciles" design. The size-tercile clause of H1
and the PhysioNet-ICH/BHSD external-site clauses remain untested pending
those runs.

METHODOLOGY NOTE (necessary generalization of
ich_gen.stats.paired_bootstrap_test): that function's "paired" bootstrap
assumes the SAME patients appear in both arms being compared (e.g. WICL vs
baseline 4c scored on the identical held-out set -- see score_decisive.py).
H1 instead compares two DIFFERENT patient populations -- the RSNA-internal
held-out fold and the CQ500 external site -- so there is no shared
patient_id to pair samples on. No existing function in ich_gen.stats
handles a two-INDEPENDENT-sample paired bootstrap, so this script
implements one directly, following the same design as
ich_gen.stats.patient_level_bootstrap_ci / paired_bootstrap_test as closely
as the two-independent-sample setting allows: RSNA-internal patients and
CQ500 patients are EACH resampled WITH replacement, INDEPENDENTLY, 1,000
times (same resampling unit -- patients, not slices -- and same
1,000-resample count as ich_gen.stats, Section VI-G). On each joint
replicate we recompute (a) pooled AUROC and accuracy degradation and (b)
subtype-stratified sensitivity degradation, then take their difference. The
two-sided percentile-bootstrap p-value (fraction of replicates on the
opposite side of zero from the observed difference) follows exactly the
same rule as ich_gen.stats.paired_bootstrap_test, just applied to this
between-site comparison. This is a documented methodological decision,
flagged per the manuscript's own "document deviations rather than silently
assume equivalence" discipline -- not a reuse of an existing, already
audited function.

Degradation is defined as RELATIVE degradation, matching H1's own wording:
    rel_deg(internal, external) = (internal - external) / internal
so positive = external performs worse (degradation), matching the
direction "from internal to external test sites" in H1's statement.
Slice-level CQ500 predictions are used throughout (both for the pooled and
the stratified components of this test), matching score_external.py's
documented convention (CQ500 has no per-slice ground truth, only
study-level labels propagated down) and avoiding an aggregation-level
mismatch between the pooled and stratified halves of the comparison.

Usage
-----
    python h1_baseline2_degradation_test.py --runs-dir runs/baselines_v1 --cq500-root data/cq500
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
    # NOTE: returned as (labels, probs, patients) to match the (y_true,
    # y_prob) convention used by compute_metrics() and every call site
    # below -- an earlier version of this function returned (probs,
    # labels, patients) while call sites unpacked it as (y_true, y_prob,
    # patient_ids), silently swapping the two and feeding continuous
    # probabilities into roc_auc_score's y_true argument (caught by the
    # resulting "continuous format is not supported" crash on the very
    # first compute_metrics() call, not silently -- but documented here
    # per the "flag deviations/bugs rather than silently patch" discipline).
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
    """H1's 'relative degradation': (internal - external) / internal.
    Positive = external performs worse than internal."""
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

    internal_metrics = compute_metrics(y_int, p_int)
    external_metrics = compute_metrics(y_ext, p_ext)
    observed = degradation_summary(internal_metrics, external_metrics)
    observed_diff = (observed["stratified_degradation_mean_over_subtypes"]
                      - observed["pooled_degradation"])

    print("\n=== baseline 2 (fixed_three_window): internal RSNA vs external CQ500 ===")
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
          f"{observed['subtype_degradations']['epidural']:+.4f} "
          f"(H1/H2 priority subtype)")
    print(f"\nobserved H1 test statistic (stratified - pooled degradation) = "
          f"{observed_diff:+.4f}")

    # --- two-independent-sample bootstrap (see module docstring) ---
    print(f"\nrunning {N_BOOTSTRAP}-resample independent two-sample bootstrap "
          f"(patients resampled within each site, RSNA-internal and CQ500 "
          f"resampled independently) ...")
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

    print(f"\n=== H1 TEST RESULT (baseline 2, RSNA-internal -> CQ500-external) ===")
    print(f"pooled degradation:      {observed['pooled_degradation']:+.4f} "
          f"95% CI {pooled_ci}")
    print(f"stratified degradation:  {observed['stratified_degradation_mean_over_subtypes']:+.4f} "
          f"95% CI {stratified_ci}")
    print(f"EDH-specific degradation:{observed['subtype_degradations']['epidural']:+.4f} "
          f"95% CI {edh_ci}")
    print(f"difference (stratified - pooled): {observed_diff:+.4f}")
    print(f"two-sided bootstrap p-value: {p_value:.4f} "
          f"({n_dropped}/{N_BOOTSTRAP} degenerate replicates dropped)")
    if p_value < 0.05 and observed_diff > 0:
        print("RESULT: stratified sensitivity degradation IS statistically "
              "significantly LARGER than pooled AUROC/accuracy degradation "
              "-- consistent with H1 (subtype-only test; size-tercile "
              "clause untested, see module docstring).")
    elif p_value < 0.05 and observed_diff < 0:
        print("RESULT: statistically significant, but in the OPPOSITE "
              "direction from H1's prediction (stratified degradation "
              "smaller than pooled) -- H1 is NOT supported by this test.")
    else:
        print("RESULT: NOT statistically significant (p >= 0.05) -- H1 is "
              "NOT supported by this single-external-site, subtype-only "
              "test as currently scoped.")

    out = {
        "internal_metrics": internal_metrics,
        "external_metrics": external_metrics,
        "observed_degradation": observed,
        "observed_diff_stratified_minus_pooled": observed_diff,
        "bootstrap": {
            "n_bootstrap": N_BOOTSTRAP, "seed": SEED,
            "n_degenerate_dropped": n_dropped,
            "pooled_degradation_ci": pooled_ci,
            "stratified_degradation_ci": stratified_ci,
            "edh_degradation_ci": edh_ci,
            "p_value": p_value,
        },
        "scope_limitation": (
            "Only CQ500 (one of three Section IV-B external sites) was "
            "available for this run; size-tercile stratification is "
            "unavailable for CQ500 (no lesion_size_mm3; BHSD is the "
            "size-stratification source per Section IV-E and was out of "
            "scope). This result tests H1's subtype-sensitivity clause "
            "against one external site only, not the full design."
        ),
        "config": {"backbone": config["backbone"], "epochs": config["epochs"],
                    "batch_size": config["batch_size"], "lr": config["lr"],
                    "weight_decay": config["weight_decay"], "fold": config["fold"],
                    "n_folds": config["n_folds"], "max_samples": config["max_samples"],
                    "seed": config["seed"]},
    }
    out_path = args.runs_dir / "h1_baseline2_degradation_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
