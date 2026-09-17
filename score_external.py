"""Score the decisive-check checkpoints on an EXTERNAL site (CQ500), not
just the RSNA held-out fold.

WHY THIS MATTERS (read before interpreting decisive_check_result.json)
=====================================================================
run_pipeline.py's decisive check scores WICL vs. baseline 4c on the RSNA
HELD-OUT VALIDATION FOLD -- i.e. same-site, same-scanner-population,
same-acquisition-protocol data as training. But the manuscript's actual
claim (title, Section V) is about WINDOWING-INVARIANT CROSS-SITE
GENERALIZATION. Internal validation is precisely the regime where a
windowing-invariance method should be expected to help LEAST, because
there is no site shift to be robust to.

So a null result on RSNA-internal is weak evidence against WICL's claim;
the load-bearing comparison is on an external site. This script runs that
comparison on CQ500 (manuscript Section IV-B, zero-shot external test),
reporting:
  * pooled slice-level AUROC + DeLong, and
  * STUDY-level AUROC (probabilities aggregated per study), which is the
    honest granularity for CQ500 because its labels are study-level and
    were propagated to slices by prepare_cq500_labels.py -- slice-level
    CQ500 metrics are noisy by construction (see that script's DECISION 4).
  * absolute sensitivity/specificity at 0.5, not just deltas.
  * subtype- and size-stratified metrics via ich_gen.evaluate's shared
    evaluate_pooled_and_stratified() (closes repro-audit gap 5, 2026-09-03)
    -- manuscript Section IV-E's primary H2 test is stratified, not just
    pooled, and EDH is the manuscript's own stated primary subtype of
    interest for H2. This stratified pass is necessarily SLICE-level (like
    the pooled slice-level report above and for the same reason: CQ500 has
    no per-slice ground truth, only study-level labels propagated down --
    see DECISION 4 in prepare_cq500_labels.py), with patient/study-level
    bootstrap CIs. No lesion_size_mm3 is available for CQ500 (study-level
    labels only, per cq500.py), so size-tercile stratification is reported
    as NaN for this site by design, not by omission -- BHSD is this
    manuscript's size-stratification source (Section IV-E).

Usage
-----
    python score_external.py --runs-dir runs/decisive_v2 --cq500-root data/cq500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from ich_gen.datasets.common import MultiSiteICHDataset, LABEL_COLUMNS, SUBTYPES
from ich_gen.datasets.cq500 import build_samples as build_cq500
from ich_gen.evaluate import EvalInputs, evaluate_pooled_and_stratified
from ich_gen.models.backbone import build_backbone
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA

ANY_IDX = LABEL_COLUMNS.index("any")


def load_model(condition: str, out_dir: Path, config: dict, device: str):
    # weights_only=False: our own checkpoint, produced locally on this box
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    if condition == "wicl":
        model = build_wicl_model(config["backbone"], pretrained=False,
                                  lambda_consistency=config["lambda_consistency"],
                                  lambda_embedding=config["lambda_embedding"])
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        return model.forward_single
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


@torch.no_grad()
def score(forward, samples, config, device, batch_size=64):
    ds = MultiSiteICHDataset(samples, view_mode="fixed_three_window",
                              seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    probs, labels, patients, sizes = [], [], [], []
    for batch in loader:
        logits = forward(batch["image"].to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
        patients.extend(batch["patient_id"])
        sizes.append(batch["lesion_size_mm3"].numpy())
    return (np.concatenate(probs), np.concatenate(labels), np.array(patients),
            np.concatenate(sizes))


def sens_spec(y_true, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    pos, neg = y_true == 1, y_true == 0
    sens = float(pred[pos].mean()) if pos.sum() else float("nan")
    spec = float((1 - pred[neg]).mean()) if neg.sum() else float("nan")
    return sens, spec


def study_level(y_true, y_prob, patients, agg="max"):
    """Aggregate slice probabilities to one value per study. `max` mirrors
    how a study is called positive if any slice shows hemorrhage; `mean`
    is reported alongside because max is sensitive to single-slice noise."""
    df = pd.DataFrame({"p": y_prob, "y": y_true, "pid": patients})
    g = df.groupby("pid").agg(p=("p", agg), y=("y", "max"))
    return g["y"].to_numpy(), g["p"].to_numpy()


def report(name, y_true, p_wicl, p_base, patients=None):
    print(f"\n=== {name} ===")
    print(f"n = {len(y_true)}, positive rate = {y_true.mean():.3f}")
    if len(np.unique(y_true)) < 2:
        print("  (degenerate: only one class present, cannot compute AUROC)")
        return None
    r = delong_paired_auc_test(y_true, p_wicl, p_base)
    sw, pw = sens_spec(y_true, p_wicl)
    sb, pb = sens_spec(y_true, p_base)
    print(f"  WICL        AUROC={r.auc_a:.4f}  sens@0.5={sw:.4f}  spec@0.5={pw:.4f}")
    print(f"  baseline 4c AUROC={r.auc_b:.4f}  sens@0.5={sb:.4f}  spec@0.5={pb:.4f}")
    print(f"  diff (WICL-4c) AUROC={r.auc_diff:+.4f}   DeLong z={r.z_statistic:+.3f}  p={r.p_value:.4f}")
    print(f"  sensitivity delta = {sw - sb:+.4f}")
    return {"auc_wicl": r.auc_a, "auc_baseline4c": r.auc_b,
            "auc_diff": r.auc_diff, "delong_p": r.p_value,
            "sens_wicl": sw, "sens_baseline4c": sb,
            "spec_wicl": pw, "spec_baseline4c": pb, "n": int(len(y_true)),
            "positive_rate": float(y_true.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    ap.add_argument("--max-cq500-slices", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.runs_dir / "decisive_wicl" / "config.json") as f:
        config = json.load(f)

    fwd_wicl = load_model("wicl", args.runs_dir / "decisive_wicl", config, device)
    fwd_base = load_model("random_window_single",
                           args.runs_dir / "decisive_random_window_single",
                           config, device)

    print("building CQ500 external test set ...")
    cq_samples = build_cq500(args.cq500_root)
    if args.max_cq500_slices:
        cq_samples = cq_samples[:args.max_cq500_slices]
    print(f"CQ500: {len(cq_samples)} slices, "
          f"{len({s.patient_id for s in cq_samples})} studies")

    pw, yw, pat, sizew = score(fwd_wicl, cq_samples, config, device)
    pb, yb, pat2, sizeb = score(fwd_base, cq_samples, config, device)
    assert np.array_equal(pat, pat2) and np.allclose(yw, yb)

    y_any = yw[:, ANY_IDX]
    results = {}
    results["cq500_slice_level"] = report(
        "CQ500 external -- SLICE level (noisy: study labels propagated)",
        y_any, pw[:, ANY_IDX], pb[:, ANY_IDX])

    for agg in ("max", "mean"):
        ys, ps_w = study_level(y_any, pw[:, ANY_IDX], pat, agg)
        _, ps_b = study_level(y_any, pb[:, ANY_IDX], pat, agg)
        results[f"cq500_study_level_{agg}"] = report(
            f"CQ500 external -- STUDY level (agg={agg})", ys, ps_w, ps_b)

    # --- subtype- and size-stratified metrics (gap 5, 2026-09-03) ---
    # SLICE-level, like cq500_slice_level above and for the same reason
    # (see module docstring): CQ500 has no per-slice ground truth.
    print("\n=== CQ500 external -- SUBTYPE/SIZE-STRATIFIED "
          "(evaluate_pooled_and_stratified, slice level, noisy: study "
          "labels propagated) ===")
    rec_wicl = evaluate_pooled_and_stratified(
        EvalInputs(y_true=yw, y_prob=pw, patient_ids=pat,
                   lesion_size_mm3=sizew, condition="wicl", site="cq500"),
        size_thresholds=None)
    rec_base = evaluate_pooled_and_stratified(
        EvalInputs(y_true=yb, y_prob=pb, patient_ids=pat2,
                   lesion_size_mm3=sizeb, condition="random_window_single",
                   site="cq500"),
        size_thresholds=None)
    results["cq500_stratified_wicl"] = rec_wicl
    results["cq500_stratified_baseline4c"] = rec_base

    print(f"pooled AUROC   WICL={rec_wicl['pooled_auroc']:.4f}  "
          f"baseline4c={rec_base['pooled_auroc']:.4f}")
    print(f"pooled sens@0.5 WICL={rec_wicl['pooled_sensitivity']:.4f} "
          f"{rec_wicl['pooled_sensitivity_ci']}  "
          f"baseline4c={rec_base['pooled_sensitivity']:.4f} "
          f"{rec_base['pooled_sensitivity_ci']}")

    subtype_summary = {}
    print(f"\n{'subtype':<18}{'AUROC(W)':>10}{'AUROC(B)':>10}"
          f"{'sens(W)':>10}{'sens(B)':>10}{'delta(pp)':>12}  flag")
    for subtype in SUBTYPES:
        auroc_w = rec_wicl[f"auroc_{subtype}"]
        auroc_b = rec_base[f"auroc_{subtype}"]
        sens_w = rec_wicl[f"sensitivity_{subtype}"]
        sens_b = rec_base[f"sensitivity_{subtype}"]
        delta = (sens_w - sens_b) if not (np.isnan(sens_w) or np.isnan(sens_b)) else float("nan")
        clinically_relevant = (not np.isnan(delta)) and abs(delta) >= MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
        flag = "" if np.isnan(delta) else (
            "*** >=5pp delta ***" if clinically_relevant else "")
        print(f"{subtype:<18}{auroc_w:>10.4f}{auroc_b:>10.4f}"
              f"{sens_w:>10.4f}{sens_b:>10.4f}{delta*100 if not np.isnan(delta) else float('nan'):>12.2f}  {flag}")
        subtype_summary[subtype] = {
            "auroc_wicl": auroc_w, "auroc_baseline4c": auroc_b,
            "sensitivity_wicl": sens_w, "sensitivity_baseline4c": sens_b,
            "sensitivity_delta_wicl_minus_baseline4c": delta,
            "meets_min_clinically_relevant_sensitivity_delta": clinically_relevant,
        }
    # EDH is the manuscript's own stated primary subtype of interest for H2
    # (repro-audit gap 5 instruction) -- call it out explicitly.
    edh = subtype_summary["epidural"]
    print(f"\n--- EDH (primary H2 subtype of interest) ---")
    print(f"  WICL sens={edh['sensitivity_wicl']:.4f}  "
          f"baseline4c sens={edh['sensitivity_baseline4c']:.4f}  "
          f"delta={edh['sensitivity_delta_wicl_minus_baseline4c']*100 if not np.isnan(edh['sensitivity_delta_wicl_minus_baseline4c']) else float('nan'):+.2f}pp "
          f"(threshold: {MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA*100:.1f}pp) "
          f"-- {'MEETS' if edh['meets_min_clinically_relevant_sensitivity_delta'] else 'does NOT meet'} "
          f"the clinically-relevant sensitivity-delta threshold")
    results["cq500_stratified_subtype_summary"] = subtype_summary

    out = args.runs_dir / "external_cq500_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
