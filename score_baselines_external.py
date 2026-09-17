"""Score baseline 1 (fixed_single_window) vs baseline 2 (fixed_three_window)
checkpoints on external CQ500 (manuscript Section IV-B), at three
aggregation levels (slice-level pooled, study-level max, study-level mean),
plus subtype-/size-stratified metrics via evaluate_pooled_and_stratified,
reusing score_external.py's methodology (DeLong's test for AUROC,
patient-level bootstrap CIs for sensitivity/specificity), generalized from
wicl-vs-baseline4c to baseline1-vs-baseline2.

METHODOLOGY NOTE: as in score_baselines_internal.py, each baseline is
scored with its OWN designated view_mode (fixed_single_window for baseline
1, fixed_three_window for baseline 2) rather than the single canonical
fixed_three_window eval-view convention score_external.py uses for
WICL/baseline 4c (whose randomized TRAINING views require a fixed canonical
EVAL view for fairness -- baseline 1/2 have no such mismatch to resolve,
see score_baselines_internal.py's module docstring for the full
justification).

CQ500 has no per-slice ground truth (study-level labels only, propagated to
slices -- see prepare_cq500_labels.py) and no lesion_size_mm3 (study-level
labels only, per cq500.py) -- size-tercile stratification is therefore NaN
here by design, matching score_external.py's documented limitation; BHSD is
this manuscript's size-stratification source (Section IV-E) and is out of
scope for this run.

Usage
-----
    python score_baselines_external.py --runs-dir runs/baselines_v1 --cq500-root data/cq500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ich_gen.datasets.common import MultiSiteICHDataset, LABEL_COLUMNS, SUBTYPES
from ich_gen.datasets.cq500 import build_samples as build_cq500
from ich_gen.evaluate import EvalInputs, evaluate_pooled_and_stratified
from ich_gen.models.backbone import build_backbone
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA

ANY_IDX = LABEL_COLUMNS.index("any")

CONDITIONS = {
    "baseline1_fixed_single": "fixed_single_window",
    "baseline2_fixed_three": "fixed_three_window",
}


def load_model(run_key: str, out_dir: Path, config: dict, device: str):
    # weights_only=False: our own checkpoint, produced locally on this box
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


@torch.no_grad()
def score(model, view_mode, samples, config, device, batch_size=64):
    ds = MultiSiteICHDataset(samples, view_mode=view_mode, seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    probs, labels, patients, sizes = [], [], [], []
    for batch in loader:
        logits = model(batch["image"].to(device))
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


def report(name, y_true, p1, p2):
    print(f"\n=== {name} ===")
    print(f"n = {len(y_true)}, positive rate = {y_true.mean():.3f}")
    if len(np.unique(y_true)) < 2:
        print("  (degenerate: only one class present, cannot compute AUROC)")
        return None
    r = delong_paired_auc_test(y_true, p1, p2)
    s1, sp1 = sens_spec(y_true, p1)
    s2, sp2 = sens_spec(y_true, p2)
    print(f"  baseline1 (fixed_single) AUROC={r.auc_a:.4f}  sens@0.5={s1:.4f}  spec@0.5={sp1:.4f}")
    print(f"  baseline2 (fixed_three)  AUROC={r.auc_b:.4f}  sens@0.5={s2:.4f}  spec@0.5={sp2:.4f}")
    print(f"  diff (b1-b2) AUROC={r.auc_diff:+.4f}   DeLong z={r.z_statistic:+.3f}  p={r.p_value:.4f}")
    print(f"  sensitivity delta (b1-b2) = {s1 - s2:+.4f}")
    return {"auc_baseline1": r.auc_a, "auc_baseline2": r.auc_b,
            "auc_diff_b1_minus_b2": r.auc_diff, "delong_p": r.p_value,
            "sens_baseline1": s1, "sens_baseline2": s2,
            "spec_baseline1": sp1, "spec_baseline2": sp2, "n": int(len(y_true)),
            "positive_rate": float(y_true.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    ap.add_argument("--max-cq500-slices", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.runs_dir / "baseline1_fixed_single" / "config.json") as f:
        config1 = json.load(f)
    with open(args.runs_dir / "baseline2_fixed_three" / "config.json") as f:
        config2 = json.load(f)

    model1 = load_model("baseline1_fixed_single", args.runs_dir / "baseline1_fixed_single",
                         config1, device)
    model2 = load_model("baseline2_fixed_three", args.runs_dir / "baseline2_fixed_three",
                         config2, device)

    print("building CQ500 external test set ...")
    cq_samples = build_cq500(args.cq500_root)
    if args.max_cq500_slices:
        cq_samples = cq_samples[:args.max_cq500_slices]
    print(f"CQ500: {len(cq_samples)} slices, "
          f"{len({s.patient_id for s in cq_samples})} studies")

    p1, y1, pat1, size1 = score(model1, CONDITIONS["baseline1_fixed_single"],
                                 cq_samples, config1, device)
    p2, y2, pat2, size2 = score(model2, CONDITIONS["baseline2_fixed_three"],
                                 cq_samples, config2, device)
    assert np.array_equal(pat1, pat2) and np.allclose(y1, y2)

    y_any = y1[:, ANY_IDX]
    results = {}
    results["cq500_slice_level"] = report(
        "CQ500 external -- SLICE level (noisy: study labels propagated)",
        y_any, p1[:, ANY_IDX], p2[:, ANY_IDX])

    for agg in ("max", "mean"):
        ys, ps1 = study_level(y_any, p1[:, ANY_IDX], pat1, agg)
        _, ps2 = study_level(y_any, p2[:, ANY_IDX], pat1, agg)
        results[f"cq500_study_level_{agg}"] = report(
            f"CQ500 external -- STUDY level (agg={agg})", ys, ps1, ps2)

    # --- subtype- and size-stratified metrics (mirrors score_external.py's
    # gap-5 close, applied to baseline1/baseline2 rather than wicl/4c) ---
    print("\n=== CQ500 external -- SUBTYPE/SIZE-STRATIFIED "
          "(evaluate_pooled_and_stratified, slice level, noisy: study "
          "labels propagated) ===")
    rec1 = evaluate_pooled_and_stratified(
        EvalInputs(y_true=y1, y_prob=p1, patient_ids=pat1, lesion_size_mm3=size1,
                   condition="baseline1_fixed_single", site="cq500"),
        size_thresholds=None)
    rec2 = evaluate_pooled_and_stratified(
        EvalInputs(y_true=y2, y_prob=p2, patient_ids=pat2, lesion_size_mm3=size2,
                   condition="baseline2_fixed_three", site="cq500"),
        size_thresholds=None)
    results["cq500_stratified_baseline1"] = rec1
    results["cq500_stratified_baseline2"] = rec2

    print(f"pooled AUROC   b1={rec1['pooled_auroc']:.4f}  b2={rec2['pooled_auroc']:.4f}")
    print(f"pooled sens@0.5 b1={rec1['pooled_sensitivity']:.4f} "
          f"{rec1['pooled_sensitivity_ci']}  "
          f"b2={rec2['pooled_sensitivity']:.4f} {rec2['pooled_sensitivity_ci']}")

    subtype_summary = {}
    print(f"\n{'subtype':<18}{'AUROC(b1)':>10}{'AUROC(b2)':>10}"
          f"{'sens(b1)':>10}{'sens(b2)':>10}{'delta(pp)':>12}  flag")
    for subtype in SUBTYPES:
        a1 = rec1[f"auroc_{subtype}"]
        a2 = rec2[f"auroc_{subtype}"]
        s1 = rec1[f"sensitivity_{subtype}"]
        s2 = rec2[f"sensitivity_{subtype}"]
        delta = (s1 - s2) if not (np.isnan(s1) or np.isnan(s2)) else float("nan")
        clinically_relevant = (not np.isnan(delta)) and abs(delta) >= MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
        flag = "" if np.isnan(delta) else ("*** >=5pp delta ***" if clinically_relevant else "")
        print(f"{subtype:<18}{a1:>10.4f}{a2:>10.4f}"
              f"{s1:>10.4f}{s2:>10.4f}{delta*100 if not np.isnan(delta) else float('nan'):>12.2f}  {flag}")
        subtype_summary[subtype] = {
            "auroc_baseline1": a1, "auroc_baseline2": a2,
            "sensitivity_baseline1": s1, "sensitivity_baseline2": s2,
            "sensitivity_delta_b1_minus_b2": delta,
            "meets_min_clinically_relevant_sensitivity_delta": clinically_relevant,
        }
    edh = subtype_summary["epidural"]
    print(f"\n--- EDH (primary H1/H2 subtype of interest) ---")
    print(f"  baseline1 sens={edh['sensitivity_baseline1']:.4f}  "
          f"baseline2 sens={edh['sensitivity_baseline2']:.4f}  "
          f"delta(b1-b2)={edh['sensitivity_delta_b1_minus_b2']*100 if not np.isnan(edh['sensitivity_delta_b1_minus_b2']) else float('nan'):+.2f}pp "
          f"(threshold: {MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA*100:.1f}pp)")
    results["cq500_stratified_subtype_summary"] = subtype_summary

    print(f"\nsize-tercile stratification: {rec2.get('size_stratification_note', 'n/a')}")

    out = args.runs_dir / "external_cq500_baselines_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
