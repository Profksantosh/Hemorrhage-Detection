"""Score baseline 3a (adaptive_hrt) and baseline 3b (adaptive_wem)
checkpoints on external CQ500 (manuscript Section IV-B), at three
aggregation levels (slice-level pooled, study-level max, study-level
mean), plus subtype-/size-stratified metrics, and compare each against
baseline 2 (fixed_three_window) and against each other (Section VI-C H2,
which explicitly names "the two adaptive-window variants 3a/3b" as part
of its comparison set).

Generalizes score_baselines4ab_external.py (baseline2 vs 4a/4b) to
baseline3a/3b. Baseline 3a uses view_mode="adaptive_hrt" (per-slice
heuristic window chosen at dataset time; plain classifier forward).
Baseline 3b uses view_mode="raw_hu_for_wem" (dataset returns raw,
un-windowed HU; WEMClassifier's forward signature is
(hu_for_wem, hu_raw) -> logits, not (image,) -> logits) -- handled
explicitly in score() below, the same way score_baselines4ab_external.py
explicitly handles dg_coral's (logits, embedding) return-tuple.

CQ500 has no per-slice ground truth (study-level labels only, propagated
to slices) and no lesion_size_mm3 -- size-tercile stratification is NaN
here by design, matching every other external-scoring script in this
repo (BHSD is the size-stratification source, out of scope here).

Also reports a SUPPLEMENTARY (not the full Section VI-C H2 test) pairwise
comparison of each of baseline 3a/3b against the audited WICL checkpoint
(runs/decisive_v2/decisive_wicl, read-only) at all three aggregation
levels, since H2's comparison set explicitly includes the adaptive-window
variants -- flagged explicitly as partial, not a substitute for the full
45-comparison-per-pair Bonferroni-corrected test.

Usage
-----
    python score_baselines3ab_external.py --runs-dir runs/baselines_v1 --cq500-root data/cq500
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
from ich_gen.models.adaptive_window import WEMClassifier
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA

ANY_IDX = LABEL_COLUMNS.index("any")

CONDITIONS = {
    "baseline2_fixed_three": ("fixed_three_window", "plain"),
    "baseline3a_hrt": ("adaptive_hrt", "plain"),
    "baseline3b_wem": ("raw_hu_for_wem", "wem"),
}


def load_model(run_key: str, out_dir: Path, config: dict, device: str):
    view_mode, model_kind = CONDITIONS[run_key]
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    if model_kind == "wem":
        model = WEMClassifier(config["backbone"], pretrained=False)
    else:
        model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, view_mode, model_kind


@torch.no_grad()
def score(model, view_mode, model_kind, samples, config, device, batch_size=64):
    ds = MultiSiteICHDataset(samples, view_mode=view_mode, seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    probs, labels, patients, sizes = [], [], [], []
    for batch in loader:
        if model_kind == "wem":
            hu_for_wem = batch["hu_for_wem"].to(device)
            hu_raw = batch["hu_raw"].to(device)
            logits = model(hu_for_wem, hu_raw)
        else:
            x = batch["image"].to(device)
            logits = model(x)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
        patients.extend(batch["patient_id"])
        sizes.append(batch["lesion_size_mm3"].numpy())
    return (np.concatenate(probs), np.concatenate(labels), np.array(patients),
            np.concatenate(sizes))


@torch.no_grad()
def score_wicl(wicl_dir: Path, samples: list, device: str, batch_size=64):
    """Same eval convention as score_decisive.py / score_baselines3ab_internal.py:
    view_mode='fixed_three_window', model.forward_single."""
    with open(wicl_dir / "config.json") as f:
        config = json.load(f)
    ckpt = torch.load(wicl_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    model = build_wicl_model(config["backbone"], pretrained=False,
                              lambda_consistency=config["lambda_consistency"],
                              lambda_embedding=config["lambda_embedding"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    ds = MultiSiteICHDataset(samples, view_mode="fixed_three_window", seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    probs, labels, patients = [], [], []
    for batch in loader:
        logits = model.forward_single(batch["image"].to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["labels"].numpy())
        patients.extend(batch["patient_id"])
    return np.concatenate(probs), np.concatenate(labels), np.array(patients)


def sens_spec(y_true, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    pos, neg = y_true == 1, y_true == 0
    sens = float(pred[pos].mean()) if pos.sum() else float("nan")
    spec = float((1 - pred[neg]).mean()) if neg.sum() else float("nan")
    return sens, spec


def study_level(y_true, y_prob, patients, agg="max"):
    df = pd.DataFrame({"p": y_prob, "y": y_true, "pid": patients})
    g = df.groupby("pid").agg(p=("p", agg), y=("y", "max"))
    return g["y"].to_numpy(), g["p"].to_numpy()


def report(name, name_a, y_true, p_a, name_b, p_b):
    print(f"\n=== {name}: {name_a} vs {name_b} ===")
    print(f"n = {len(y_true)}, positive rate = {y_true.mean():.3f}")
    if len(np.unique(y_true)) < 2:
        print("  (degenerate: only one class present, cannot compute AUROC)")
        return None
    r = delong_paired_auc_test(y_true, p_a, p_b)
    sa, spa = sens_spec(y_true, p_a)
    sb, spb = sens_spec(y_true, p_b)
    print(f"  {name_a:<12} AUROC={r.auc_a:.4f}  sens@0.5={sa:.4f}  spec@0.5={spa:.4f}")
    print(f"  {name_b:<12} AUROC={r.auc_b:.4f}  sens@0.5={sb:.4f}  spec@0.5={spb:.4f}")
    print(f"  diff ({name_a}-{name_b}) AUROC={r.auc_diff:+.4f}   DeLong z={r.z_statistic:+.3f}  p={r.p_value:.4f}")
    print(f"  sensitivity delta ({name_a}-{name_b}) = {sa - sb:+.4f}")
    return {f"auc_{name_a}": r.auc_a, f"auc_{name_b}": r.auc_b,
            "auc_diff": r.auc_diff, "delong_z": r.z_statistic, "delong_p": r.p_value,
            f"sens_{name_a}": sa, f"sens_{name_b}": sb,
            f"spec_{name_a}": spa, f"spec_{name_b}": spb, "n": int(len(y_true)),
            "positive_rate": float(y_true.mean())}


def subtype_table(name_a, rec_a, name_b, rec_b):
    print(f"\n{'subtype':<18}{'AUROC(' + name_a + ')':>18}{'AUROC(' + name_b + ')':>18}"
          f"{'sens(' + name_a + ')':>14}{'sens(' + name_b + ')':>14}{'delta(pp)':>12}  flag")
    summary = {}
    for subtype in SUBTYPES:
        a1 = rec_a[f"auroc_{subtype}"]; a2 = rec_b[f"auroc_{subtype}"]
        s1 = rec_a[f"sensitivity_{subtype}"]; s2 = rec_b[f"sensitivity_{subtype}"]
        delta = (s1 - s2) if not (np.isnan(s1) or np.isnan(s2)) else float("nan")
        clinically_relevant = (not np.isnan(delta)) and abs(delta) >= MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
        flag = "" if np.isnan(delta) else ("*** >=5pp delta ***" if clinically_relevant else "")
        print(f"{subtype:<18}{a1:>18.4f}{a2:>18.4f}{s1:>14.4f}{s2:>14.4f}"
              f"{delta*100 if not np.isnan(delta) else float('nan'):>12.2f}  {flag}")
        summary[subtype] = {
            f"auroc_{name_a}": a1, f"auroc_{name_b}": a2,
            f"sensitivity_{name_a}": s1, f"sensitivity_{name_b}": s2,
            "sensitivity_delta": delta,
            "meets_min_clinically_relevant_sensitivity_delta": clinically_relevant,
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    ap.add_argument("--max-cq500-slices", type=int, default=None)
    ap.add_argument("--wicl-dir", type=Path, default=Path("runs/decisive_v2/decisive_wicl"))
    ap.add_argument("--skip-wicl", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = {}
    for key in CONDITIONS:
        with open(args.runs_dir / key / "config.json") as f:
            configs[key] = json.load(f)

    models = {}
    for key in CONDITIONS:
        models[key] = load_model(key, args.runs_dir / key, configs[key], device)

    print("building CQ500 external test set ...")
    cq_samples = build_cq500(args.cq500_root)
    if args.max_cq500_slices:
        cq_samples = cq_samples[:args.max_cq500_slices]
    print(f"CQ500: {len(cq_samples)} slices, "
          f"{len({s.patient_id for s in cq_samples})} studies")

    scored = {}
    for key, (model, view_mode, model_kind) in models.items():
        p, y, pat, sz = score(model, view_mode, model_kind, cq_samples, configs[key], device)
        scored[key] = dict(p=p, y=y, pat=pat, sz=sz)

    ref = scored["baseline2_fixed_three"]
    for key in ("baseline3a_hrt", "baseline3b_wem"):
        assert np.array_equal(ref["pat"], scored[key]["pat"]) and np.allclose(ref["y"], scored[key]["y"]), \
            f"sample ordering/label mismatch: {key}"

    y_any = ref["y"][:, ANY_IDX]
    results = {}

    short_names = {"baseline2_fixed_three": "baseline2",
                   "baseline3a_hrt": "baseline3a",
                   "baseline3b_wem": "baseline3b"}

    pairs = [
        ("baseline3a_hrt", "baseline2_fixed_three"),
        ("baseline3b_wem", "baseline2_fixed_three"),
        ("baseline3a_hrt", "baseline3b_wem"),
    ]

    for ka, kb in pairs:
        na, nb = short_names[ka], short_names[kb]
        pair_key = f"{na}_vs_{nb}"
        results[pair_key] = {}
        results[pair_key]["slice_level"] = report(
            "CQ500 external -- SLICE level (noisy: study labels propagated)",
            na, y_any, scored[ka]["p"][:, ANY_IDX], nb, scored[kb]["p"][:, ANY_IDX])
        for agg in ("max", "mean"):
            ys, ps_a = study_level(y_any, scored[ka]["p"][:, ANY_IDX], scored[ka]["pat"], agg)
            _, ps_b = study_level(y_any, scored[kb]["p"][:, ANY_IDX], scored[kb]["pat"], agg)
            results[pair_key][f"study_level_{agg}"] = report(
                f"CQ500 external -- STUDY level (agg={agg})", na, ys, ps_a, nb, ps_b)

    print("\n=== CQ500 external -- SUBTYPE/SIZE-STRATIFIED "
          "(evaluate_pooled_and_stratified, slice level, noisy: study "
          "labels propagated) ===")
    strat = {}
    for key, short in short_names.items():
        s = scored[key]
        rec = evaluate_pooled_and_stratified(
            EvalInputs(y_true=s["y"], y_prob=s["p"], patient_ids=s["pat"],
                       lesion_size_mm3=s["sz"], condition=key, site="cq500"),
            size_thresholds=None)
        strat[short] = rec
        print(f"  {short}: pooled AUROC={rec['pooled_auroc']:.4f}  "
              f"pooled sens@0.5={rec['pooled_sensitivity']:.4f} {rec['pooled_sensitivity_ci']}")
    results["cq500_stratified"] = strat

    for na, nb in (("baseline3a", "baseline2"), ("baseline3b", "baseline2"),
                   ("baseline3a", "baseline3b")):
        print(f"\n--- subtype table: {na} vs {nb} ---")
        key = f"cq500_subtype_{na}_vs_{nb}"
        summary = subtype_table(na, strat[na], nb, strat[nb])
        results[key] = summary
        edh = summary["epidural"]
        sens_keys = [k for k in edh.keys() if k.startswith("sensitivity_") and k != "sensitivity_delta"]
        print(f"\n--- EDH (primary H1/H2 subtype of interest), {na} vs {nb} ---")
        print(f"  {sens_keys[0]}={edh[sens_keys[0]]:.4f}  {sens_keys[1]}={edh[sens_keys[1]]:.4f}  "
              f"delta={edh['sensitivity_delta']*100 if not np.isnan(edh['sensitivity_delta']) else float('nan'):+.2f}pp "
              f"(threshold: {MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA*100:.1f}pp)")

    print(f"\nsize-tercile stratification: {strat['baseline2'].get('size_stratification_note', 'n/a')}")

    if not args.skip_wicl and args.wicl_dir.exists():
        print("\n=== SUPPLEMENTARY: WICL (runs/decisive_v2/decisive_wicl, read-only) "
              "vs baseline 3a/3b on CQ500 -- partial contribution toward Section "
              "VI-C's H2 comparison set, NOT the full Bonferroni-corrected test ===")
        p_w, y_w, pat_w = score_wicl(args.wicl_dir, cq_samples, device)
        assert np.array_equal(ref["pat"], pat_w) and np.allclose(ref["y"], y_w), \
            "WICL sample ordering/label mismatch"
        p_w_any = p_w[:, ANY_IDX]
        wicl_pairs = [("baseline3a_hrt",), ("baseline3b_wem",)]
        for (ka,) in wicl_pairs:
            na = short_names[ka]
            pair_key = f"wicl_vs_{na}"
            results[pair_key] = {}
            results[pair_key]["slice_level"] = report(
                "CQ500 external -- SLICE level (noisy)",
                "wicl", y_any, p_w_any, na, scored[ka]["p"][:, ANY_IDX])
            for agg in ("max", "mean"):
                ys, ps_w = study_level(y_any, p_w_any, pat_w, agg)
                _, ps_a = study_level(y_any, scored[ka]["p"][:, ANY_IDX], scored[ka]["pat"], agg)
                results[pair_key][f"study_level_{agg}"] = report(
                    f"CQ500 external -- STUDY level (agg={agg})", "wicl", ys, ps_w, na, ps_a)
        results["supplementary_wicl_scope_note"] = (
            "Pooled DeLong comparisons of WICL (audited "
            "runs/decisive_v2/decisive_wicl checkpoint, read-only, not "
            "retrained here) against baseline 3a/3b on CQ500, at all three "
            "aggregation levels. This is a partial, early contribution "
            "toward Section VI-C's H2 comparison set (which names the "
            "adaptive-window variants explicitly) -- NOT the full "
            "45-comparison-per-pair, Bonferroni-corrected H2 test, which "
            "requires all baselines 1-5, all three external sites, and BHSD "
            "size-tercile stratification, none of which are all complete yet.")
    else:
        print("\n(skipping supplementary WICL comparison: --skip-wicl set or "
              f"{args.wicl_dir} not found)")

    out = args.runs_dir / "external_cq500_baselines3ab_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
