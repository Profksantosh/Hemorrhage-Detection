"""Score baseline 4a (dg_augment) and baseline 4b (dg_coral) checkpoints
on external CQ500 (manuscript Section IV-B), at three aggregation levels
(slice-level pooled, study-level max, study-level mean), plus subtype-/
size-stratified metrics, and compare each against baseline 2
(fixed_three_window) and against each other (Section VI-C H2).

Generalizes score_baselines_external.py (baseline1 vs baseline2) to
baseline4a/4b. Both 4a and 4b use view_mode="fixed_three_window" for
both training and eval (neither uses WICL's randomized-window mechanism,
so there is no train/eval canonical-view mismatch to resolve here,
unlike WICL/baseline4c).

CQ500 has no per-slice ground truth (study-level labels only, propagated
to slices) and no lesion_size_mm3 -- size-tercile stratification is NaN
here by design, matching every other external-scoring script in this
repo (BHSD is the size-stratification source, out of scope here).

Usage
-----
    python score_baselines4ab_external.py --runs-dir runs/baselines_v1 --cq500-root data/cq500
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
from ich_gen.models.backbone import build_backbone, EmbeddingBackbone
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA

ANY_IDX = LABEL_COLUMNS.index("any")

CONDITIONS = {
    "baseline2_fixed_three": ("fixed_three_window", "plain"),
    "baseline4a_augmentation": ("fixed_three_window", "plain"),
    "baseline4b_coral": ("fixed_three_window", "embedding"),
}


def load_model(run_key: str, out_dir: Path, config: dict, device: str):
    view_mode, model_kind = CONDITIONS[run_key]
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    if model_kind == "embedding":
        model = EmbeddingBackbone(config["backbone"], pretrained=False)
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
        x = batch["image"].to(device)
        if model_kind == "embedding":
            logits, _emb = model(x)
        else:
            logits = model(x)
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
    for key in ("baseline4a_augmentation", "baseline4b_coral"):
        assert np.array_equal(ref["pat"], scored[key]["pat"]) and np.allclose(ref["y"], scored[key]["y"]), \
            f"sample ordering/label mismatch: {key}"

    y_any = ref["y"][:, ANY_IDX]
    results = {}

    short_names = {"baseline2_fixed_three": "baseline2",
                   "baseline4a_augmentation": "baseline4a",
                   "baseline4b_coral": "baseline4b"}

    pairs = [
        ("baseline4a_augmentation", "baseline2_fixed_three"),
        ("baseline4b_coral", "baseline2_fixed_three"),
        ("baseline4a_augmentation", "baseline4b_coral"),
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

    for na, nb in (("baseline4a", "baseline2"), ("baseline4b", "baseline2"),
                   ("baseline4a", "baseline4b")):
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

    out = args.runs_dir / "external_cq500_baselines4ab_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
