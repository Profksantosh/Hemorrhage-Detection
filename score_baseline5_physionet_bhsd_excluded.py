"""Complete the missing piece of the baseline-5 (Lin & Yuh self-training
reimplementation) comparator set: PhysioNet-ICH external pairwise DeLong
tests of baseline5 vs baseline2, and WICL vs baseline5, at slice level and
patient-level (agg=max/mean). CQ500 external and RSNA internal pairwise
tests for baseline5 already exist (score_baseline5_external.py /
score_baseline5_internal_bhsd_excluded.py); score_physionet_ich_external_
bhsd_excluded.py computes baseline5's pooled/stratified PhysioNet numbers
but does not run the baseline5-vs-baseline2 or wicl-vs-baseline5 pairwise
DeLong tests on this site -- this script fills exactly that gap, reusing
the same conventions (patient-level aggregation, since PhysioNet has
genuine per-slice ground truth unlike CQ500's study-propagated labels).

INFERENCE ONLY. No checkpoint is retrained. Loads baseline2, baseline5,
and (unless --skip-wicl) WICL read-only from the BHSD-excluded runs dir.

Usage
-----
    python score_baseline5_physionet_bhsd_excluded.py \
        --runs-dir runs/baselines_bhsd_excluded --physionet-root data/physionet_ich
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
from ich_gen.datasets.physionet_ich import build_samples as build_physionet_ich
from ich_gen.evaluate import EvalInputs, evaluate_pooled_and_stratified
from ich_gen.models.backbone import build_backbone
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA

ANY_IDX = LABEL_COLUMNS.index("any")

CONDITIONS = {
    "baseline2_fixed_three": "fixed_three_window",
    "baseline5_selftraining": "fixed_three_window",
}
SHORT_NAMES = {"baseline2_fixed_three": "baseline2", "baseline5_selftraining": "baseline5"}


def load_model(run_key, out_dir, config, device):
    view_mode = CONDITIONS[run_key]
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, view_mode


@torch.no_grad()
def score(model, view_mode, samples, config, device, batch_size=64):
    ds = MultiSiteICHDataset(samples, view_mode=view_mode, seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    probs, labels, patients, sizes = [], [], [], []
    for batch in loader:
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
    with open(wicl_dir / "config.json") as f:
        config = json.load(f)
    ckpt = torch.load(wicl_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
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


def patient_level(y_true, y_prob, patients, agg="max"):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--physionet-root", type=Path, default=Path("data/physionet_ich"))
    ap.add_argument("--max-physionet-slices", type=int, default=None)
    ap.add_argument("--wicl-dir", type=Path, default=None,
                     help="defaults to <runs-dir>/decisive_wicl")
    ap.add_argument("--skip-wicl", action="store_true")
    args = ap.parse_args()
    if args.wicl_dir is None:
        args.wicl_dir = args.runs_dir / "decisive_wicl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs, models = {}, {}
    for key in CONDITIONS:
        with open(args.runs_dir / key / "config.json") as f:
            configs[key] = json.load(f)
        models[key] = load_model(key, args.runs_dir / key, configs[key], device)

    print("building PhysioNet-ICH external test set (REAL data) ...")
    pn_samples = build_physionet_ich(args.physionet_root)
    if args.max_physionet_slices:
        pn_samples = pn_samples[:args.max_physionet_slices]
    n_patients_total = len({s.patient_id for s in pn_samples})
    print(f"PhysioNet-ICH: {len(pn_samples)} slices, {n_patients_total} patients")

    scored = {}
    for key, (model, view_mode) in models.items():
        p, y, pat, sz = score(model, view_mode, pn_samples, configs[key], device)
        scored[key] = dict(p=p, y=y, pat=pat, sz=sz)

    ref = scored["baseline2_fixed_three"]
    assert np.array_equal(ref["pat"], scored["baseline5_selftraining"]["pat"]) and \
        np.allclose(ref["y"], scored["baseline5_selftraining"]["y"]), \
        "sample ordering/label mismatch: baseline5_selftraining"

    y_any = ref["y"][:, ANY_IDX]
    results = {
        "documented_simplification_note": (
            "baseline5's unlabeled pool is RSNA's own withheld-label "
            "patient-disjoint subset (--unlabeled-fraction), not a "
            "genuinely separate/external unlabeled corpus -- see "
            "models/semi_supervised.py's module docstring."
        ),
        "physionet_ich_dataset_summary": {
            "n_slices": len(pn_samples), "n_patients": n_patients_total,
        },
    }

    pair_key = "baseline5_vs_baseline2"
    results[pair_key] = {}
    results[pair_key]["slice_level"] = report(
        "PhysioNet-ICH external -- SLICE level (genuine per-slice ground truth)",
        "baseline5", y_any, scored["baseline5_selftraining"]["p"][:, ANY_IDX],
        "baseline2", scored["baseline2_fixed_three"]["p"][:, ANY_IDX])
    for agg in ("max", "mean"):
        ys, ps_a = patient_level(y_any, scored["baseline5_selftraining"]["p"][:, ANY_IDX],
                                  scored["baseline5_selftraining"]["pat"], agg)
        _, ps_b = patient_level(y_any, scored["baseline2_fixed_three"]["p"][:, ANY_IDX],
                                 scored["baseline2_fixed_three"]["pat"], agg)
        results[pair_key][f"patient_level_{agg}"] = report(
            f"PhysioNet-ICH external -- PATIENT level (agg={agg})",
            "baseline5", ys, ps_a, "baseline2", ps_b)

    strat = {}
    for key, short in SHORT_NAMES.items():
        s = scored[key]
        rec = evaluate_pooled_and_stratified(
            EvalInputs(y_true=s["y"], y_prob=s["p"], patient_ids=s["pat"],
                       lesion_size_mm3=s["sz"], condition=key, site="physionet_ich"),
            size_thresholds=None)
        strat[short] = rec
    results["physionet_ich_stratified_baseline5_baseline2"] = strat

    if not args.skip_wicl and args.wicl_dir.exists():
        print(f"\n=== WICL ({args.wicl_dir}) vs baseline5 on PhysioNet-ICH ===")
        p_w, y_w, pat_w = score_wicl(args.wicl_dir, pn_samples, device)
        assert np.array_equal(ref["pat"], pat_w) and np.allclose(ref["y"], y_w), \
            "WICL sample ordering/label mismatch"
        p_w_any = p_w[:, ANY_IDX]
        pair_key = "wicl_vs_baseline5"
        results[pair_key] = {}
        results[pair_key]["slice_level"] = report(
            "PhysioNet-ICH external -- SLICE level (genuine per-slice ground truth)",
            "wicl", y_any, p_w_any, "baseline5", scored["baseline5_selftraining"]["p"][:, ANY_IDX])
        for agg in ("max", "mean"):
            ys, ps_w = patient_level(y_any, p_w_any, pat_w, agg)
            _, ps_a = patient_level(y_any, scored["baseline5_selftraining"]["p"][:, ANY_IDX],
                                     scored["baseline5_selftraining"]["pat"], agg)
            results[pair_key][f"patient_level_{agg}"] = report(
                f"PhysioNet-ICH external -- PATIENT level (agg={agg})",
                "wicl", ys, ps_w, "baseline5", ps_a)
    else:
        print("\n(skipping WICL comparison: --skip-wicl set or wicl-dir not found)")

    out = args.runs_dir / "physionet_baseline5_vs_baseline2_and_wicl_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
