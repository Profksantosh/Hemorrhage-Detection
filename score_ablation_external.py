"""H2-style external scoring of the WICL component ablation checkpoints
(manuscript revision, requested by the multi-expert review's Reviewer 2:
"the ablation was only run on the synthetic H4 axis, never on real
external sites -- does embedding-only WICL do any better on CQ500/
PhysioNet-ICH than full WICL did?").

Scores wicl_ablation_pred_only and wicl_ablation_embed_only (both
already trained, read-only checkpoints) against baseline 4c (the H2
comparator) and full WICL, on CQ500 (slice/study-max/study-mean) and
PhysioNet-ICH (slice/patient-max/patient-mean), using the exact same
DeLong-test/aggregation conventions as score_external_bhsd_excluded.py
and score_physionet_ich_external_bhsd_excluded.py.

INFERENCE ONLY. No checkpoint is retrained.

Usage
-----
    python score_ablation_external.py --runs-dir runs/baselines_bhsd_excluded \
        --cq500-root data/cq500 --physionet-root data/physionet_ich
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
from ich_gen.datasets.physionet_ich import build_samples as build_physionet_ich
from ich_gen.models.backbone import build_backbone
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stats import delong_paired_auc_test

ANY_IDX = LABEL_COLUMNS.index("any")

# run_key -> (short_name, is_wicl_wrapper)
# baseline4c is architecturally a PLAIN classifier (build_backbone): it only
# ever sees one stochastic view per training step, so it never needed the
# WICL dual-branch wrapper -- unlike pred_only/embed_only/wicl_full, which
# are all WICL-condition checkpoints (build_wicl_model), just with
# different (lambda_consistency, lambda_embedding) weights.
CONDITIONS = {
    "baseline4c_random_window_single": ("baseline4c", False),
    "wicl_ablation_pred_only": ("pred_only", True),
    "wicl_ablation_embed_only": ("embed_only", True),
    "decisive_wicl": ("wicl_full", True),
}


@torch.no_grad()
def score_checkpoint(run_dir: Path, is_wicl_wrapper: bool, samples: list, device: str, batch_size=64):
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    ckpt = torch.load(run_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
    if is_wicl_wrapper:
        model = build_wicl_model(config["backbone"], pretrained=False,
                                  lambda_consistency=config["lambda_consistency"],
                                  lambda_embedding=config["lambda_embedding"])
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        forward_fn = lambda m, x: m.forward_single(x)
    else:
        model = build_backbone(config["backbone"], pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        forward_fn = lambda m, x: m(x)
    ds = MultiSiteICHDataset(samples, view_mode="fixed_three_window", seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    probs, labels, patients, sizes = [], [], [], []
    for batch in loader:
        logits = forward_fn(model, batch["image"].to(device))
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


def agg_level(y_true, y_prob, patients, agg="max"):
    df = pd.DataFrame({"p": y_prob, "y": y_true, "pid": patients})
    g = df.groupby("pid").agg(p=("p", agg), y=("y", "max"))
    return g["y"].to_numpy(), g["p"].to_numpy()


def report(name, name_a, y_true, p_a, name_b, p_b):
    print(f"\n=== {name}: {name_a} vs {name_b} ===")
    print(f"n = {len(y_true)}, positive rate = {y_true.mean():.3f}")
    if len(np.unique(y_true)) < 2:
        print("  (degenerate: only one class present)")
        return None
    r = delong_paired_auc_test(y_true, p_a, p_b)
    sa, spa = sens_spec(y_true, p_a)
    sb, spb = sens_spec(y_true, p_b)
    print(f"  {name_a:<14} AUROC={r.auc_a:.4f}  sens@0.5={sa:.4f}  spec@0.5={spa:.4f}")
    print(f"  {name_b:<14} AUROC={r.auc_b:.4f}  sens@0.5={sb:.4f}  spec@0.5={spb:.4f}")
    print(f"  diff ({name_a}-{name_b}) AUROC={r.auc_diff:+.4f}   DeLong z={r.z_statistic:+.3f}  p={r.p_value:.4f}")
    print(f"  sensitivity delta ({name_a}-{name_b}) = {sa - sb:+.4f}")
    return {f"auc_{name_a}": r.auc_a, f"auc_{name_b}": r.auc_b,
            "auc_diff": r.auc_diff, "delong_z": r.z_statistic, "delong_p": r.p_value,
            f"sens_{name_a}": sa, f"sens_{name_b}": sb, "n": int(len(y_true)),
            "sensitivity_delta": sa - sb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    ap.add_argument("--physionet-root", type=Path, default=Path("data/physionet_ich"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("building CQ500 external test set ...")
    cq_samples = build_cq500(args.cq500_root)
    print(f"CQ500: {len(cq_samples)} slices, {len({s.patient_id for s in cq_samples})} studies")

    print("building PhysioNet-ICH external test set ...")
    pn_samples = build_physionet_ich(args.physionet_root)
    print(f"PhysioNet-ICH: {len(pn_samples)} slices, {len({s.patient_id for s in pn_samples})} patients")

    scored_cq, scored_pn = {}, {}
    for key, (short, is_wicl) in CONDITIONS.items():
        run_dir = args.runs_dir / key
        print(f"\nscoring {short} ({key}, wicl_wrapper={is_wicl}) ...")
        p, y, pat, sz = score_checkpoint(run_dir, is_wicl, cq_samples, device)
        scored_cq[short] = dict(p=p, y=y, pat=pat, sz=sz)
        p2, y2, pat2, sz2 = score_checkpoint(run_dir, is_wicl, pn_samples, device)
        scored_pn[short] = dict(p=p2, y=y2, pat=pat2, sz=sz2)

    ref_cq = scored_cq["baseline4c"]
    ref_pn = scored_pn["baseline4c"]
    for short, _ in CONDITIONS.values():
        assert np.array_equal(ref_cq["pat"], scored_cq[short]["pat"]), f"CQ500 patient mismatch: {short}"
        assert np.array_equal(ref_pn["pat"], scored_pn[short]["pat"]), f"PhysioNet patient mismatch: {short}"

    y_any_cq = ref_cq["y"][:, ANY_IDX]
    y_any_pn = ref_pn["y"][:, ANY_IDX]

    results = {"note": (
        "H2-style external scoring of the WICL component ablation "
        "(pred_only, embed_only) against baseline4c and full WICL, "
        "requested by the multi-expert review (Reviewer 2) to check "
        "whether the H4-robust embedding-only variant also does better "
        "on real external transfer than full WICL, which failed H2."
    )}

    pairs = [
        ("pred_only", "baseline4c"), ("embed_only", "baseline4c"),
        ("pred_only", "wicl_full"), ("embed_only", "wicl_full"),
        ("pred_only", "embed_only"),
    ]

    print("\n" + "=" * 20 + " CQ500 " + "=" * 20)
    for a, b in pairs:
        pair_key = f"cq500_{a}_vs_{b}"
        results[pair_key] = {}
        results[pair_key]["slice"] = report(
            f"CQ500 slice", a, y_any_cq, scored_cq[a]["p"][:, ANY_IDX], b, scored_cq[b]["p"][:, ANY_IDX])
        for agg in ("max", "mean"):
            ys, pa = agg_level(y_any_cq, scored_cq[a]["p"][:, ANY_IDX], scored_cq[a]["pat"], agg)
            _, pb = agg_level(y_any_cq, scored_cq[b]["p"][:, ANY_IDX], scored_cq[b]["pat"], agg)
            results[pair_key][f"study_{agg}"] = report(
                f"CQ500 study(agg={agg})", a, ys, pa, b, pb)

    print("\n" + "=" * 20 + " PhysioNet-ICH " + "=" * 20)
    for a, b in pairs:
        pair_key = f"physionet_{a}_vs_{b}"
        results[pair_key] = {}
        results[pair_key]["slice"] = report(
            f"PhysioNet slice", a, y_any_pn, scored_pn[a]["p"][:, ANY_IDX], b, scored_pn[b]["p"][:, ANY_IDX])
        for agg in ("max", "mean"):
            ys, pa = agg_level(y_any_pn, scored_pn[a]["p"][:, ANY_IDX], scored_pn[a]["pat"], agg)
            _, pb = agg_level(y_any_pn, scored_pn[b]["p"][:, ANY_IDX], scored_pn[b]["pat"], agg)
            results[pair_key][f"patient_{agg}"] = report(
                f"PhysioNet patient(agg={agg})", a, ys, pa, b, pb)

    out = args.runs_dir / "ablation_external_h2style_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
