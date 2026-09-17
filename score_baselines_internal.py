"""Score baseline 1 (fixed_single_window) vs baseline 2 (fixed_three_window)
on the RSNA held-out validation fold (same patient-grouped fold used at
training time), per manuscript Section IV-E / VI-A.

METHODOLOGY NOTE (deviation from score_decisive.py's eval-view convention):
score_decisive.py / score_external.py always evaluate with
view_mode="fixed_three_window", because that convention exists specifically
for conditions trained with a RANDOMIZED view (WICL's random_window_dual,
baseline 4c's random_window_single) that need one FIXED canonical view at
inference time for a fair, equal-inference-cost comparison against baseline
2 (see MultiSiteICHDataset.set_view_mode's docstring, manuscript Section
V-E). Baseline 1 and baseline 2 have NO such train/eval view mismatch to
resolve -- baseline 1's whole definition IS a single fixed window used
throughout, and baseline 2's is a fixed three-window composite used
throughout (manuscript Section VI-A / train.py's CONDITION_VIEW_MODE).
Scoring baseline 1 with view_mode="fixed_three_window" would feed it
3-window-composite images it was never trained on, silently corrupting the
comparison. This script therefore scores EACH condition with its OWN
training-time view_mode, which is also each baseline's intended real-world
inference-time preprocessing.

Usage
-----
    python score_baselines_internal.py --runs-dir runs/baselines_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ich_gen.datasets.common import MultiSiteICHDataset, LABEL_COLUMNS, SUBTYPES
from ich_gen.models.backbone import build_backbone
from ich_gen.evaluate import EvalInputs, evaluate_pooled_and_stratified
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
from ich_gen import train as train_mod

ANY_IDX = LABEL_COLUMNS.index("any")

CONDITIONS = {
    "baseline1_fixed_single": "fixed_single_window",
    "baseline2_fixed_three": "fixed_three_window",
}


def load_val_split(config: dict):
    ns = argparse.Namespace(
        rsna_root=Path(config["rsna_root"]),
        max_samples=config["max_samples"],
        n_folds=config["n_folds"],
        seed=config["seed"],
    )
    samples, folds = train_mod.build_folds(ns)
    fold = folds[config["fold"]]
    val_samples = [samples[i] for i in fold.val_idx]
    return val_samples


@torch.no_grad()
def score_checkpoint(run_key: str, out_dir: Path, config: dict, val_samples: list, device: str):
    # weights_only=False: our own checkpoint, produced locally on this box
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    view_mode = CONDITIONS[run_key]
    val_ds = MultiSiteICHDataset(val_samples, view_mode=view_mode, seed=config["seed"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=4)

    all_probs, all_labels, all_patients, all_sizes = [], [], [], []
    for batch in val_loader:
        x = batch["image"].to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())
        all_patients.extend(batch["patient_id"])
        all_sizes.append(batch["lesion_size_mm3"].numpy())
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_prob, np.array(all_patients), np.concatenate(all_sizes)


def sens_spec(y_true, y_prob, thr=0.5):
    pred = (y_prob >= thr).astype(int)
    pos, neg = y_true == 1, y_true == 0
    sens = float(pred[pos].mean()) if pos.sum() else float("nan")
    spec = float((1 - pred[neg]).mean()) if neg.sum() else float("nan")
    return sens, spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/baselines_v1"))
    cli = ap.parse_args()
    RUNS_DIR = cli.runs_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(RUNS_DIR / "baseline1_fixed_single" / "config.json") as f:
        config1 = json.load(f)
    with open(RUNS_DIR / "baseline2_fixed_three" / "config.json") as f:
        config2 = json.load(f)
    for key in ("rsna_root", "fold", "n_folds", "max_samples", "seed", "backbone",
                "epochs", "batch_size", "lr", "weight_decay"):
        assert config1[key] == config2[key], (
            f"config mismatch on {key}: {config1[key]!r} vs {config2[key]!r} "
            f"-- manuscript Section VI-B requires identical hyperparameters "
            f"across conditions for a valid comparison")

    val_samples = load_val_split(config1)
    print(f"held-out validation fold: {len(val_samples)} slices, "
          f"{len({s.patient_id for s in val_samples})} patients")

    y1, p1, pat1, sz1 = score_checkpoint(
        "baseline1_fixed_single", RUNS_DIR / "baseline1_fixed_single",
        config1, val_samples, device)
    y2, p2, pat2, sz2 = score_checkpoint(
        "baseline2_fixed_three", RUNS_DIR / "baseline2_fixed_three",
        config2, val_samples, device)

    assert np.array_equal(pat1, pat2), "val ordering mismatch between runs"
    assert np.allclose(y1, y2), "label mismatch between runs"

    y_any = y1[:, ANY_IDX]

    print("\n=== RSNA in-distribution, held-out fold %d: pooled 'any' ICH ===" % config1["fold"])
    r = delong_paired_auc_test(y_any, p1[:, ANY_IDX], p2[:, ANY_IDX])
    s1, sp1 = sens_spec(y_any, p1[:, ANY_IDX])
    s2, sp2 = sens_spec(y_any, p2[:, ANY_IDX])
    print(f"  baseline1 (fixed_single) AUROC={r.auc_a:.4f} sens@0.5={s1:.4f} spec@0.5={sp1:.4f}")
    print(f"  baseline2 (fixed_three)  AUROC={r.auc_b:.4f} sens@0.5={s2:.4f} spec@0.5={sp2:.4f}")
    print(f"  diff (b1-b2) AUROC={r.auc_diff:+.4f}  DeLong z={r.z_statistic:+.3f}  p={r.p_value:.4f}")

    results = {
        "config": {"backbone": config1["backbone"], "epochs": config1["epochs"],
                    "batch_size": config1["batch_size"], "lr": config1["lr"],
                    "weight_decay": config1["weight_decay"], "fold": config1["fold"],
                    "n_folds": config1["n_folds"], "max_samples": config1["max_samples"],
                    "seed": config1["seed"]},
        "n_val_slices": int(len(y_any)),
        "n_val_patients": int(len(set(pat1.tolist()))),
        "pooled_any": {
            "auroc_baseline1": r.auc_a, "auroc_baseline2": r.auc_b,
            "auroc_diff_b1_minus_b2": r.auc_diff, "delong_z": r.z_statistic,
            "delong_p": r.p_value,
            "sensitivity_baseline1": s1, "sensitivity_baseline2": s2,
            "specificity_baseline1": sp1, "specificity_baseline2": sp2,
        },
    }

    print("\n=== stratified (evaluate_pooled_and_stratified) ===")
    rec1 = evaluate_pooled_and_stratified(
        EvalInputs(y_true=y1, y_prob=p1, patient_ids=pat1, lesion_size_mm3=sz1,
                   condition="baseline1_fixed_single", site="rsna_internal"),
        size_thresholds=None)
    rec2 = evaluate_pooled_and_stratified(
        EvalInputs(y_true=y2, y_prob=p2, patient_ids=pat2, lesion_size_mm3=sz2,
                   condition="baseline2_fixed_three", site="rsna_internal"),
        size_thresholds=None)
    results["stratified_baseline1"] = rec1
    results["stratified_baseline2"] = rec2

    print(f"pooled accuracy   b1={rec1['pooled_accuracy']:.4f} {rec1['pooled_accuracy_ci']}  "
          f"b2={rec2['pooled_accuracy']:.4f} {rec2['pooled_accuracy_ci']}")

    print(f"\n{'subtype':<18}{'AUROC(b1)':>10}{'AUROC(b2)':>10}"
          f"{'sens(b1)':>10}{'sens(b2)':>10}{'delta(pp)':>12}")
    subtype_summary = {}
    for subtype in SUBTYPES:
        a1 = rec1[f"auroc_{subtype}"]; a2 = rec2[f"auroc_{subtype}"]
        s1_ = rec1[f"sensitivity_{subtype}"]; s2_ = rec2[f"sensitivity_{subtype}"]
        delta = (s1_ - s2_) if not (np.isnan(s1_) or np.isnan(s2_)) else float("nan")
        print(f"{subtype:<18}{a1:>10.4f}{a2:>10.4f}{s1_:>10.4f}{s2_:>10.4f}"
              f"{delta*100 if not np.isnan(delta) else float('nan'):>12.2f}")
        subtype_summary[subtype] = {
            "auroc_baseline1": a1, "auroc_baseline2": a2,
            "sensitivity_baseline1": s1_, "sensitivity_baseline2": s2_,
            "sensitivity_delta_b1_minus_b2": delta,
        }
    results["subtype_summary"] = subtype_summary

    edh = subtype_summary["epidural"]
    print(f"\n--- EDH (H1/H2 priority subtype) ---")
    print(f"  baseline1 sens={edh['sensitivity_baseline1']:.4f}  "
          f"baseline2 sens={edh['sensitivity_baseline2']:.4f}  "
          f"delta(b1-b2)={edh['sensitivity_delta_b1_minus_b2']*100 if not np.isnan(edh['sensitivity_delta_b1_minus_b2']) else float('nan'):+.2f}pp")

    print(f"\nsize-tercile stratification: {rec2.get('size_stratification_note', 'n/a')}")

    out = RUNS_DIR / "baselines_internal_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
