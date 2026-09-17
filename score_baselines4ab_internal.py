"""Score baseline 4a (dg_augment) and baseline 4b (dg_coral) on the RSNA
held-out validation fold (same patient-grouped fold used at training
time), per manuscript Section IV-E / VI-A, and compare each against
baseline 2 (fixed_three_window) and against each other (Section VI-C's
H2 pairwise-comparison requirement).

Generalizes score_baselines_internal.py (baseline1 vs baseline2) to
baseline4a/4b, reusing its exact scoring methodology (own training-time
view_mode per condition -- here both 4a and 4b use fixed_three_window,
so there is no train/eval view mismatch to resolve for either; DeLong's
test for AUROC; evaluate_pooled_and_stratified for subtype/size metrics;
size_thresholds=None, matching every other internal/external scoring
script in this repo, since RSNA/CQ500 carry no lesion_size_mm3 -- BHSD is
the size-stratification source and out of scope here).

dg_coral's checkpoint is an EmbeddingBackbone (forward returns
(logits, embedding)) rather than a plain classifier (forward returns
logits only) -- score_checkpoint() below handles this per-condition
model-class difference explicitly rather than assuming a uniform
interface, since assuming uniformity here would silently break dg_coral
scoring (calling sigmoid() on a tuple).

Usage
-----
    python score_baselines4ab_internal.py --runs-dir runs/baselines_v1 \
        --baseline2-dir runs/baselines_v1/baseline2_fixed_three
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ich_gen.datasets.common import MultiSiteICHDataset, LABEL_COLUMNS, SUBTYPES
from ich_gen.models.backbone import build_backbone, EmbeddingBackbone
from ich_gen.evaluate import EvalInputs, evaluate_pooled_and_stratified
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
from ich_gen import train as train_mod

ANY_IDX = LABEL_COLUMNS.index("any")

# condition key -> (dir name, view_mode, model_kind)
CONDITIONS = {
    "baseline2_fixed_three": ("fixed_three_window", "plain"),
    "baseline4a_augmentation": ("fixed_three_window", "plain"),
    "baseline4b_coral": ("fixed_three_window", "embedding"),
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
    view_mode, model_kind = CONDITIONS[run_key]
    # weights_only=False: our own checkpoint, produced locally on this box
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    if model_kind == "embedding":
        model = EmbeddingBackbone(config["backbone"], pretrained=False)
    else:
        model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    val_ds = MultiSiteICHDataset(val_samples, view_mode=view_mode, seed=config["seed"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=4)

    all_probs, all_labels, all_patients, all_sizes = [], [], [], []
    for batch in val_loader:
        x = batch["image"].to(device)
        if model_kind == "embedding":
            logits, _emb = model(x)
        else:
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


def pairwise_report(name_a, y_true, p_a, name_b, p_b):
    print(f"\n--- {name_a} vs {name_b}: pooled 'any' ICH ---")
    r = delong_paired_auc_test(y_true, p_a, p_b)
    sa, spa = sens_spec(y_true, p_a)
    sb, spb = sens_spec(y_true, p_b)
    print(f"  {name_a:<28} AUROC={r.auc_a:.4f} sens@0.5={sa:.4f} spec@0.5={spa:.4f}")
    print(f"  {name_b:<28} AUROC={r.auc_b:.4f} sens@0.5={sb:.4f} spec@0.5={spb:.4f}")
    print(f"  diff ({name_a}-{name_b}) AUROC={r.auc_diff:+.4f}  DeLong z={r.z_statistic:+.3f}  p={r.p_value:.4f}")
    return {
        f"auroc_{name_a}": r.auc_a, f"auroc_{name_b}": r.auc_b,
        "auroc_diff": r.auc_diff, "delong_z": r.z_statistic, "delong_p": r.p_value,
        f"sensitivity_{name_a}": sa, f"sensitivity_{name_b}": sb,
        f"specificity_{name_a}": spa, f"specificity_{name_b}": spb,
    }


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/baselines_v1"))
    cli = ap.parse_args()
    RUNS_DIR = cli.runs_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = {}
    for key in CONDITIONS:
        with open(RUNS_DIR / key / "config.json") as f:
            configs[key] = json.load(f)

    shared_keys = ("rsna_root", "fold", "n_folds", "max_samples", "seed", "backbone",
                   "epochs", "batch_size", "lr", "weight_decay")
    base_cfg = configs["baseline2_fixed_three"]
    for key in ("baseline4a_augmentation", "baseline4b_coral"):
        for k in shared_keys:
            assert configs[key][k] == base_cfg[k], (
                f"{key} config mismatch on {k}: {configs[key][k]!r} vs {base_cfg[k]!r} "
                f"-- manuscript Section VI-B requires identical hyperparameters "
                f"across conditions for a valid comparison")

    val_samples = load_val_split(base_cfg)
    print(f"held-out validation fold: {len(val_samples)} slices, "
          f"{len({s.patient_id for s in val_samples})} patients")

    scored = {}
    for key in CONDITIONS:
        y, p, pat, sz = score_checkpoint(key, RUNS_DIR / key, configs[key], val_samples, device)
        scored[key] = dict(y=y, p=p, pat=pat, sz=sz)

    # sanity: identical label/patient ordering across conditions (same val split)
    ref = scored["baseline2_fixed_three"]
    for key in ("baseline4a_augmentation", "baseline4b_coral"):
        assert np.array_equal(ref["pat"], scored[key]["pat"]), f"val ordering mismatch: {key}"
        assert np.allclose(ref["y"], scored[key]["y"]), f"label mismatch: {key}"

    y_any = ref["y"][:, ANY_IDX]
    results = {
        "config": {k: base_cfg[k] for k in shared_keys},
        "n_val_slices": int(len(y_any)),
        "n_val_patients": int(len(set(ref["pat"].tolist()))),
    }

    print("\n=== RSNA in-distribution, held-out fold %d: pairwise pooled 'any' ICH ===" % base_cfg["fold"])
    pairs = [
        ("baseline4a", "baseline2", "baseline4a_augmentation", "baseline2_fixed_three"),
        ("baseline4b", "baseline2", "baseline4b_coral", "baseline2_fixed_three"),
        ("baseline4a", "baseline4b", "baseline4a_augmentation", "baseline4b_coral"),
    ]
    results["pairwise_pooled_any"] = {}
    for na, nb, ka, kb in pairs:
        pa = scored[ka]["p"][:, ANY_IDX]
        pb = scored[kb]["p"][:, ANY_IDX]
        results["pairwise_pooled_any"][f"{na}_vs_{nb}"] = pairwise_report(na, y_any, pa, nb, pb)

    print("\n=== stratified (evaluate_pooled_and_stratified) ===")
    strat = {}
    for key, short in (("baseline2_fixed_three", "baseline2"),
                        ("baseline4a_augmentation", "baseline4a"),
                        ("baseline4b_coral", "baseline4b")):
        s = scored[key]
        rec = evaluate_pooled_and_stratified(
            EvalInputs(y_true=s["y"], y_prob=s["p"], patient_ids=s["pat"],
                       lesion_size_mm3=s["sz"], condition=key, site="rsna_internal"),
            size_thresholds=None)
        strat[short] = rec
        print(f"  {short}: pooled AUROC={rec['pooled_auroc']:.4f}  "
              f"pooled accuracy={rec['pooled_accuracy']:.4f} {rec['pooled_accuracy_ci']}  "
              f"pooled sens={rec['pooled_sensitivity']:.4f} {rec['pooled_sensitivity_ci']}")
    results["stratified"] = strat

    print("\n--- subtype table: baseline4a vs baseline2 ---")
    results["subtype_baseline4a_vs_baseline2"] = subtype_table(
        "baseline4a", strat["baseline4a"], "baseline2", strat["baseline2"])
    print("\n--- subtype table: baseline4b vs baseline2 ---")
    results["subtype_baseline4b_vs_baseline2"] = subtype_table(
        "baseline4b", strat["baseline4b"], "baseline2", strat["baseline2"])
    print("\n--- subtype table: baseline4a vs baseline4b ---")
    results["subtype_baseline4a_vs_baseline4b"] = subtype_table(
        "baseline4a", strat["baseline4a"], "baseline4b", strat["baseline4b"])

    for label, summary in (("baseline4a vs baseline2", results["subtype_baseline4a_vs_baseline2"]),
                            ("baseline4b vs baseline2", results["subtype_baseline4b_vs_baseline2"]),
                            ("baseline4a vs baseline4b", results["subtype_baseline4a_vs_baseline4b"])):
        edh = summary["epidural"]
        keys = list(edh.keys())
        sens_keys = [k for k in keys if k.startswith("sensitivity_") and k != "sensitivity_delta"]
        print(f"\n--- EDH (H1/H2 priority subtype), {label} ---")
        print(f"  {sens_keys[0]}={edh[sens_keys[0]]:.4f}  {sens_keys[1]}={edh[sens_keys[1]]:.4f}  "
              f"delta={edh['sensitivity_delta']*100 if not np.isnan(edh['sensitivity_delta']) else float('nan'):+.2f}pp")

    print(f"\nsize-tercile stratification: {strat['baseline2'].get('size_stratification_note', 'n/a')}")

    out = RUNS_DIR / "baselines4ab_internal_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
