# AUTO-GENERATED bhsd_excluded variant of score_baseline5_internal.py, created
# 2026-09-12 to fix the BHSD/RSNA patient-overlap scoring bug.
# Root cause: score_baseline5_internal.py was run against
# runs/baselines_bhsd_excluded/ WITHOUT bhsd_exclusion=True in
# load_val_split()'s Namespace, so build_folds() used the full
# (un-excluded) RSNA pool -> 2378 patients instead of the correct
# 2370.  This produced a contaminated val set and an inflated B5
# AUROC of 0.959 vs B2 0.984 (wrong fold) instead of the correct
# pair on n=2370 patients.
#
# Only functional change from the original: load_val_split()'s
# Namespace now sets bhsd_exclusion=True so it reconstructs the
# SAME (patient-excluded) RSNA fold split used to train the
# runs/baselines_bhsd_excluded checkpoints this script is meant to
# score.  Point --runs-dir at runs/baselines_bhsd_excluded and
# --wicl-dir at runs/baselines_bhsd_excluded/decisive_wicl when
# invoking.  See score_baseline5_internal.py (the unmodified
# original, used for runs/baselines_v1 scoring) for full methodology
# documentation.
"""Score baseline 5 (semi_supervised, teacher-student self-training
reimplementation of Lin and Yuh [34]) on the RSNA held-out validation fold
(same patient-grouped fold used at training time), per manuscript
Section IV-E / VI-A, and compare against baseline 2 (fixed_three_window).

BHSD-EXCLUDED VARIANT: val-fold built with bhsd_exclusion=True (2370
patients, not the original 2378). Run with --runs-dir
runs/baselines_bhsd_excluded.
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
from ich_gen.models.wicl import build_wicl_model
from ich_gen.evaluate import EvalInputs, evaluate_pooled_and_stratified
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
from ich_gen import train as train_mod

ANY_IDX = LABEL_COLUMNS.index("any")

# condition key -> dir name / view_mode
CONDITIONS = {
    "baseline2_fixed_three": "fixed_three_window",
    "baseline5_selftraining": "fixed_three_window",
}


def load_val_split(config: dict):
    ns = argparse.Namespace(
        rsna_root=Path(config["rsna_root"]),
        max_samples=config["max_samples"],
        n_folds=config["n_folds"],
        seed=config["seed"],
        bhsd_exclusion=True,  # BHSD/RSNA overlap remediation (2026-09-11) -- see
                              # ich_gen/train.py build_folds() docstring: this
                              # must be explicit; the default for a hand-built
                              # Namespace (vs one from build_argparser()) is
                              # False, to keep OLD scoring scripts reconstructing
                              # their checkpoints' exact original (unexcluded)
                              # folds if ever rerun.  This script is the
                              # post-remediation variant: it scores the
                              # bhsd_excluded retrained checkpoints and must
                              # therefore use the SAME excluded fold.
    )
    samples, folds = train_mod.build_folds(ns)
    fold = folds[config["fold"]]
    val_samples = [samples[i] for i in fold.val_idx]
    return val_samples


@torch.no_grad()
def score_checkpoint(run_key: str, out_dir: Path, config: dict, val_samples: list, device: str):
    view_mode = CONDITIONS[run_key]
    # weights_only=False: our own checkpoint, produced locally on this box
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

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


@torch.no_grad()
def score_wicl(wicl_dir: Path, val_samples: list, device: str):
    """Score the bhsd_excluded WICL checkpoint on the same held-out fold."""
    with open(wicl_dir / "config.json") as f:
        config = json.load(f)
    ckpt = torch.load(wicl_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    model = build_wicl_model(config["backbone"], pretrained=False,
                              lambda_consistency=config["lambda_consistency"],
                              lambda_embedding=config["lambda_embedding"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    val_ds = MultiSiteICHDataset(val_samples, view_mode="fixed_three_window",
                                  seed=config["seed"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=4)
    all_probs, all_labels, all_patients = [], [], []
    for batch in val_loader:
        x = batch["image"].to(device)
        logits = model.forward_single(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())
        all_patients.extend(batch["patient_id"])
    return (np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0),
            np.array(all_patients))


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
    ap.add_argument("--runs-dir", type=Path,
                     default=Path("runs/baselines_bhsd_excluded"))
    ap.add_argument("--wicl-dir", type=Path,
                     default=Path("runs/baselines_bhsd_excluded/decisive_wicl"))
    ap.add_argument("--skip-wicl", action="store_true",
                     help="skip the supplementary WICL comparison")
    cli = ap.parse_args()
    RUNS_DIR = cli.runs_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = {}
    for key in CONDITIONS:
        with open(RUNS_DIR / key / "config.json") as f:
            configs[key] = json.load(f)

    # baseline5's config.json carries extra semi_supervised-only keys
    # (unlabeled_fraction, teacher_epochs) not present in baseline2's
    # config -- restrict equality check to keys both share.
    shared_keys = ("rsna_root", "fold", "n_folds", "max_samples", "seed", "backbone",
                   "epochs", "batch_size", "lr", "weight_decay")
    base_cfg = configs["baseline2_fixed_three"]
    for key in ("baseline5_selftraining",):
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
    for key in ("baseline5_selftraining",):
        assert np.array_equal(ref["pat"], scored[key]["pat"]), f"val ordering mismatch: {key}"
        assert np.allclose(ref["y"], scored[key]["y"]), f"label mismatch: {key}"

    y_any = ref["y"][:, ANY_IDX]
    results = {
        "config": {k: base_cfg[k] for k in shared_keys},
        "baseline5_unlabeled_fraction": configs["baseline5_selftraining"].get("unlabeled_fraction"),
        "baseline5_teacher_epochs": configs["baseline5_selftraining"].get("teacher_epochs"),
        "n_val_slices": int(len(y_any)),
        "n_val_patients": int(len(set(ref["pat"].tolist()))),
        "scoring_variant": "bhsd_excluded",
        "bhsd_exclusion_note": (
            "Val fold built with bhsd_exclusion=True (8 BHSD/RSNA-overlapping "
            "patients excluded from the RSNA pool before fold construction). "
            "Correct n_val_patients is 2370, not 2378 as in the contaminated "
            "run from score_baseline5_internal.py (which lacked this flag)."),
        "documented_simplification_note": (
            "baseline5's unlabeled pool is RSNA's own withheld-label "
            "patient-disjoint subset of this training fold (--unlabeled-fraction), "
            "not a genuinely separate/external unlabeled corpus -- see "
            "ich_gen.train.run_semi_supervised()'s docstring and models/"
            "semi_supervised.py's module docstring for the full, corrected "
            "characterization against [34]'s full text."
        ),
    }

    print("\n=== RSNA in-distribution (BHSD-excluded), held-out fold %d: pairwise pooled 'any' ICH ===" % base_cfg["fold"])
    pairs = [
        ("baseline5", "baseline2", "baseline5_selftraining", "baseline2_fixed_three"),
    ]
    results["pairwise_pooled_any"] = {}
    for na, nb, ka, kb in pairs:
        pa = scored[ka]["p"][:, ANY_IDX]
        pb = scored[kb]["p"][:, ANY_IDX]
        results["pairwise_pooled_any"][f"{na}_vs_{nb}"] = pairwise_report(na, y_any, pa, nb, pb)

    print("\n=== stratified (evaluate_pooled_and_stratified) ===")
    strat = {}
    for key, short in (("baseline2_fixed_three", "baseline2"),
                        ("baseline5_selftraining", "baseline5")):
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

    print("\n--- subtype table: baseline5 vs baseline2 ---")
    results["subtype_baseline5_vs_baseline2"] = subtype_table(
        "baseline5", strat["baseline5"], "baseline2", strat["baseline2"])

    edh = results["subtype_baseline5_vs_baseline2"]["epidural"]
    keys = list(edh.keys())
    sens_keys = [k for k in keys if k.startswith("sensitivity_") and k != "sensitivity_delta"]
    print("\n--- EDH (H1/H2 priority subtype), baseline5 vs baseline2 ---")
    print(f"  {sens_keys[0]}={edh[sens_keys[0]]:.4f}  {sens_keys[1]}={edh[sens_keys[1]]:.4f}  "
          f"delta={edh['sensitivity_delta']*100 if not np.isnan(edh['sensitivity_delta']) else float('nan'):+.2f}pp")

    print(f"\nsize-tercile stratification: {strat['baseline2'].get('size_stratification_note', 'n/a')}")

    if not cli.skip_wicl and cli.wicl_dir.exists():
        print("\n=== SUPPLEMENTARY: WICL vs baseline 5 (bhsd_excluded) ===")
        y_w, p_w, pat_w = score_wicl(cli.wicl_dir, val_samples, device)
        assert np.array_equal(ref["pat"], pat_w), "WICL val ordering mismatch"
        assert np.allclose(ref["y"], y_w), "WICL label mismatch"
        p_w_any = p_w[:, ANY_IDX]
        results["supplementary_wicl_pairwise_pooled_any"] = {}
        pa = scored["baseline5_selftraining"]["p"][:, ANY_IDX]
        results["supplementary_wicl_pairwise_pooled_any"]["wicl_vs_baseline5"] = \
            pairwise_report("wicl", y_any, p_w_any, "baseline5", pa)
        results["supplementary_wicl_scope_note"] = (
            "Pooled-AUROC-only DeLong comparison of WICL against baseline "
            "5 on this internal fold (bhsd_excluded), using the bhsd_excluded "
            "WICL checkpoint from runs/baselines_bhsd_excluded/decisive_wicl "
            "(read-only, not retrained here). Partial contribution toward "
            "Section VI-C's H2 comparison set.")
    else:
        print("\n(skipping supplementary WICL comparison: --skip-wicl set or "
              f"{cli.wicl_dir} not found)")

    out = RUNS_DIR / "baselines5_internal_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
