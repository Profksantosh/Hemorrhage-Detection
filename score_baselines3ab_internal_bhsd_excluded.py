# AUTO-GENERATED bhsd_excluded variant of score_baselines3ab_internal.py, created
# 2026-09-11 for the BHSD/RSNA patient-overlap remediation task.
# Only functional change from the original: load_val_split()'s
# Namespace now sets bhsd_exclusion=True so it reconstructs the
# SAME (patient-excluded) RSNA fold split used to train the
# runs/baselines_bhsd_excluded / runs/decisive_bhsd_excluded
# checkpoints this script is meant to score. Point --runs-dir
# (and --wicl-dir / --baseline2-dir / --decisive-dir, if this
# script has one) at the bhsd_excluded run directories via CLI
# flags when invoking. See score_baselines3ab_internal.py (the unmodified original)
# for full methodology documentation -- not duplicated here.
"""Score baseline 3a (adaptive_hrt) and baseline 3b (adaptive_wem) on the
RSNA held-out validation fold (same patient-grouped fold used at training
time), per manuscript Section IV-E / VI-A, and compare each against
baseline 2 (fixed_three_window) and against each other (Section VI-C's
H2 pairwise-comparison requirement, which explicitly names "the two
adaptive-window variants 3a/3b" as part of its comparison set).

Generalizes score_baselines4ab_internal.py (baseline2 vs 4a/4b) to
baseline3a/3b, reusing its exact scoring methodology (own training-time
view_mode per condition -- here baseline 3a uses "adaptive_hrt" (a
per-slice heuristic window applied at dataset time, plain classifier
forward) and baseline 3b uses "raw_hu_for_wem" (the dataset returns raw,
un-windowed HU; the WindowEstimatorModule inside WEMClassifier performs
the actual windowing at forward time, so its forward signature is
(hu_for_wem, hu_raw) -> logits, not (image,) -> logits like every other
condition); DeLong's test for AUROC; evaluate_pooled_and_stratified for
subtype/size metrics; size_thresholds=None, matching every other
internal/external scoring script in this repo (RSNA/CQ500 carry no
lesion_size_mm3 -- BHSD is the size-stratification source, out of scope
here).

score_checkpoint() below handles the "wem" model-kind's two-input
forward pass explicitly, exactly as score_baselines4ab_internal.py
explicitly handles dg_coral's (logits, embedding) return-tuple, rather
than assuming a uniform single-tensor-in/single-tensor-out interface
across all conditions.

Additionally reports a SUPPLEMENTARY (not the full Section VI-C H2 test)
pairwise comparison of each of baseline 3a/3b against the audited WICL
checkpoint (runs/decisive_v2/decisive_wicl, read-only), since Section
VI-C's H2 comparison set explicitly includes the two adaptive-window
variants. This is NOT the full 45-comparison-per-pair,
Bonferroni-corrected H2 test specified in Section VI-C (that test needs
all baselines 1-5, all three external sites, and size-tercile
stratification via BHSD, none of which are all complete yet) -- it is a
single pooled-AUROC DeLong comparison per pair, reported as a partial,
early contribution toward H2, not a substitute for it.

Usage
-----
    python score_baselines3ab_internal.py --runs-dir runs/baselines_v1 \
        --wicl-dir runs/decisive_v2/decisive_wicl
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
from ich_gen.models.adaptive_window import WEMClassifier
from ich_gen.models.wicl import build_wicl_model
from ich_gen.evaluate import EvalInputs, evaluate_pooled_and_stratified
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
from ich_gen import train as train_mod

ANY_IDX = LABEL_COLUMNS.index("any")

# condition key -> (dir name, view_mode, model_kind)
CONDITIONS = {
    "baseline2_fixed_three": ("fixed_three_window", "plain"),
    "baseline3a_hrt": ("adaptive_hrt", "plain"),
    "baseline3b_wem": ("raw_hu_for_wem", "wem"),
}


def load_val_split(config: dict):
    ns = argparse.Namespace(
        rsna_root=Path(config["rsna_root"]),
        max_samples=config["max_samples"],
        n_folds=config["n_folds"],
        seed=config["seed"],
        bhsd_exclusion=True,  # BHSD/RSNA overlap remediation (2026-09-11) -- see
                              # ich_gen/train.py build_folds() docstring: this
                              # must be explicit, the default for a hand-built
                              # Namespace (as opposed to one from build_argparser())
                              # is False, to keep OLD (pre-remediation) scoring
                              # scripts reconstructing their checkpoints' exact
                              # original (unexcluded) folds if ever rerun.
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
    if model_kind == "wem":
        model = WEMClassifier(config["backbone"], pretrained=False)
    else:
        model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    val_ds = MultiSiteICHDataset(val_samples, view_mode=view_mode, seed=config["seed"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                             num_workers=4)

    all_probs, all_labels, all_patients, all_sizes = [], [], [], []
    for batch in val_loader:
        if model_kind == "wem":
            hu_for_wem = batch["hu_for_wem"].to(device)
            hu_raw = batch["hu_raw"].to(device)
            logits = model(hu_for_wem, hu_raw)
        else:
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
    """Score the audited WICL checkpoint (runs/decisive_v2/decisive_wicl,
    read-only -- not touched by this script) on the SAME held-out fold
    as baseline 2/3a/3b, using the identical eval-time convention
    score_decisive.py established: view_mode="fixed_three_window" (the
    fixed canonical composite, matching Section V-E's "WICL is trained
    with randomized windows but evaluated with the fixed canonical
    composite for fair, equal-inference-cost comparison") and
    model.forward_single (WICL's plain-classifier-equivalent forward
    path, not its dual-view training_step)."""
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
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/baselines_v1"))
    ap.add_argument("--wicl-dir", type=Path, default=Path("runs/decisive_v2/decisive_wicl"))
    ap.add_argument("--skip-wicl", action="store_true",
                     help="skip the supplementary WICL comparison (e.g. if "
                          "runs/decisive_v2 is unavailable)")
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
    for key in ("baseline3a_hrt", "baseline3b_wem"):
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
    for key in ("baseline3a_hrt", "baseline3b_wem"):
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
        ("baseline3a", "baseline2", "baseline3a_hrt", "baseline2_fixed_three"),
        ("baseline3b", "baseline2", "baseline3b_wem", "baseline2_fixed_three"),
        ("baseline3a", "baseline3b", "baseline3a_hrt", "baseline3b_wem"),
    ]
    results["pairwise_pooled_any"] = {}
    for na, nb, ka, kb in pairs:
        pa = scored[ka]["p"][:, ANY_IDX]
        pb = scored[kb]["p"][:, ANY_IDX]
        results["pairwise_pooled_any"][f"{na}_vs_{nb}"] = pairwise_report(na, y_any, pa, nb, pb)

    print("\n=== stratified (evaluate_pooled_and_stratified) ===")
    strat = {}
    for key, short in (("baseline2_fixed_three", "baseline2"),
                        ("baseline3a_hrt", "baseline3a"),
                        ("baseline3b_wem", "baseline3b")):
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

    print("\n--- subtype table: baseline3a vs baseline2 ---")
    results["subtype_baseline3a_vs_baseline2"] = subtype_table(
        "baseline3a", strat["baseline3a"], "baseline2", strat["baseline2"])
    print("\n--- subtype table: baseline3b vs baseline2 ---")
    results["subtype_baseline3b_vs_baseline2"] = subtype_table(
        "baseline3b", strat["baseline3b"], "baseline2", strat["baseline2"])
    print("\n--- subtype table: baseline3a vs baseline3b ---")
    results["subtype_baseline3a_vs_baseline3b"] = subtype_table(
        "baseline3a", strat["baseline3a"], "baseline3b", strat["baseline3b"])

    for label, summary in (("baseline3a vs baseline2", results["subtype_baseline3a_vs_baseline2"]),
                            ("baseline3b vs baseline2", results["subtype_baseline3b_vs_baseline2"]),
                            ("baseline3a vs baseline3b", results["subtype_baseline3a_vs_baseline3b"])):
        edh = summary["epidural"]
        keys = list(edh.keys())
        sens_keys = [k for k in keys if k.startswith("sensitivity_") and k != "sensitivity_delta"]
        print(f"\n--- EDH (H1/H2 priority subtype), {label} ---")
        print(f"  {sens_keys[0]}={edh[sens_keys[0]]:.4f}  {sens_keys[1]}={edh[sens_keys[1]]:.4f}  "
              f"delta={edh['sensitivity_delta']*100 if not np.isnan(edh['sensitivity_delta']) else float('nan'):+.2f}pp")

    print(f"\nsize-tercile stratification: {strat['baseline2'].get('size_stratification_note', 'n/a')}")

    if not cli.skip_wicl and cli.wicl_dir.exists():
        print("\n=== SUPPLEMENTARY: WICL (runs/decisive_v2/decisive_wicl, read-only) "
              "vs baseline 3a/3b -- partial contribution toward Section VI-C's H2 "
              "comparison set, NOT the full 45-comparison Bonferroni-corrected test ===")
        y_w, p_w, pat_w = score_wicl(cli.wicl_dir, val_samples, device)
        assert np.array_equal(ref["pat"], pat_w), "WICL val ordering mismatch"
        assert np.allclose(ref["y"], y_w), "WICL label mismatch"
        p_w_any = p_w[:, ANY_IDX]
        results["supplementary_wicl_pairwise_pooled_any"] = {}
        for na, ka in (("baseline3a", "baseline3a_hrt"), ("baseline3b", "baseline3b_wem")):
            pa = scored[ka]["p"][:, ANY_IDX]
            results["supplementary_wicl_pairwise_pooled_any"][f"wicl_vs_{na}"] = \
                pairwise_report("wicl", y_any, p_w_any, na, pa)
        results["supplementary_wicl_scope_note"] = (
            "Pooled-AUROC-only DeLong comparison of WICL against baseline "
            "3a/3b on this internal fold, using the audited "
            "runs/decisive_v2/decisive_wicl checkpoint (read-only, not "
            "retrained here). This is a partial, early contribution toward "
            "Section VI-C's H2 comparison set (which names the two "
            "adaptive-window variants explicitly) -- it is NOT the full "
            "45-comparison-per-pair (5 subtypes x 3 size terciles x 3 "
            "external sites), Bonferroni-corrected H2 test, which requires "
            "all baselines 1-5, all three external sites, and BHSD "
            "size-tercile data, none of which are all complete yet.")
    else:
        print("\n(skipping supplementary WICL comparison: --skip-wicl set or "
              f"{cli.wicl_dir} not found)")

    out = RUNS_DIR / "baselines3ab_internal_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
