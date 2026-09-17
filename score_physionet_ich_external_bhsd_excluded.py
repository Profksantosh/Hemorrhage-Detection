"""Score all seven audited checkpoints (WICL + baselines 1/2/3a/3b/4a/4b),
plus the read-only baseline-4c checkpoint needed to reproduce the
already-established WICL-vs-4c comparison pair, on the external
PhysioNet-ICH site (manuscript Section IV-C -- the cheap SECONDARY
external stress test).

INFERENCE ONLY. No checkpoint is retrained, fine-tuned, or otherwise
modified by this script; every run_key below is loaded strictly
read-only from its existing runs/ directory (runs/decisive_v2/decisive_wicl,
runs/decisive_v2/decisive_random_window_single, and all six directories
under runs/baselines_v1/). Nothing under runs/decisive*, runs/baselines_v1/
{baseline1_fixed_single,...,baseline4b_coral}, or the CQ500 label/leakage
files is written to by this script -- the only artifact written is this
script's own new output JSON (see bottom of main()).

WHY THIS SITE IS DIFFERENT FROM CQ500 (read before interpreting output)
=========================================================================
1. PhysioNet-ICH (Hssayeni et al., manuscript ref [29]) ships GENUINE
   PER-SLICE diagnosis labels (hemorrhage_diagnosis_raw_ct.csv has one row
   per (PatientNumber, SliceNumber) with its own subtype flags) -- unlike
   CQ500, which only has STUDY-level labels propagated down to every slice
   (prepare_cq500_labels.py DECISION 4). So, unlike every CQ500-external
   scoring script in this repo, the SLICE-level report below is the
   genuinely fine-grained ground truth for this site, not a noisy
   propagated proxy -- do not carry over CQ500's "noisy: study labels
   propagated" caveat language to this site, it does not apply here.
   Slice-level AUROC/DeLong IS still computed on pseudo-replicated
   (within-patient-correlated) samples, though, so the STUDY/PATIENT-level
   aggregated report (agg=max/mean per patient) is still reported
   alongside it, for the different reason of avoiding within-patient
   correlation inflating the effective sample size that DeLong's test
   assumes -- not because slice-level is dishonest here.
2. Severe small-n. PhysioNet-ICH v1.3.1 releases 75 of the original 82
   patients (patients #59-65 are confirmed missing from the public
   release; physionet_ich.py docstring). Of those 75: 36 have >=1
   ICH-positive slice, 39 are entirely negative. Per-subtype POSITIVE
   PATIENT counts (computed directly from hemorrhage_diagnosis_raw_ct.csv
   at script-run time, and also emitted into this script's output JSON
   under "physionet_ich_small_n_caveats") are far smaller still -- e.g.
   subdural=4 and intraventricular=5 positive patients pooled across the
   WHOLE site. This is more severe than the CQ500 EDH n=12 caveat already
   flagged elsewhere in this project: several PhysioNet-ICH subtypes here
   have FEWER THAN 10 positive PATIENTS in total, not just for one
   subgroup. Subtype-level AUROC/DeLong/sensitivity numbers at this n
   should be read as exploratory/descriptive only, not as a basis for any
   generalization claim on their own -- consistent with how
   physionet_ich.py's own docstring already instructs (citing Chen et
   al. [8]'s treatment of the same dataset) and consistent with this
   project's standing "no result is final until it is real data + a clean
   patient-level split" policy (there is no split to leak here: this
   script performs NO training and NO patient-level split of its own --
   every one of PhysioNet-ICH's 75 patients is used, once, as external
   test-only data, exactly as CQ500 is used elsewhere in this repo).
3. No lesion_size_mm3 for this site (physionet_ich.py explicitly sets it
   to None; the masks/ directory would support a size computation but
   that conversion is out of scope for this loader, matching every other
   external site's size-tercile-stratification limitation in this repo)
   -- size-tercile stratification is NaN here by design, not omission;
   BHSD remains this manuscript's sole size-stratification source.

PAIRWISE COMPARISONS (mirrors the CQ500-external comparison set already
audited across score_baselines_external.py, score_baselines3ab_external.py,
score_baselines4ab_external.py, and score_external.py)
=========================================================================
  * baseline1 vs baseline2       (score_baselines_external.py)
  * baseline3a vs baseline2, baseline3b vs baseline2, baseline3a vs
    baseline3b                    (score_baselines3ab_external.py)
  * baseline4a vs baseline2, baseline4b vs baseline2, baseline4a vs
    baseline4b                    (score_baselines4ab_external.py)
  * WICL vs baseline4c (runs/decisive_v2/decisive_random_window_single,
    read-only, loaded here ONLY for this comparison, exactly as
    score_external.py does)       (score_external.py)
  * WICL vs baseline3a, WICL vs baseline3b, supplementary, same partial-
    contribution-to-H2 caveat as score_baselines3ab_external.py's
    supplementary section)

Usage
-----
    python score_physionet_ich_external.py \
        --runs-dir runs/baselines_v1 --physionet-root data/physionet_ich
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
from ich_gen.models.backbone import build_backbone, EmbeddingBackbone
from ich_gen.models.adaptive_window import WEMClassifier
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA

ANY_IDX = LABEL_COLUMNS.index("any")

# run_key -> (view_mode, model_kind, out_dir-relative-to-{baselines_dir,decisive_dir})
# model_kind in {"plain", "wem", "embedding", "wicl"}
BASELINE_CONDITIONS = {
    "baseline1_fixed_single": ("fixed_single_window", "plain"),
    "baseline2_fixed_three": ("fixed_three_window", "plain"),
    "baseline3a_hrt": ("adaptive_hrt", "plain"),
    "baseline3b_wem": ("raw_hu_for_wem", "wem"),
    "baseline4a_augmentation": ("fixed_three_window", "plain"),
    "baseline4b_coral": ("fixed_three_window", "embedding"),
    # baseline5 (semi_supervised/self-training, manuscript Section VI-A):
    # CONDITION_VIEW_MODE["semi_supervised"] == "fixed_three_window"
    # (ich_gen/train.py), and run_semi_supervised()'s returned student is
    # built via build_backbone(...) -- i.e. the same "plain" architecture
    # as baseline1/2/4a, just warm-started + trained with the teacher-
    # student loss. load_model()'s "plain" branch (build_backbone +
    # load_state_dict) loads it without any surgery. Verified against
    # runs/baselines_bhsd_excluded/baseline5_selftraining/config.json
    # (condition="semi_supervised", backbone="densenet121") rather than
    # assumed.
    "baseline5_selftraining": ("fixed_three_window", "plain"),
}

SHORT_NAMES = {
    "baseline1_fixed_single": "baseline1",
    "baseline2_fixed_three": "baseline2",
    "baseline3a_hrt": "baseline3a",
    "baseline3b_wem": "baseline3b",
    "baseline4a_augmentation": "baseline4a",
    "baseline4b_coral": "baseline4b",
    "baseline4c_random_window_single": "baseline4c",
    "baseline5_selftraining": "baseline5",
    "wicl": "wicl",
}


def load_model(view_mode: str, model_kind: str, out_dir: Path, config: dict, device: str):
    # weights_only=False: our own checkpoints, produced locally on this box
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    if model_kind == "wem":
        model = WEMClassifier(config["backbone"], pretrained=False)
    elif model_kind == "embedding":
        model = EmbeddingBackbone(config["backbone"], pretrained=False)
    elif model_kind == "wicl":
        model = build_wicl_model(config["backbone"], pretrained=False,
                                  lambda_consistency=config["lambda_consistency"],
                                  lambda_embedding=config["lambda_embedding"])
    else:
        model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


@torch.no_grad()
def score(model, view_mode: str, model_kind: str, samples: list, config: dict,
          device: str, batch_size: int = 64):
    ds = MultiSiteICHDataset(samples, view_mode=view_mode, seed=config["seed"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8)
    probs, labels, patients, sizes = [], [], [], []
    for batch in loader:
        if model_kind == "wem":
            hu_for_wem = batch["hu_for_wem"].to(device)
            hu_raw = batch["hu_raw"].to(device)
            logits = model(hu_for_wem, hu_raw)
        elif model_kind == "embedding":
            logits, _emb = model(batch["image"].to(device))
        elif model_kind == "wicl":
            logits = model.forward_single(batch["image"].to(device))
        else:
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
    """Aggregate slice probabilities to one value per PATIENT (PhysioNet-ICH
    calls its unit a 'patient', not a 'study', but the mechanics are
    identical to the CQ500 scripts' study_level helper). Done here to avoid
    within-patient correlation inflating the effective n that DeLong's test
    assumes -- NOT because slice-level ground truth is unreliable for this
    site (it is genuine per-slice ground truth here; see module docstring
    point 1)."""
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


def subtype_table(name_a, rec_a, name_b, rec_b, patient_counts):
    print(f"\n{'subtype':<18}{'n_pos_pat':>10}{'AUROC(' + name_a + ')':>18}{'AUROC(' + name_b + ')':>18}"
          f"{'sens(' + name_a + ')':>14}{'sens(' + name_b + ')':>14}{'delta(pp)':>12}  flag")
    summary = {}
    for subtype in SUBTYPES:
        a1 = rec_a[f"auroc_{subtype}"]; a2 = rec_b[f"auroc_{subtype}"]
        s1 = rec_a[f"sensitivity_{subtype}"]; s2 = rec_b[f"sensitivity_{subtype}"]
        delta = (s1 - s2) if not (np.isnan(s1) or np.isnan(s2)) else float("nan")
        clinically_relevant = (not np.isnan(delta)) and abs(delta) >= MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
        n_pos_pat = patient_counts[subtype]
        severe_small_n = n_pos_pat < 10
        flag_bits = []
        if not np.isnan(delta) and clinically_relevant:
            flag_bits.append(">=5pp delta")
        if severe_small_n:
            flag_bits.append(f"SEVERE SMALL-N (n_pos_pat={n_pos_pat})")
        flag = " *** " + ", ".join(flag_bits) + " ***" if flag_bits else ""
        print(f"{subtype:<18}{n_pos_pat:>10d}{a1:>18.4f}{a2:>18.4f}{s1:>14.4f}{s2:>14.4f}"
              f"{delta*100 if not np.isnan(delta) else float('nan'):>12.2f}  {flag}")
        summary[subtype] = {
            f"auroc_{name_a}": a1, f"auroc_{name_b}": a2,
            f"sensitivity_{name_a}": s1, f"sensitivity_{name_b}": s2,
            "sensitivity_delta": delta,
            "meets_min_clinically_relevant_sensitivity_delta": clinically_relevant,
            "n_positive_patients": n_pos_pat,
            "severe_small_n_flag": severe_small_n,
        }
    return summary


def compute_subtype_patient_counts(y_true: np.ndarray, patients: np.ndarray) -> dict:
    """Per-subtype count of DISTINCT PATIENTS with >=1 positive slice,
    pooled across the whole PhysioNet-ICH site -- the small-n number that
    actually matters for interpreting subtype AUROC/sensitivity here, since
    a single patient can contribute many (correlated) positive slices."""
    counts = {}
    for subtype in SUBTYPES:
        idx = LABEL_COLUMNS.index(subtype)
        pos_mask = y_true[:, idx] == 1
        counts[subtype] = int(len(np.unique(patients[pos_mask]))) if pos_mask.any() else 0
    any_mask = y_true[:, ANY_IDX] == 1
    counts["any"] = int(len(np.unique(patients[any_mask]))) if any_mask.any() else 0
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/baselines_v1"),
                     help="directory holding baseline1/2/3a/3b/4a/4b checkpoints")
    ap.add_argument("--decisive-dir", type=Path, default=Path("runs/decisive_v2"),
                     help="directory holding decisive_wicl and "
                          "decisive_random_window_single (baseline4c) checkpoints")
    ap.add_argument("--physionet-root", type=Path, default=Path("data/physionet_ich"))
    ap.add_argument("--max-physionet-slices", type=int, default=None,
                     help="truncate the sample list for a quick smoke test; "
                          "omit for the real, full-site run")
    ap.add_argument("--skip-wicl", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load baseline1/2/3a/3b/4a/4b configs + checkpoints (read-only) ---
    configs, models = {}, {}
    for key, (view_mode, model_kind) in BASELINE_CONDITIONS.items():
        out_dir = args.runs_dir / key
        with open(out_dir / "config.json") as f:
            configs[key] = json.load(f)
        models[key] = (load_model(view_mode, model_kind, out_dir, configs[key], device),
                        view_mode, model_kind)

    print("building PhysioNet-ICH external test set (REAL data, no synthetic "
          "fallback -- build_samples() raises if any expected file is "
          "missing) ...")
    pn_samples = build_physionet_ich(args.physionet_root)
    if args.max_physionet_slices:
        pn_samples = pn_samples[:args.max_physionet_slices]
    n_patients_total = len({s.patient_id for s in pn_samples})
    print(f"PhysioNet-ICH: {len(pn_samples)} slices, {n_patients_total} patients "
          f"(75 patients / 2814 slices expected for the full real site per "
          f"physionet_ich_loader_verification.log; if --max-physionet-slices "
          f"was NOT passed and these numbers differ, treat that as a real "
          f"discrepancy to investigate, not something to silently proceed past)")

    results = {}
    results["physionet_ich_dataset_summary"] = {
        "n_slices": len(pn_samples),
        "n_patients": n_patients_total,
        "note": ("PhysioNet-ICH v1.3.1 releases 75 of the original 82 patients "
                 "(#59-65 confirmed missing). This is REAL data (restricted-"
                 "access, SHA256-verified download; see "
                 "physionet_ich_structure_verification.log and "
                 "physionet_ich_loader_verification.log), used here ONLY as "
                 "external test data -- no training and no patient-level "
                 "split performed by this script."),
    }

    scored = {}
    for key, (model, view_mode, model_kind) in models.items():
        p, y, pat, sz = score(model, view_mode, model_kind, pn_samples, configs[key], device)
        scored[key] = dict(p=p, y=y, pat=pat, sz=sz)

    ref = scored["baseline2_fixed_three"]
    for key in BASELINE_CONDITIONS:
        if key == "baseline2_fixed_three":
            continue
        assert np.array_equal(ref["pat"], scored[key]["pat"]) and np.allclose(ref["y"], scored[key]["y"]), \
            f"sample ordering/label mismatch: {key}"

    y_any = ref["y"][:, ANY_IDX]
    subtype_patient_counts = compute_subtype_patient_counts(ref["y"], ref["pat"])
    print("\n=== PhysioNet-ICH per-subtype POSITIVE PATIENT counts "
          "(pooled across the whole 75-patient site) ===")
    for subtype in SUBTYPES:
        n = subtype_patient_counts[subtype]
        flag = " *** SEVERE SMALL-N (<10 positive patients) ***" if n < 10 else ""
        print(f"  {subtype:<18} n_positive_patients={n:>3d}{flag}")
    print(f"  {'any (pooled)':<18} n_positive_patients={subtype_patient_counts['any']:>3d} "
          f"/ {n_patients_total} total patients")
    results["physionet_ich_small_n_caveats"] = {
        "n_patients_total": n_patients_total,
        "n_slices_total": len(pn_samples),
        "subtype_positive_patient_counts": subtype_patient_counts,
        "severe_small_n_subtypes": [s for s in SUBTYPES if subtype_patient_counts[s] < 10],
        "note": ("Subtype-level AUROC/sensitivity/DeLong results for any "
                 "subtype flagged here (<10 positive patients pooled across "
                 "the WHOLE site, before any comparison-specific subsetting) "
                 "are exploratory/descriptive only and must not be reported "
                 "as a standalone generalization claim -- this is a more "
                 "severe small-n regime than the CQ500 EDH n=12 caveat "
                 "already flagged elsewhere in this project."),
    }

    # --- pairwise comparisons: each baseline vs baseline2 ---
    pairs_vs_baseline2 = [
        ("baseline1_fixed_single", "baseline2_fixed_three"),
        ("baseline3a_hrt", "baseline2_fixed_three"),
        ("baseline3b_wem", "baseline2_fixed_three"),
        ("baseline4a_augmentation", "baseline2_fixed_three"),
        ("baseline4b_coral", "baseline2_fixed_three"),
    ]
    # --- established within-family comparisons (Section VI-C H2 names the
    # adaptive-window 3a/3b pair explicitly; 4a/4b compared the same way in
    # score_baselines4ab_external.py) ---
    within_family_pairs = [
        ("baseline3a_hrt", "baseline3b_wem"),
        ("baseline4a_augmentation", "baseline4b_coral"),
    ]

    for ka, kb in pairs_vs_baseline2 + within_family_pairs:
        na, nb = SHORT_NAMES[ka], SHORT_NAMES[kb]
        pair_key = f"{na}_vs_{nb}"
        results[pair_key] = {}
        results[pair_key]["slice_level"] = report(
            "PhysioNet-ICH external -- SLICE level (genuine per-slice ground truth)",
            na, y_any, scored[ka]["p"][:, ANY_IDX], nb, scored[kb]["p"][:, ANY_IDX])
        for agg in ("max", "mean"):
            ys, ps_a = study_level(y_any, scored[ka]["p"][:, ANY_IDX], scored[ka]["pat"], agg)
            _, ps_b = study_level(y_any, scored[kb]["p"][:, ANY_IDX], scored[kb]["pat"], agg)
            results[pair_key][f"patient_level_{agg}"] = report(
                f"PhysioNet-ICH external -- PATIENT level (agg={agg})", na, ys, ps_a, nb, ps_b)

    print("\n=== PhysioNet-ICH external -- SUBTYPE/SIZE-STRATIFIED "
          "(evaluate_pooled_and_stratified, slice level: genuine per-slice "
          "ground truth for this site, unlike CQ500) ===")
    strat = {}
    for key in BASELINE_CONDITIONS:
        short = SHORT_NAMES[key]
        s = scored[key]
        rec = evaluate_pooled_and_stratified(
            EvalInputs(y_true=s["y"], y_prob=s["p"], patient_ids=s["pat"],
                       lesion_size_mm3=s["sz"], condition=key, site="physionet_ich"),
            size_thresholds=None)
        strat[short] = rec
        print(f"  {short}: pooled AUROC={rec['pooled_auroc']:.4f}  "
              f"pooled sens@0.5={rec['pooled_sensitivity']:.4f} {rec['pooled_sensitivity_ci']}")
    results["physionet_ich_stratified"] = strat

    subtype_pairs = [(SHORT_NAMES[ka], SHORT_NAMES[kb]) for ka, kb in
                      pairs_vs_baseline2 + within_family_pairs]
    for na, nb in subtype_pairs:
        print(f"\n--- subtype table: {na} vs {nb} ---")
        key = f"physionet_ich_subtype_{na}_vs_{nb}"
        summary = subtype_table(na, strat[na], nb, strat[nb], subtype_patient_counts)
        results[key] = summary
        edh = summary["epidural"]
        sens_keys = [k for k in edh.keys() if k.startswith("sensitivity_") and k != "sensitivity_delta"]
        print(f"\n--- EDH (primary H1/H2 subtype of interest), {na} vs {nb} "
              f"(n_positive_patients={edh['n_positive_patients']}"
              f"{' -- SEVERE SMALL-N' if edh['severe_small_n_flag'] else ''}) ---")
        print(f"  {sens_keys[0]}={edh[sens_keys[0]]:.4f}  {sens_keys[1]}={edh[sens_keys[1]]:.4f}  "
              f"delta={edh['sensitivity_delta']*100 if not np.isnan(edh['sensitivity_delta']) else float('nan'):+.2f}pp "
              f"(threshold: {MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA*100:.1f}pp)")

    print(f"\nsize-tercile stratification: {strat['baseline2'].get('size_stratification_note', 'n/a')}")

    # --- WICL comparisons: vs baseline4c (main, mirrors score_external.py),
    # vs baseline3a/3b (supplementary, mirrors score_baselines3ab_external.py) ---
    wicl_dir = args.decisive_dir / "decisive_wicl"
    # NOTE (bhsd_excluded fork): in runs/baselines_bhsd_excluded the
    # baseline4c checkpoint directory is named "baseline4c_random_window_single",
    # not "decisive_random_window_single" (the runs/decisive_v2 name used by
    # the original v1 script). decisive_wicl's directory name is unchanged
    # and needs no fix.
    baseline4c_dir = args.decisive_dir / "baseline4c_random_window_single"
    if not args.skip_wicl and wicl_dir.exists() and baseline4c_dir.exists():
        print(f"\n=== WICL ({wicl_dir}, read-only) vs "
              f"baseline4c ({baseline4c_dir}, "
              "read-only) and vs baseline3a/3b on PhysioNet-ICH ===")
        with open(wicl_dir / "config.json") as f:
            wicl_config = json.load(f)
        wicl_model = load_model("fixed_three_window", "wicl", wicl_dir, wicl_config, device)
        p_w, y_w, pat_w, sz_w = score(wicl_model, "fixed_three_window", "wicl",
                                       pn_samples, wicl_config, device)
        assert np.array_equal(ref["pat"], pat_w) and np.allclose(ref["y"], y_w), \
            "WICL sample ordering/label mismatch"

        with open(baseline4c_dir / "config.json") as f:
            b4c_config = json.load(f)
        b4c_model = load_model("fixed_three_window", "plain", baseline4c_dir, b4c_config, device)
        p_4c, y_4c, pat_4c, sz_4c = score(b4c_model, "fixed_three_window", "plain",
                                           pn_samples, b4c_config, device)
        assert np.array_equal(ref["pat"], pat_4c) and np.allclose(ref["y"], y_4c), \
            "baseline4c sample ordering/label mismatch"

        p_w_any = p_w[:, ANY_IDX]
        wicl_targets = [("baseline4c_random_window_single", p_4c[:, ANY_IDX], pat_4c),
                         ("baseline3a_hrt", scored["baseline3a_hrt"]["p"][:, ANY_IDX], scored["baseline3a_hrt"]["pat"]),
                         ("baseline3b_wem", scored["baseline3b_wem"]["p"][:, ANY_IDX], scored["baseline3b_wem"]["pat"])]
        for key, p_target, pat_target in wicl_targets:
            na = SHORT_NAMES[key]
            pair_key = f"wicl_vs_{na}"
            results[pair_key] = {}
            results[pair_key]["slice_level"] = report(
                "PhysioNet-ICH external -- SLICE level (genuine per-slice ground truth)",
                "wicl", y_any, p_w_any, na, p_target)
            for agg in ("max", "mean"):
                ys, ps_w = study_level(y_any, p_w_any, pat_w, agg)
                _, ps_t = study_level(y_any, p_target, pat_target, agg)
                results[pair_key][f"patient_level_{agg}"] = report(
                    f"PhysioNet-ICH external -- PATIENT level (agg={agg})", "wicl", ys, ps_w, na, ps_t)

        rec_wicl = evaluate_pooled_and_stratified(
            EvalInputs(y_true=y_w, y_prob=p_w, patient_ids=pat_w, lesion_size_mm3=sz_w,
                       condition="wicl", site="physionet_ich"),
            size_thresholds=None)
        rec_4c = evaluate_pooled_and_stratified(
            EvalInputs(y_true=y_4c, y_prob=p_4c, patient_ids=pat_4c, lesion_size_mm3=sz_4c,
                       condition="baseline4c_random_window_single", site="physionet_ich"),
            size_thresholds=None)
        strat["wicl"] = rec_wicl
        strat["baseline4c"] = rec_4c
        results["physionet_ich_stratified"]["wicl"] = rec_wicl
        results["physionet_ich_stratified"]["baseline4c"] = rec_4c

        for na, nb in (("wicl", "baseline4c"), ("wicl", "baseline3a"), ("wicl", "baseline3b")):
            print(f"\n--- subtype table: {na} vs {nb} ---")
            key = f"physionet_ich_subtype_{na}_vs_{nb}"
            summary = subtype_table(na, strat[na], nb, strat[nb], subtype_patient_counts)
            results[key] = summary

        results["wicl_scope_note"] = (
            "WICL vs baseline4c mirrors score_external.py's main CQ500 "
            "comparison; WICL vs baseline3a/3b mirrors "
            "score_baselines3ab_external.py's supplementary CQ500 section. "
            "Both checkpoints (decisive_wicl, decisive_random_window_single) "
            "are the same audited, read-only checkpoints used in those prior "
            "CQ500 runs -- not retrained here. As in that script, this is a "
            "partial contribution toward Section VI-C's full H2 comparison "
            "set, not the full Bonferroni-corrected test.")
    else:
        print(f"\n(skipping WICL comparisons: --skip-wicl set, or {wicl_dir} "
              f"or {baseline4c_dir} not found)")

    out = args.runs_dir / "physionet_ich_external_result.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
