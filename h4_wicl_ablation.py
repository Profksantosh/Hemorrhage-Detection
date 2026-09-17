"""WICL component ablation on H4 (manuscript revision, 2026-09-14): extends
h4_hu_recalibration_stress_test_bhsd_excluded.py's exact methodology (same
held-out patient-disjoint RSNA subset, same provisional perturbation grid,
same corner-degradation definition, same paired patient-level bootstrap) to
compare FOUR conditions instead of two: baseline2, full WICL
(lambda_c=1.0, lambda_e=0.5), prediction-consistency-only
(lambda_c=1.0, lambda_e=0.0), and embedding-consistency-only
(lambda_c=0.0, lambda_e=0.5) -- attributing H4's positive result to WICL's
two loss components individually.

Reuses select_holdout_samples(), score_grid(), corner_mean_degradation(),
and the bootstrap resampling exactly as in the audited H4 script; only the
condition set scored is extended from 2 to 4.

Usage
-----
    python h4_wicl_ablation.py --runs-dir runs/baselines_bhsd_excluded \
        --wicl-dir runs/baselines_bhsd_excluded/decisive_wicl \
        --pred-only-dir runs/baselines_bhsd_excluded/wicl_ablation_pred_only \
        --embed-only-dir runs/baselines_bhsd_excluded/wicl_ablation_embed_only \
        --rsna-root data/rsna/rsna-intracranial-hemorrhage-detection
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from ich_gen.datasets.common import MultiSiteICHDataset, LABEL_COLUMNS, SUBTYPES
from ich_gen.datasets.rsna import build_samples as build_rsna_samples
from ich_gen.models.backbone import build_backbone
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stress_test import make_perturbation_grid
from ich_gen.datasets.common import Sample

ANY_IDX = LABEL_COLUMNS.index("any")
N_BOOTSTRAP = 1000
SEED = 0
TRAIN_POOL_MAX_SAMPLES = 20000
HOLDOUT_RAW_N = 3000
HOLDOUT_FINAL_N = 2000


def safe_auroc(y_true_bin, y_score):
    if len(np.unique(y_true_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_bin, y_score))


def rel_deg(reference_val, perturbed_val):
    if reference_val is None or np.isnan(reference_val) or reference_val == 0:
        return float("nan")
    return float((reference_val - perturbed_val) / reference_val)


def load_baseline2(runs_dir: Path, device: str):
    b2_dir = runs_dir / "baseline2_fixed_three"
    with open(b2_dir / "config.json") as f:
        config = json.load(f)
    ckpt = torch.load(b2_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
    model = build_backbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, config, (lambda m, x: m(x))


def load_wicl(wicl_dir: Path, device: str):
    with open(wicl_dir / "config.json") as f:
        config = json.load(f)
    ckpt = torch.load(wicl_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
    model = build_wicl_model(config["backbone"], pretrained=False,
                              lambda_consistency=config["lambda_consistency"],
                              lambda_embedding=config["lambda_embedding"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, config, (lambda m, x: m.forward_single(x))


def select_holdout_samples(rsna_root: Path, seed_config: dict):
    from ich_gen.datasets.rsna import load_bhsd_overlap_exclusion_ids
    from ich_gen.train import DEFAULT_BHSD_EXCLUSION_JSON
    exclude_ids = load_bhsd_overlap_exclusion_ids(DEFAULT_BHSD_EXCLUSION_JSON)
    print(f"loaded {len(exclude_ids)} BHSD-overlap RSNA patient id(s) to exclude")

    pool_samples = build_rsna_samples(rsna_root, max_samples=TRAIN_POOL_MAX_SAMPLES,
                                       exclude_patient_ids=exclude_ids)
    pool_patients = {s.patient_id for s in pool_samples}
    print(f"  pool: {len(pool_samples)} slices, {len(pool_patients)} patients")

    raw_all = build_rsna_samples(rsna_root, max_samples=TRAIN_POOL_MAX_SAMPLES + HOLDOUT_RAW_N,
                                  exclude_patient_ids=exclude_ids)
    candidate = raw_all[TRAIN_POOL_MAX_SAMPLES:]
    assert len(candidate) == HOLDOUT_RAW_N

    filtered = [s for s in candidate if s.patient_id not in pool_patients]
    n_removed = len(candidate) - len(filtered)
    held_out = filtered[:HOLDOUT_FINAL_N]
    held_out_patients = {s.patient_id for s in held_out}
    assert held_out_patients.isdisjoint(pool_patients), "held-out set NOT patient-disjoint -- bug"
    print(f"  final held-out subset: {len(held_out)} slices, {len(held_out_patients)} patients")
    return held_out, n_removed


@torch.no_grad()
def score_grid(model, forward_fn, held_out, seed, perturbations, device, batch_size=64, num_workers=4):
    out = {}
    for pert in perturbations:
        key = (pert.scale, pert.offset)

        def _wrap(original_loader, p=pert):
            return lambda: p.apply(original_loader())

        perturbed = [
            Sample(sample_id=s.sample_id, patient_id=s.patient_id, site=s.site,
                   hu_loader=_wrap(s.hu_loader), labels=s.labels,
                   lesion_size_mm3=s.lesion_size_mm3)
            for s in held_out
        ]
        ds = MultiSiteICHDataset(perturbed, view_mode="fixed_three_window", seed=seed)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        probs, labels, patients = [], [], []
        for batch in loader:
            logits = forward_fn(model, batch["image"].to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(batch["labels"].numpy())
            patients.extend(batch["patient_id"])
        out[key] = {"y_true": np.concatenate(labels), "y_prob": np.concatenate(probs),
                     "patients": np.array(patients)}
        print(f"    scored scale={pert.scale:.3f} offset={pert.offset:+.1f}  n={len(out[key]['y_true'])}")
    return out


def bootstrap_resample_indices(patient_ids, rng):
    unique_patients = np.unique(patient_ids)
    sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
    return np.concatenate([np.where(patient_ids == p)[0] for p in sampled])


def corner_mean_degradation(scores, identity_key, corner_keys, idx):
    y_id, p_id = scores[identity_key]["y_true"][idx], scores[identity_key]["y_prob"][idx]
    auroc_id = safe_auroc(y_id[:, ANY_IDX], p_id[:, ANY_IDX])
    degs = []
    for key in corner_keys:
        y_c, p_c = scores[key]["y_true"][idx], scores[key]["y_prob"][idx]
        auroc_c = safe_auroc(y_c[:, ANY_IDX], p_c[:, ANY_IDX])
        degs.append(rel_deg(auroc_id, auroc_c))
    finite = [d for d in degs if not np.isnan(d)]
    return (float(np.mean(finite)) if finite else float("nan")), auroc_id


def paired_bootstrap_diff(deg_a_boot, deg_b_boot, observed_diff):
    diffs = deg_a_boot - deg_b_boot
    diffs_finite = diffs[~np.isnan(diffs)]
    if observed_diff >= 0:
        p_value = 2 * min((diffs_finite <= 0).mean(), 0.5)
    else:
        p_value = 2 * min((diffs_finite >= 0).mean(), 0.5)
    return float(min(p_value, 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--wicl-dir", type=Path, required=True)
    ap.add_argument("--pred-only-dir", type=Path, required=True)
    ap.add_argument("--embed-only-dir", type=Path, required=True)
    ap.add_argument("--rsna-root", type=Path, default=Path("data/rsna/rsna-intracranial-hemorrhage-detection"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading baseline2, full WICL, pred-only, embed-only checkpoints (read-only) ...")
    b2_model, b2_config, b2_fwd = load_baseline2(args.runs_dir, device)
    full_model, full_config, full_fwd = load_wicl(args.wicl_dir, device)
    pred_model, pred_config, pred_fwd = load_wicl(args.pred_only_dir, device)
    embed_model, embed_config, embed_fwd = load_wicl(args.embed_only_dir, device)

    print(f"full WICL config: lambda_c={full_config['lambda_consistency']} lambda_e={full_config['lambda_embedding']}")
    print(f"pred-only config: lambda_c={pred_config['lambda_consistency']} lambda_e={pred_config['lambda_embedding']}")
    print(f"embed-only config: lambda_c={embed_config['lambda_consistency']} lambda_e={embed_config['lambda_embedding']}")

    held_out, n_removed_overlap = select_holdout_samples(args.rsna_root, b2_config)

    perturbations = make_perturbation_grid()
    print(f"\nperturbation grid: {len(perturbations)} points (PROVISIONAL bounds)")

    conditions = {
        "baseline2": (b2_model, b2_fwd, b2_config),
        "wicl_full": (full_model, full_fwd, full_config),
        "wicl_pred_only": (pred_model, pred_fwd, pred_config),
        "wicl_embed_only": (embed_model, embed_fwd, embed_config),
    }

    scores = {}
    for name, (model, fwd, cfg) in conditions.items():
        print(f"\nscoring {name} across the perturbation grid ...")
        scores[name] = score_grid(model, fwd, held_out, cfg["seed"], perturbations, device)

    ref_patients = scores["baseline2"][(perturbations[0].scale, perturbations[0].offset)]["patients"]
    for name, sc in scores.items():
        for key, v in sc.items():
            assert np.array_equal(v["patients"], ref_patients), f"patient mismatch: {name} {key}"

    identity_key = (1.0, 0.0)
    corner_keys = [(s, o) for s in (0.85, 1.15) for o in (-60.0, 60.0)]

    n = len(ref_patients)
    full_idx = np.arange(n)
    observed = {}
    for name, sc in scores.items():
        deg, auroc_id = corner_mean_degradation(sc, identity_key, corner_keys, full_idx)
        observed[name] = {"identity_auroc": auroc_id, "mean_corner_degradation": deg}
        print(f"{name:<18} identity AUROC={auroc_id:.4f}  mean corner degradation={deg:+.4f}")

    print(f"\nrunning {N_BOOTSTRAP}-resample paired patient-level bootstrap "
          "(single shared resample per replicate across all 4 conditions) ...")
    rng = np.random.RandomState(SEED)
    deg_boot = {name: np.empty(N_BOOTSTRAP) for name in conditions}
    for b in range(N_BOOTSTRAP):
        idx = bootstrap_resample_indices(ref_patients, rng)
        for name, sc in scores.items():
            d, _ = corner_mean_degradation(sc, identity_key, corner_keys, idx)
            deg_boot[name][b] = d

    pairs = [
        ("baseline2", "wicl_pred_only"),
        ("baseline2", "wicl_embed_only"),
        ("baseline2", "wicl_full"),
        ("wicl_pred_only", "wicl_full"),
        ("wicl_embed_only", "wicl_full"),
        ("wicl_pred_only", "wicl_embed_only"),
    ]
    pairwise_results = {}
    print("\n=== PAIRWISE CORNER-DEGRADATION COMPARISONS (positive diff = first condition degrades more) ===")
    for a, b in pairs:
        obs_diff = observed[a]["mean_corner_degradation"] - observed[b]["mean_corner_degradation"]
        p_value = paired_bootstrap_diff(deg_boot[a], deg_boot[b], obs_diff)
        ci_a = tuple(np.nanpercentile(deg_boot[a], [2.5, 97.5]).tolist())
        ci_b = tuple(np.nanpercentile(deg_boot[b], [2.5, 97.5]).tolist())
        key = f"{a}_vs_{b}"
        pairwise_results[key] = {
            "a": a, "b": b,
            "deg_a": observed[a]["mean_corner_degradation"], "deg_a_ci95": ci_a,
            "deg_b": observed[b]["mean_corner_degradation"], "deg_b_ci95": ci_b,
            "observed_diff_a_minus_b": obs_diff, "p_value": p_value,
        }
        print(f"{a:<18} deg={observed[a]['mean_corner_degradation']:+.4f} CI{ci_a}  vs  "
              f"{b:<18} deg={observed[b]['mean_corner_degradation']:+.4f} CI{ci_b}  "
              f"diff={obs_diff:+.4f}  p={p_value:.4f}")

    results = {
        "held_out_subset": {"n_slices": len(held_out), "n_patients": len(set(ref_patients)),
                             "n_removed_for_patient_overlap": n_removed_overlap},
        "observed": observed,
        "pairwise": pairwise_results,
        "configs": {name: cfg for name, (_, _, cfg) in conditions.items()},
        "note": ("WICL component ablation on H4 (manuscript revision request): "
                 "pred-only = lambda_consistency=1.0, lambda_embedding=0.0; "
                 "embed-only = lambda_consistency=0.0, lambda_embedding=0.5; "
                 "wicl_full is the audited main-text WICL checkpoint "
                 "(lambda_consistency=1.0, lambda_embedding=0.5), read-only, "
                 "not retrained. All four conditions scored on the identical "
                 "held-out patient-disjoint RSNA subset and identical "
                 "provisional perturbation grid as the audited H4 test."),
    }
    out_path = args.runs_dir / "h4_wicl_ablation_result.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
