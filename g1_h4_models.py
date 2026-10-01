"""H4 HU-corner stress test for an arbitrary set of checkpoints (G1 revision, 2026-10-01).

Reuses h4_wicl_ablation.py's audited held-out subset, perturbation grid, corner-degradation
definition and paired patient-level bootstrap unchanged; only the set of models is configurable.
Main use: add baseline 4c (single-view random window), which the original H4 test never scored.

    python g1_h4_models.py --out runs/baselines_bhsd_excluded/h4_4c_result.json \
        --model baseline2=runs/baselines_bhsd_excluded/baseline2_fixed_three:plain \
        --model b4c=runs/baselines_bhsd_excluded/baseline4c_random_window_single:plain \
        --model wicl=runs/baselines_bhsd_excluded/decisive_wicl:wicl
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from h4_wicl_ablation import (N_BOOTSTRAP, SEED, bootstrap_resample_indices, corner_mean_degradation,
                              paired_bootstrap_diff, score_grid, select_holdout_samples)
from ich_gen.models.backbone import build_backbone
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stress_test import make_perturbation_grid


def load_model(run_dir: Path, kind: str, device: str):
    config = json.loads((run_dir / "config.json").read_text())
    assert config.get("bhsd_exclusion") is True, f"{run_dir}: not leakage-clean"
    ckpt = torch.load(run_dir / "checkpoint_last.pt", map_location=device, weights_only=False)
    if kind == "wicl":
        model = build_wicl_model(config["backbone"], pretrained=False,
                                 lambda_consistency=config["lambda_consistency"],
                                 lambda_embedding=config["lambda_embedding"])
        fwd = lambda m, x: m.forward_single(x)
    else:
        model = build_backbone(config["backbone"], pretrained=False)
        fwd = lambda m, x: m(x)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), config, fwd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name=run_dir:kind (kind: plain|wicl)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rsna-root", type=Path, default=Path("data/rsna/rsna-intracranial-hemorrhage-detection"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        run_dir, kind = rest.rsplit(":", 1)
        models[name] = load_model(Path(run_dir), kind, device)
        print(f"loaded {name} from {run_dir} ({kind}), seed={models[name][1]['seed']}", flush=True)

    first_cfg = next(iter(models.values()))[1]
    held_out, n_removed = select_holdout_samples(args.rsna_root, first_cfg)
    perturbations = make_perturbation_grid()

    scores = {}
    for name, (model, cfg, fwd) in models.items():
        print(f"\nscoring {name} across the perturbation grid ...", flush=True)
        scores[name] = score_grid(model, fwd, held_out, cfg["seed"], perturbations, device)

    first = next(iter(scores))
    ref_patients = scores[first][(perturbations[0].scale, perturbations[0].offset)]["patients"]
    for name, sc in scores.items():
        for key, v in sc.items():
            assert np.array_equal(v["patients"], ref_patients), f"patient mismatch: {name} {key}"

    identity_key = (1.0, 0.0)
    corner_keys = [(s, o) for s in (0.85, 1.15) for o in (-60.0, 60.0)]
    full_idx = np.arange(len(ref_patients))
    observed = {}
    for name, sc in scores.items():
        deg, auroc_id = corner_mean_degradation(sc, identity_key, corner_keys, full_idx)
        observed[name] = {"identity_auroc": auroc_id, "mean_corner_degradation": deg}
        print(f"{name:<12} identity AUROC={auroc_id:.4f}  mean corner degradation={deg:+.4f}", flush=True)

    rng = np.random.RandomState(SEED)
    deg_boot = {name: np.empty(N_BOOTSTRAP) for name in scores}
    for b in range(N_BOOTSTRAP):
        idx = bootstrap_resample_indices(ref_patients, rng)
        for name, sc in scores.items():
            deg_boot[name][b] = corner_mean_degradation(sc, identity_key, corner_keys, idx)[0]
    for name in scores:
        observed[name]["ci95"] = np.nanpercentile(deg_boot[name], [2.5, 97.5]).tolist()

    pairwise = {}
    for a, b in itertools.combinations(scores, 2):
        diff = observed[a]["mean_corner_degradation"] - observed[b]["mean_corner_degradation"]
        p = paired_bootstrap_diff(deg_boot[a], deg_boot[b], diff)
        pairwise[f"{a}_vs_{b}"] = {"diff_a_minus_b": diff, "p_value": p}
        print(f"{a} vs {b}: diff={diff:+.4f} p={p:.4f}", flush=True)

    result ={"held_out_subset": {"n_slices": len(held_out), "n_patients": len(set(ref_patients.tolist())),
                                  "n_removed_for_patient_overlap": n_removed},
              "observed": observed, "pairwise": pairwise,
              "models": {n: {"run_dir": s.split("=", 1)[1]} for n, s in zip(models, args.model)},
              "method_note": "held-out subset, grid, corner definition and bootstrap reused unchanged "
                             "from h4_wicl_ablation.py"}
    args.out.write_text(json.dumps(result, indent=2, default=str))
    np.savez_compressed(args.out.with_suffix(".npz"), patients=ref_patients,
                        y=scores[first][identity_key]["y_true"],
                        **{f"{n}__{k[0]}_{k[1]}": v["y_prob"] for n, sc in scores.items() for k, v in sc.items()})
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
