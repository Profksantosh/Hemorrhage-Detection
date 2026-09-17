"""Follow-up scoring for the decisive WICL-vs-baseline-4c check.

run_pipeline.py's cmd_decisive_check trains both checkpoints and
deliberately leaves scoring as an explicit follow-up call (it does not
hardcode which dataset paths are configured). This script performs that
follow-up: rebuild the same patient-grouped fold/held-out split used at
training time (same rsna-root/seed/max-samples/n-folds), score both
checkpoints on it with evaluate.py's shared scoring path, and feed the
pooled "any"-ICH probabilities to stats.delong_paired_auc_test, per the
Research Decision Report's pre-committed interpretation rule.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ich_gen.datasets.common import MultiSiteICHDataset, LABEL_COLUMNS
from ich_gen.models.backbone import build_backbone
from ich_gen.models.wicl import build_wicl_model
from ich_gen.stats import delong_paired_auc_test, MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
from ich_gen import train as train_mod

ANY_IDX = LABEL_COLUMNS.index("any")


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
def score_checkpoint(condition: str, out_dir: Path, config: dict,
                      val_samples: list, device: str):
    # weights_only=False: this is our own checkpoint from a training run we
    # just launched on this same trusted machine (not an untrusted download).
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)

    if condition == "wicl":
        model = build_wicl_model(config["backbone"], pretrained=False,
                                  lambda_consistency=config["lambda_consistency"],
                                  lambda_embedding=config["lambda_embedding"])
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        forward = model.forward_single
    else:
        model = build_backbone(config["backbone"], pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()
        forward = model

    val_ds = MultiSiteICHDataset(val_samples, view_mode="fixed_three_window",
                                  seed=config["seed"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                             shuffle=False, num_workers=4)

    all_probs, all_labels, all_patients = [], [], []
    for batch in val_loader:
        x = batch["image"].to(device)
        logits = forward(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())
        all_patients.extend(batch["patient_id"])

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_prob, np.array(all_patients)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/decisive"),
                     help="directory holding decisive_wicl/ and "
                          "decisive_random_window_single/")
    cli = ap.parse_args()
    RUNS_DIR = cli.runs_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(RUNS_DIR / "decisive_wicl" / "config.json") as f:
        config = json.load(f)

    val_samples = load_val_split(config)
    print(f"held-out validation fold: {len(val_samples)} slices, "
          f"{len({s.patient_id for s in val_samples})} patients")

    y_true_w, y_prob_w, patients_w = score_checkpoint(
        "wicl", RUNS_DIR / "decisive_wicl", config, val_samples, device)
    y_true_b, y_prob_b, patients_b = score_checkpoint(
        "random_window_single", RUNS_DIR / "decisive_random_window_single",
        config, val_samples, device)

    assert np.array_equal(patients_w, patients_b), "val ordering mismatch between runs"
    assert np.allclose(y_true_w, y_true_b), "label mismatch between runs"

    y_true_any = y_true_w[:, ANY_IDX]
    y_prob_wicl_any = y_prob_w[:, ANY_IDX]
    y_prob_baseline_any = y_prob_b[:, ANY_IDX]

    result = delong_paired_auc_test(y_true_any, y_prob_wicl_any, y_prob_baseline_any)

    print("\n=== DECISIVE CHECK RESULT (pooled 'any' ICH, RSNA held-out fold 0) ===")
    print(f"WICL AUROC:                {result.auc_a:.4f}")
    print(f"baseline 4c AUROC:         {result.auc_b:.4f}")
    print(f"AUROC diff (WICL - 4c):    {result.auc_diff:+.4f}")
    print(f"DeLong z:                  {result.z_statistic:.4f}")
    print(f"DeLong p-value:            {result.p_value:.4f}")

    threshold = 0.5
    sens_w = ((y_prob_wicl_any >= threshold) & (y_true_any == 1)).sum() / max(y_true_any.sum(), 1)
    sens_b = ((y_prob_baseline_any >= threshold) & (y_true_any == 1)).sum() / max(y_true_any.sum(), 1)
    sens_delta = float(sens_w - sens_b)
    print(f"sensitivity delta (WICL - 4c) @0.5: {sens_delta:+.4f} "
          f"(clinically-relevant threshold: {MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA})")

    significant = result.p_value < 0.05
    clinically_relevant = abs(sens_delta) >= MIN_CLINICALLY_RELEVANT_SENSITIVITY_DELTA
    print("\n--- interpretation (Research Decision Report Section 9 rule) ---")
    if significant and clinically_relevant:
        print("WICL beats baseline 4c: statistically significant AND clinically "
              "relevant. Proceed to full grid treating WICL as the positive result.")
    elif not significant:
        print("NOT statistically significant (DeLong p >= 0.05) on this fast "
              "5-epoch/5000-sample subset -- per the pre-committed rule, this is "
              "a signal to revise the manuscript's framing before investing in the "
              "full grid, UNLESS you attribute this to the deliberately short/fast "
              "decisive-check schedule (5 epochs, 5000-sample cap) rather than a "
              "real null result -- consider re-running with more epochs/samples "
              "before treating this as final.")
    else:
        print("Statistically significant but below the clinically-relevant "
              "sensitivity-delta threshold -- consider framing as a narrowed/"
              "hedged contribution rather than a clean positive result.")

    with open(RUNS_DIR / "decisive_check_result.json", "w") as f:
        json.dump({
            "auc_wicl": result.auc_a, "auc_baseline4c": result.auc_b,
            "auc_diff": result.auc_diff, "delong_z": result.z_statistic,
            "delong_p": result.p_value, "sensitivity_delta": sens_delta,
            "n_val_slices": len(y_true_any), "n_val_patients": int(len(set(patients_w.tolist()))),
        }, f, indent=2)
    print(f"\nresult saved to {RUNS_DIR / 'decisive_check_result.json'}")


if __name__ == "__main__":
    main()
