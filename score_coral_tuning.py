"""Re-tune baseline 4b's (dg_coral) --coral-weight on the patient-grouped
validation fold, per the manuscript's "Notes for Manuscript Completion"
flagged simplification: CORAL's raw covariance-alignment term is
numerically tiny by construction (normalized by 4*d^2), so the CLI
default of 1.0 under-weights it against the supervised BCE loss and must
be re-tuned on the validation fold rather than treated as already
calibrated (see ich_gen/train.py's --coral-weight help text and
ich_gen/models/augmentation_baselines.py's module docstring).

This script scores each already-trained candidate weight's checkpoint
(one full 12-epoch run per candidate, same hyperparameters as every
other baseline in this grid -- Section VI-B) on the SAME held-out
validation fold used throughout, and selects the weight that maximizes
pooled "any"-ICH AUROC on that validation fold (ties broken by pooled
sensitivity @ 0.5). This is a real, if small (5-point), grid search
against real held-out data -- not a synthetic placeholder and not the
untuned default.

Usage
-----
    python score_coral_tuning.py --tuning-dir runs/baselines_v1/baseline4b_coral_tuning \
        --weights 1 10 50 200 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ich_gen.datasets.common import MultiSiteICHDataset, LABEL_COLUMNS
from ich_gen.models.backbone import EmbeddingBackbone
from ich_gen.evaluate import EvalInputs, evaluate_pooled_and_stratified
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
def score_checkpoint(out_dir: Path, config: dict, val_samples: list, device: str):
    ckpt = torch.load(out_dir / "checkpoint_last.pt", map_location=device,
                       weights_only=False)
    model = EmbeddingBackbone(config["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    val_ds = MultiSiteICHDataset(val_samples, view_mode="fixed_three_window", seed=config["seed"])
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=4)

    all_probs, all_labels, all_patients, all_sizes = [], [], [], []
    for batch in val_loader:
        x = batch["image"].to(device)
        logits, _emb = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())
        all_patients.extend(batch["patient_id"])
        all_sizes.append(batch["lesion_size_mm3"].numpy())
    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    return y_true, y_prob, np.array(all_patients), np.concatenate(all_sizes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tuning-dir", type=Path, required=True)
    ap.add_argument("--weights", type=float, nargs="+", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {}
    val_samples = None
    for w in args.weights:
        wdir = args.tuning_dir / f"w{w:g}"
        with open(wdir / "config.json") as f:
            config = json.load(f)
        if val_samples is None:
            val_samples = load_val_split(config)
            print(f"held-out validation fold: {len(val_samples)} slices, "
                  f"{len({s.patient_id for s in val_samples})} patients")
        y, p, pat, sz = score_checkpoint(wdir, config, val_samples, device)
        y_any = y[:, ANY_IDX]
        p_any = p[:, ANY_IDX]
        rec = evaluate_pooled_and_stratified(
            EvalInputs(y_true=y, y_prob=p, patient_ids=pat, lesion_size_mm3=sz,
                       condition=f"dg_coral_w{w:g}", site="rsna_internal"),
            size_thresholds=None)
        # also read final-epoch training loss breakdown to report how much
        # the coral term actually moved during training at this weight
        with open(wdir / "history.json") as f:
            hist = json.load(f)
        last = hist[-1]
        summary[f"w{w:g}"] = {
            "coral_weight": w,
            "val_pooled_auroc": rec["pooled_auroc"],
            "val_pooled_sensitivity": rec["pooled_sensitivity"],
            "val_pooled_specificity": rec["pooled_specificity"],
            "val_pooled_accuracy": rec["pooled_accuracy"],
            "final_epoch_loss": last["loss"],
            "final_epoch_supervised_loss": last["supervised_loss"],
            "final_epoch_coral_loss": last["coral_loss"],
            "final_epoch_weighted_coral_contribution": w * last["coral_loss"],
        }
        print(f"w={w:g}: val AUROC={rec['pooled_auroc']:.4f}  "
              f"val sens={rec['pooled_sensitivity']:.4f}  "
              f"final supervised_loss={last['supervised_loss']:.4f}  "
              f"final coral_loss(raw)={last['coral_loss']:.6f}  "
              f"weighted_coral={w * last['coral_loss']:.4f}")

    # selection criterion: max val pooled AUROC, ties broken by val sensitivity
    best_key = max(summary, key=lambda k: (summary[k]["val_pooled_auroc"],
                                            summary[k]["val_pooled_sensitivity"]))
    print(f"\nSELECTED coral_weight = {summary[best_key]['coral_weight']:g} "
          f"(val AUROC={summary[best_key]['val_pooled_auroc']:.4f}, "
          f"val sens={summary[best_key]['val_pooled_sensitivity']:.4f})")

    out = {
        "criterion": "max val pooled 'any'-ICH AUROC on the patient-grouped "
                     "held-out fold (fold 0), ties broken by pooled sensitivity@0.5",
        "grid": summary,
        "selected_weight": summary[best_key]["coral_weight"],
        "selected_key": best_key,
    }
    out_path = args.tuning_dir / "coral_weight_tuning_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
