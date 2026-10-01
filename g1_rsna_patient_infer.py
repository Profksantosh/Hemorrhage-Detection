"""RSNA internal per-patient scoring for G1 (2026-10-01).

The original internal evaluation scored only the fold's 4,000 validation slices from the 20,000-slice
training pool (about 1.7 slices per patient), so no per-patient RSNA result exists. This script takes
the validation-fold patients of each checkpoint's own split (same build_folds call, same seed), then
scores EVERY RSNA slice of those patients from the full BHSD-excluded pool. None of these patients'
slices were used for training (training used only train-fold patients of the 20,000-slice pool).
Models sharing a seed share one data pass.

    python g1_rsna_patient_infer.py --out-dir runs/g1_rsna_patient \
        --model B2=runs/baselines_bhsd_excluded/baseline2_fixed_three:plain \
        --model B4c=runs/baselines_bhsd_excluded/baseline4c_random_window_single:plain \
        --model WICL=runs/baselines_bhsd_excluded/decisive_wicl:wicl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ich_gen import train as train_mod
from ich_gen.datasets.common import MultiSiteICHDataset
from ich_gen.datasets.rsna import build_samples, load_bhsd_overlap_exclusion_ids
from ich_gen.train import DEFAULT_BHSD_EXCLUSION_JSON
from g1_h4_models import load_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name=run_dir:kind (kind: plain|wicl)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for spec in args.model:
        name, rest = spec.split("=", 1)
        run_dir, kind = rest.rsplit(":", 1)
        if (args.out_dir / f"{name}.npz").exists():
            print(f"skip {name} (exists)", flush=True)
            continue
        model, cfg, fwd = load_model(Path(run_dir), kind, device)
        groups[(cfg["seed"], cfg["fold"], cfg["n_folds"], cfg["max_samples"], cfg["rsna_root"])].append(
            (name, model, fwd))
    if not groups:
        return

    rsna_root = Path(next(iter(groups))[4])
    full = build_samples(rsna_root, max_samples=None,
                         exclude_patient_ids=load_bhsd_overlap_exclusion_ids(DEFAULT_BHSD_EXCLUSION_JSON))
    print(f"full BHSD-excluded RSNA pool: {len(full)} slices", flush=True)

    for (seed, fold, n_folds, max_samples, root), members in groups.items():
        ns = argparse.Namespace(rsna_root=Path(root), max_samples=max_samples, n_folds=n_folds,
                                seed=seed, bhsd_exclusion=True)
        pool, folds = train_mod.build_folds(ns)
        f = folds[fold]
        val_patients, train_patients = f.val_patients, f.train_patients
        assert val_patients.isdisjoint(train_patients)
        val = [s for s in full if s.patient_id in val_patients]
        n_pool_val = len(f.val_idx)
        print(f"seed {seed} fold {fold}: {len(val_patients)} val patients, {n_pool_val} val slices in "
              f"training pool -> {len(val)} slices in full pool", flush=True)

        ds = MultiSiteICHDataset(val, view_mode="fixed_three_window", seed=seed)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        probs = {name: [] for name, _, _ in members}
        labels, pids = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                x = batch["image"].to(device)
                for name, model, fwd in members:
                    probs[name].append(torch.sigmoid(fwd(model, x)).cpu().numpy())
                labels.append(batch["labels"].numpy())
                pids.extend(batch["patient_id"])
                if i % 100 == 0:
                    print(f"  batch {i}/{len(loader)}", flush=True)
        y, pid = np.concatenate(labels), np.array(pids)
        for name, _, _ in members:
            np.savez_compressed(args.out_dir / f"{name}.npz", prob=np.concatenate(probs[name]), y=y, pid=pid,
                                seed=seed, fold=fold)
            print(f"saved {name}.npz ({len(y)} slices, {len(set(pids))} patients)", flush=True)
        (args.out_dir / f"split_seed{seed}_fold{fold}.json").write_text(json.dumps(
            {"n_val_patients": len(val_patients), "n_val_slices_full_pool": len(val),
             "n_val_slices_training_pool": n_pool_val, "models": [m[0] for m in members]}, indent=1))


if __name__ == "__main__":
    main()
