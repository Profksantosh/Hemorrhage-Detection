"""THROWAWAY (but evidence-preserving) script for gap 1: reproduce the
EXACT same deterministic subsample and phash computation that
leakage_check_rsna_cq500.py used (same seed=0, same n_rsna_sample=400,
n_cq500_sample=2000, same train_mod.build_folds/build_cq500 samples), so we
can recover the FULL list of 302 near-duplicate candidate pairs (the
original run only persisted the top 20 to
runs/decisive_v2/leakage_check_rsna_cq500_result.json).

Writes the full hit list to runs/decisive_v2/leakage_check_full_hits.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ich_gen import train as train_mod
from ich_gen.datasets.cq500 import build_samples as build_cq500
from ich_gen.datasets.splits import compute_phash, find_near_duplicates
from ich_gen.windowing import CANONICAL_WINDOWS, window_hu


def to_uint8_brain_window(hu: np.ndarray) -> np.ndarray:
    level, width = CANONICAL_WINDOWS["brain"]
    windowed = window_hu(hu, level, width)
    return np.clip(windowed, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/decisive_v2"))
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    ap.add_argument("--n-rsna-sample", type=int, default=400)
    ap.add_argument("--n-cq500-sample", type=int, default=2000)
    ap.add_argument("--max-hamming", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.runs_dir / "decisive_wicl" / "config.json") as f:
        config = json.load(f)

    ns = argparse.Namespace(rsna_root=Path(config["rsna_root"]),
                             max_samples=config["max_samples"],
                             n_folds=config["n_folds"], seed=config["seed"])
    samples, folds = train_mod.build_folds(ns)
    fold = folds[config["fold"]]
    train_samples = [samples[i] for i in fold.train_idx]

    cq_samples = build_cq500(args.cq500_root)

    rng = random.Random(args.seed)
    rsna_sub = rng.sample(train_samples, min(args.n_rsna_sample, len(train_samples)))
    cq_sub = rng.sample(cq_samples, min(args.n_cq500_sample, len(cq_samples)))
    print(f"reproducing subsample: {len(rsna_sub)} RSNA, {len(cq_sub)} CQ500 "
          f"(seed={args.seed}) -- must match original run's n_hits=302 for "
          f"this to be a valid reproduction")

    # keep sample objects (not just hashes) so we can go straight from hit
    # -> Sample (hu_loader, file path) without re-scanning either dataset
    rsna_by_id = {s.sample_id: s for s in rsna_sub}
    cq_by_id = {s.sample_id: s for s in cq_sub}

    def hash_all(samples_list, tag):
        hashes = {}
        for i, s in enumerate(samples_list):
            hu = s.hu_loader()
            hashes[s.sample_id] = compute_phash(to_uint8_brain_window(hu))
            if (i + 1) % max(1, len(samples_list) // 8) == 0:
                print(f"  ...{tag}: hashed {i + 1}/{len(samples_list)}")
        return hashes

    hashes_rsna = hash_all(rsna_sub, "RSNA")
    hashes_cq = hash_all(cq_sub, "CQ500")

    hits = find_near_duplicates(hashes_rsna, hashes_cq,
                                 max_hamming_distance=args.max_hamming)
    hits_sorted = sorted(hits, key=lambda x: x[2])
    print(f"\nreproduced n_hits = {len(hits)} (original run reported 302)")

    by_hamming = {}
    for a, b, d in hits_sorted:
        by_hamming.setdefault(int(d), []).append({"rsna_id": a, "cq500_id": b})
    for d in sorted(by_hamming):
        print(f"  hamming={d}: {len(by_hamming[d])} pair(s)")

    out = {
        "n_hits": len(hits),
        "seed": args.seed,
        "max_hamming_distance": args.max_hamming,
        "by_hamming_distance": by_hamming,
        "all_hits": [{"rsna_id": a, "cq500_id": b, "hamming": int(d)}
                      for a, b, d in hits_sorted],
    }
    # Path reconstruction (NOT via closure introspection -- `p` in
    # `lambda p=path: read_hu(p)` is a bound default arg, not a free
    # variable, so hu_loader.__closure__ is None; use the known, documented
    # naming conventions from rsna.build_samples / cq500.build_samples
    # instead):
    #   RSNA:  sample_id IS the sop_uid -> <rsna_root>/stage_2_train/<sop_uid>.dcm
    #   CQ500: sample_id = f"CQ500_{study_id}_{i}" where i is the row's
    #          position in cq500_labels_aggregated.csv -> root/slice_path[i]
    rsna_dicom_dir = Path(config["rsna_root"]) / "stage_2_train"
    out["rsna_paths"] = {sid: str(rsna_dicom_dir / f"{sid}.dcm")
                          for sid in rsna_by_id}
    import pandas as pd
    cq_df = pd.read_csv(args.cq500_root / "cq500_labels_aggregated.csv")
    out["cq500_paths"] = {}
    for sid in cq_by_id:
        i = int(sid.rsplit("_", 1)[1])
        out["cq500_paths"][sid] = str(args.cq500_root / cq_df.iloc[i]["slice_path"])

    out_path = args.runs_dir / "leakage_check_full_hits.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
