"""Real-data cross-dataset leakage check: RSNA training pool vs. CQ500.

Closes gap 1 from the repro-audit: verify_disjoint_from_external and
find_near_duplicates (ich_gen/datasets/splits.py) were previously only
exercised in tests/test_splits_no_leakage.py against synthetic
(hand-constructed) IDs. This script runs them against the REAL RSNA
patient-ID pool used to train the decisive_v2 checkpoints (same
rsna_root/seed/max_samples/n_folds/fold as runs/decisive_v2/*/config.json,
via ich_gen.train.build_folds) and the REAL CQ500 studies used in
runs/decisive_v2/external_cq500_result.json's evaluation (via
ich_gen.datasets.cq500.build_samples on the CQ500 root's
cq500_labels_aggregated.csv, the same file score_external.py reads).

Two checks:
  1. Exact patient/study-ID string overlap (verify_disjoint_from_external).
     Fast, exhaustive -- covers ALL patients/studies, not a sample.
  2. Perceptual-hash (phash) near-duplicate image check
     (find_near_duplicates), which is the check that would catch shared
     provenance despite disjoint ID namespaces (e.g. the same underlying
     Kaggle-pool slice re-issued with different site-format IDs). This is
     O(|A|*|B|) pairwise, which is prohibitive at full scale on CPU-only
     hardware for a 16k+ (RSNA train) x 22k+ (CQ500) grid, so this check
     is run over a documented random subsample of BOTH sides (seeded, so
     reproducible) rather than skipped -- consistent with the
     "sub-sample or shard" TODO already written into
     find_near_duplicates's own docstring.

Usage
-----
    python leakage_check_rsna_cq500.py --runs-dir runs/decisive_v2 \
        --cq500-root data/cq500 2>&1 | tee leakage_check_rsna_cq500.log
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from ich_gen import train as train_mod
from ich_gen.datasets.cq500 import build_samples as build_cq500
from ich_gen.datasets.splits import (
    compute_phash,
    find_near_duplicates,
    verify_disjoint_from_external,
)
from ich_gen.windowing import CANONICAL_WINDOWS, window_hu


def to_uint8_brain_window(hu: np.ndarray) -> np.ndarray:
    """Brain-window (level=40, width=80) HU->uint8 conversion, used only
    to produce a perceptually-comparable image for phash hashing -- NOT
    the model's actual multi-window input pipeline (see
    ich_gen/windowing.py three_window_composite for that)."""
    level, width = CANONICAL_WINDOWS["brain"]
    windowed = window_hu(hu, level, width)  # float32, range [0, 255]
    return np.clip(windowed, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/decisive_v2"))
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    ap.add_argument("--n-rsna-sample", type=int, default=400,
                     help="random subsample size of RSNA training slices "
                          "for the phash near-duplicate check")
    ap.add_argument("--n-cq500-sample", type=int, default=2000,
                     help="random subsample size of CQ500 slices for the "
                          "phash near-duplicate check")
    ap.add_argument("--max-hamming", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    with open(args.runs_dir / "decisive_wicl" / "config.json") as f:
        config = json.load(f)
    print(f"using training config from {args.runs_dir / 'decisive_wicl' / 'config.json'}: "
          f"rsna_root={config['rsna_root']} max_samples={config['max_samples']} "
          f"n_folds={config['n_folds']} seed={config['seed']} fold={config['fold']}")

    ns = argparse.Namespace(rsna_root=Path(config["rsna_root"]),
                             max_samples=config["max_samples"],
                             n_folds=config["n_folds"], seed=config["seed"])
    samples, folds = train_mod.build_folds(ns)
    fold = folds[config["fold"]]
    train_samples = [samples[i] for i in fold.train_idx]
    train_patient_ids = sorted({s.patient_id for s in train_samples})
    print(f"RSNA training pool actually used for the decisive_v2 checkpoints "
          f"(fold {config['fold']} train split): {len(train_samples)} slices, "
          f"{len(train_patient_ids)} unique patients")

    print("building CQ500 sample list (same cq500_labels_aggregated.csv "
          "score_external.py read for external_cq500_result.json)...")
    cq_samples = build_cq500(args.cq500_root)
    cq_patient_ids = sorted({s.patient_id for s in cq_samples})
    print(f"CQ500: {len(cq_samples)} slices, {len(cq_patient_ids)} unique "
          f"patients/studies")

    print("\n=== Check 1: exact patient/study-ID overlap "
          "(verify_disjoint_from_external, EXHAUSTIVE over all IDs) ===")
    try:
        verify_disjoint_from_external(train_patient_ids, cq_patient_ids, "CQ500")
        check1_result = "PASS: no patient/study ID string shared between the " \
                         "RSNA training pool and CQ500."
    except AssertionError as e:
        check1_result = f"FAIL: {e}"
    print(check1_result)

    print("\n=== Check 2: perceptual-hash near-duplicate check "
          "(find_near_duplicates, SUBSAMPLED -- see module docstring) ===")
    rng = random.Random(args.seed)
    rsna_sub = rng.sample(train_samples, min(args.n_rsna_sample, len(train_samples)))
    cq_sub = rng.sample(cq_samples, min(args.n_cq500_sample, len(cq_samples)))
    print(f"hashing {len(rsna_sub)}/{len(train_samples)} RSNA training slices "
          f"and {len(cq_sub)}/{len(cq_samples)} CQ500 slices (seed={args.seed}); "
          f"full all-pairs ({len(train_samples)}x{len(cq_samples)} = "
          f"{len(train_samples) * len(cq_samples):,} pairs) is infeasible on "
          f"CPU-only hardware in this session -- this is a documented "
          f"subsample, not an exhaustive check.")

    def hash_all(samples_list, tag):
        hashes = {}
        n_err = 0
        for i, s in enumerate(samples_list):
            try:
                hu = s.hu_loader()
                hashes[s.sample_id] = compute_phash(to_uint8_brain_window(hu))
            except Exception as e:
                n_err += 1
                if n_err <= 5:
                    print(f"  [skip] {tag} {s.sample_id}: {type(e).__name__}: {e}")
            if (i + 1) % max(1, len(samples_list) // 8) == 0:
                print(f"  ...{tag}: hashed {i + 1}/{len(samples_list)}")
        if n_err:
            print(f"  {tag}: {n_err}/{len(samples_list)} slices failed to load/hash")
        return hashes

    hashes_rsna = hash_all(rsna_sub, "RSNA")
    hashes_cq = hash_all(cq_sub, "CQ500")

    hits = find_near_duplicates(hashes_rsna, hashes_cq,
                                 max_hamming_distance=args.max_hamming)
    hits_sorted = sorted(hits, key=lambda x: x[2])
    print(f"\nnear-duplicate candidates (Hamming distance <= {args.max_hamming}) "
          f"among {len(hashes_rsna)} x {len(hashes_cq)} = "
          f"{len(hashes_rsna) * len(hashes_cq):,} compared pairs: {len(hits)}")
    for a, b, d in hits_sorted[:20]:
        print(f"  {a}  ~  {b}   (hamming={d})")
    if len(hits) > 20:
        print(f"  ... and {len(hits) - 20} more")

    check2_result = ("PASS: no near-duplicate candidates found in the "
                      "subsample." if len(hits) == 0 else
                      f"FLAGGED: {len(hits)} near-duplicate candidate pair(s) "
                      f"found in the subsample -- inspect before trusting the "
                      f"CQ500 zero-shot result as leakage-free.")
    print(f"\n{check2_result}")

    result = {
        "rsna_training_pool": {
            "rsna_root": config["rsna_root"], "fold": config["fold"],
            "n_folds": config["n_folds"], "seed": config["seed"],
            "max_samples": config["max_samples"],
            "n_train_slices": len(train_samples),
            "n_train_patients": len(train_patient_ids),
        },
        "cq500": {
            "n_slices": len(cq_samples), "n_studies": len(cq_patient_ids),
        },
        "check1_exact_id_overlap": {
            "method": "verify_disjoint_from_external (exhaustive over all "
                      "patient/study IDs)",
            "result": check1_result,
        },
        "check2_phash_near_duplicates": {
            "method": "find_near_duplicates (imagehash.phash, brain window "
                      "40/80, Hamming distance)",
            "max_hamming_distance": args.max_hamming,
            "n_rsna_hashed": len(hashes_rsna),
            "n_cq500_hashed": len(hashes_cq),
            "n_compared_pairs": len(hashes_rsna) * len(hashes_cq),
            "subsampled": True,
            "seed": args.seed,
            "n_hits": len(hits),
            "top_hits": [{"rsna_id": a, "cq500_id": b, "hamming": int(d)}
                         for a, b, d in hits_sorted[:20]],
            "result": check2_result,
        },
    }
    out_path = args.out or (args.runs_dir / "leakage_check_rsna_cq500_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
