"""CQ500 slice-thickness harmonization test (G1 revision, 2026-10-01). Inference only.

Training data (RSNA) and most CQ500 series are ~5 mm; 80 CQ500 studies only have thinner series (0.625-3.75 mm). For
every study whose selected series is thinner than 4.5 mm, consecutive slices (ordered by ImagePositionPatient z) are
averaged in HU into slabs of k = round(5 / thickness) slices, approximating a 5 mm reconstruction; 5 mm studies are
left unchanged. All listed models are scored on the harmonized CQ500 in one data pass; output has the same format as
study_level_rescore (prob, y, pid per slab) plus the per-study thickness.

    python g1_thickness_harmonize_infer.py --out-dir runs/g1_cq500_harmonized \
        --model B2_seed0=runs/baselines_bhsd_excluded/baseline2_fixed_three:plain ...
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
import torch
from torch.utils.data import DataLoader

from g1_h4_models import load_model
from ich_gen.datasets.common import MultiSiteICHDataset, Sample
from ich_gen.datasets.cq500 import build_samples
from ich_gen.datasets.rsna import read_hu


def slab_loader(paths):
    def load():
        arrs = [read_hu(p) for p in paths]
        shape = arrs[0].shape
        arrs = [a for a in arrs if a.shape == shape]
        return np.mean(arrs, axis=0).astype(np.float32)
    return load


def harmonized_samples(root: Path):
    thickness = {r["study_id"]: float(r["thickness"]) for r in csv.DictReader(open(root / "cq500_series_manifest.csv"))}
    by_study = defaultdict(list)
    for s in build_samples(root):
        by_study[s.patient_id].append(s)
    rows = list(csv.DictReader(open(root / "cq500_labels_aggregated.csv")))
    path_of = {f"CQ500_{r['study_id']}_{i}": root / r["slice_path"] for i, r in enumerate(rows)}
    out, n_thin, info = [], 0, {}
    for pid, ss in by_study.items():
        study_id = pid[len("CQ500_"):]
        t = thickness[study_id]
        info[pid] = t
        if t >= 4.5:
            out.extend(ss)
            continue
        n_thin += 1
        z = [float(pydicom.dcmread(str(path_of[s.sample_id]), stop_before_pixels=True).ImagePositionPatient[2])
             for s in ss]
        order = np.argsort(z)
        k = max(1, int(round(5.0 / t)))
        for j in range(0, len(order), k):
            grp = [ss[i] for i in order[j:j + k]]
            out.append(Sample(sample_id=f"{grp[0].sample_id}_slab{j // k}", patient_id=pid, site="cq500",
                              hu_loader=slab_loader([path_of[g.sample_id] for g in grp]),
                              labels=grp[0].labels, lesion_size_mm3=None))
    print(f"harmonized {n_thin} thin-series studies; {len(out)} slices/slabs in total", flush=True)
    return out, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="name=run_dir:kind")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for spec in args.model:
        name, rest = spec.split("=", 1)
        if (args.out_dir / f"{name}__cq500h.npz").exists():
            continue
        run_dir, kind = rest.rsplit(":", 1)
        m, _, fwd = load_model(Path(run_dir), kind, device)
        models.append((name, m, fwd))
    if not models:
        return
    samples, info = harmonized_samples(args.cq500_root)
    loader = DataLoader(MultiSiteICHDataset(samples, view_mode="fixed_three_window", seed=0),
                        batch_size=64, shuffle=False, num_workers=8)
    probs = {n: [] for n, _, _ in models}
    labels, pids = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            for n, m, fwd in models:
                probs[n].append(torch.sigmoid(fwd(m, x)).cpu().numpy())
            labels.append(batch["labels"].numpy())
            pids.extend(batch["patient_id"])
    y, pid = np.concatenate(labels), np.array(pids)
    th = np.array([info[p] for p in pid])
    for n, _, _ in models:
        np.savez_compressed(args.out_dir / f"{n}__cq500h.npz", prob=np.concatenate(probs[n]), y=y, pid=pid,
                            thickness=th)
        print(f"saved {n}__cq500h.npz", flush=True)


if __name__ == "__main__":
    main()
