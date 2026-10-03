"""Read-only: patient/slice counts of the 20,000-slice training pool and its fold-0 split (same call path as ich_gen/train.py)."""
import argparse, json
from pathlib import Path
import numpy as np
from ich_gen import train as T
cfg = json.load(open("runs/baselines_bhsd_excluded/baseline2_fixed_three/config.json"))
args = argparse.Namespace(**{k: (Path(v) if k in ("rsna_root", "out_dir", "bhsd_exclusion_json") and v else v) for k, v in cfg.items()})
samples, folds = T.build_folds(args)
f = folds[args.fold]
lab = np.stack([s.labels for s in samples])
pid = np.array([s.patient_id for s in samples])
def rep(name, idx):
    ps = set(pid[idx].tolist())
    print(f"{name}: slices={len(idx)} patients={len(ps)} any-ICH slice prevalence={lab[idx,0].mean():.4f} "
          f"patients_with_any_positive_slice={len({p for p,l in zip(pid[idx], lab[idx,0]) if l>0})}")
rep("pool(max_samples=%s)" % args.max_samples, np.arange(len(samples)))
rep("fold0 train", f.train_idx); rep("fold0 val", f.val_idx)
print("train/val patient overlap:", len(f.train_patients & f.val_patients))
print("first/last sample ids in pool:", samples[0].sample_id, samples[-1].sample_id)
print("slice positives per label (pool):", dict(zip(["any","epidural","intraparenchymal","intraventricular","subarachnoid","subdural"], lab.sum(0).astype(int).tolist())))
