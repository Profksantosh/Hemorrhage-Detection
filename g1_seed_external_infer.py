"""Score G1 seed-replicate checkpoints on CQ500 and PhysioNet-ICH with study_level_rescore.infer
unchanged (same loaders, view modes, output format); only the model list is replaced."""
import argparse
from pathlib import Path

import study_level_rescore as slr

KINDS = {"B2": ("fixed_three_window", "plain"), "B4c": ("fixed_three_window", "plain"),
         "WICL": ("fixed_three_window", "wicl")}
RUN_DIRS = {"B2": "b2", "B4c": "b4c", "WICL": "wicl"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/g1_seeds"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/g1_seeds_ext"))
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    ap.add_argument("--physionet-root", type=Path, default=Path("data/physionet_ich"))
    args = ap.parse_args()
    slr.MODELS = {f"{m}_seed{s}": (f"{RUN_DIRS[m]}_seed{s}", *KINDS[m]) for s in args.seeds for m in KINDS}
    slr.infer(args)
