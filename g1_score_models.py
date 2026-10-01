"""Score arbitrary checkpoints on CQ500 and PhysioNet-ICH (radiological orientation) with study_level_rescore.infer
unchanged otherwise (G1 revision, 2026-10-01).

    python g1_score_models.py --out-dir runs/g1_ext_more \
        --model CNX_B2_seed0=runs/g1_convnext/b2_seed0:plain --model CNX_WICL_seed0=runs/g1_convnext/wicl_seed0:wicl
"""
import argparse
from pathlib import Path

import ich_gen.datasets.physionet_ich as phys
import study_level_rescore as slr
from g1_physionet_reoriented_infer import _reoriented_reader

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="name=run_dir:kind (kind: plain|wicl)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    ap.add_argument("--physionet-root", type=Path, default=Path("data/physionet_ich"))
    args = ap.parse_args()
    args.runs_dir = Path(".")
    phys._read_volume_slice = _reoriented_reader
    slr.MODELS = {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        run_dir, kind = rest.rsplit(":", 1)
        slr.MODELS[name] = (run_dir, "fixed_three_window", kind)
    slr.infer(args)
