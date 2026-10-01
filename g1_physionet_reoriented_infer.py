"""Re-score PhysioNet-ICH with slices in the same display orientation as RSNA/CQ500 (G1 revision, 2026-10-01).

All 75 PhysioNet-ICH NIfTI volumes are stored LAS (nib.aff2axcodes), so data[..., k] has patient-left along rows and
anterior along columns: the face points right, a 90-degree rotation relative to the DICOM-derived training images.
np.rot90(slice, 1) gives the radiological orientation (anterior up, patient right on image left). The shared loader
is not edited; its slice reader is wrapped here, and study_level_rescore.infer is reused unchanged otherwise.

    python g1_physionet_reoriented_infer.py --out-dir runs/g1_physionet_fixed --set seed0
    python g1_physionet_reoriented_infer.py --out-dir runs/g1_physionet_fixed --set seeds12
"""
import argparse
from pathlib import Path

import numpy as np

import ich_gen.datasets.physionet_ich as phys
import study_level_rescore as slr

_original_reader = phys._read_volume_slice


def _reoriented_reader(nii_path, slice_index_0based):
    return np.ascontiguousarray(np.rot90(_original_reader(nii_path, slice_index_0based), 1))


SEED0 = {m: slr.MODELS[m] for m in ["B1", "B2", "B4b", "B4c", "B5", "WICL"]}
SEEDS12 = {f"{m}_seed{s}": (f"../g1_seeds/{d}_seed{s}", "fixed_three_window", kind)
           for s in (1, 2) for m, d, kind in [("B2", "b2", "plain"), ("B4c", "b4c", "plain"), ("WICL", "wicl", "wicl")]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["seed0", "seeds12"], required=True)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/baselines_bhsd_excluded"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/g1_physionet_fixed"))
    ap.add_argument("--cq500-root", type=Path, default=Path("data/cq500"))
    ap.add_argument("--physionet-root", type=Path, default=Path("data/physionet_ich"))
    args = ap.parse_args()
    phys._read_volume_slice = _reoriented_reader
    slr.SITES = ("physionet",)
    slr.MODELS = SEED0 if args.set == "seed0" else SEEDS12
    slr.infer(args)
