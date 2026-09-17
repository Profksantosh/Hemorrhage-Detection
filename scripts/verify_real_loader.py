"""Verification that the FIXED, real ich_gen.datasets.physionet_ich.build_samples()
loads successfully against real PhysioNet-ICH data (post zero-padding fix).
This calls the actual public loader entry point directly -- not the standalone
bypass smoke-test script.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "ich_windowing"))

from ich_gen.datasets.physionet_ich import build_samples
from ich_gen.datasets.common import LABEL_COLUMNS

root = Path.home() / "ich_windowing" / "data" / "physionet_ich"
print(f"root = {root}")
print(f"root exists = {root.exists()}")

samples = build_samples(root)
print(f"build_samples() succeeded: {len(samples)} samples")

patient_ids = sorted(set(s.patient_id for s in samples))
print(f"unique patients: {len(patient_ids)}")
print(f"first 5 patient_ids: {patient_ids[:5]}")
print(f"last 5 patient_ids: {patient_ids[-5:]}")

# spot-check a couple of sample_ids and label shapes
for s in samples[:3]:
    print(f"  sample_id={s.sample_id} patient_id={s.patient_id} site={s.site} "
          f"labels_shape={s.labels.shape} lesion_size_mm3={s.lesion_size_mm3}")

assert len(LABEL_COLUMNS) == samples[0].labels.shape[0]

# actually invoke hu_loader on a couple of samples to confirm end-to-end file IO works
import numpy as np
for s in samples[:2]:
    arr = s.hu_loader()
    print(f"  hu_loader() for {s.sample_id}: shape={arr.shape}, dtype={arr.dtype}, "
          f"range=[{arr.min():.1f}, {arr.max():.1f}]")

print()
print("VERIFICATION PASSED: real build_samples() loader runs end-to-end against "
      "real data with no FileNotFoundError.")
