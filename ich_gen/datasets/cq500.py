"""CQ500 external test-site loader (manuscript ref [28], Chilamkurthy et al.,
The Lancet 2018; used as external test data in manuscript refs [8], [10],
[11], [14], [18], [20]).

SCHEMA ASSUMPTION -- VERIFY BEFORE USE
=======================================
CQ500's public release ships per-study, per-reader annotation CSVs (three
independent radiologist reads per study) rather than a single
already-aggregated label file. This loader expects an intermediate,
ALREADY-AGGREGATED CSV named `cq500_labels_aggregated.csv` with columns:

    study_id, slice_path, any, epidural, intraparenchymal,
    intraventricular, subarachnoid, subdural

We have not independently verified CQ500's exact raw column names against
a downloaded copy of the dataset as part of preparing this manuscript --
`prepare_cq500_labels.py` (in this same directory) is a documented,
best-effort starting point for producing that aggregated file via
majority vote across the three readers, and MUST be checked against your
actual downloaded CSV's real column names before running it. This
loader's job starts only after that aggregation step; require_columns()
below will fail loudly, not silently, if the aggregated file's schema is
wrong.

CQ500 is used STRICTLY as zero-shot external test data (Section IV-B):
no CQ500 slice is ever used for training or hyperparameter selection for
any condition compared in this study, and validation-set threshold
locking (per Zhao et al. [20]'s protocol) is implemented in evaluate.py,
not here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ich_gen.datasets.common import LABEL_COLUMNS, Sample, require_columns
from ich_gen.datasets.rsna import read_hu  # DICOM HU reader, format-agnostic


def build_samples(root: Path,
                   aggregated_labels_csv: str = "cq500_labels_aggregated.csv"
                   ) -> list[Sample]:
    root = Path(root)
    df = pd.read_csv(root / aggregated_labels_csv)
    require_columns(df.columns, ["study_id", "slice_path"] + list(LABEL_COLUMNS),
                     "CQ500 aggregated labels CSV")

    label_matrix = df[list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
    samples = []
    for i, row in df.iterrows():
        slice_path = root / row["slice_path"]
        samples.append(Sample(
            sample_id=f"CQ500_{row['study_id']}_{i}",
            patient_id=f"CQ500_{row['study_id']}",  # CQ500 provides one
                                                       # study per patient;
                                                       # verify this 1:1
                                                       # assumption against
                                                       # your release
            site="cq500",
            hu_loader=lambda p=slice_path: read_hu(p),
            labels=label_matrix[i],
            lesion_size_mm3=None,  # CQ500 has study-level, not voxel-level,
                                     # labels -- no size stratification
                                     # available for this site (Section IV-E
                                     # notes BHSD as the size-stratification
                                     # source)
        ))
    return samples
