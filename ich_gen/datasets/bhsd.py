"""BHSD (Brain Hemorrhage Segmentation Dataset) external test-site loader
(manuscript ref [30], Wu et al., MLMI 2023; used as external test data by
manuscript ref [18]).

Implements manuscript Section IV-D: because BHSD is voxel-annotated
rather than natively slice-labeled for classification, subtype and
lesion-size labels are DERIVED per slice from the volumetric masks:
  - subtype presence: which mask class value(s) appear in a given slice
  - lesion size: 3D connected-component volume (voxel count * voxel
    volume, from the NIfTI affine) of the component a slice's positive
    region belongs to, matched to a small/medium/large tercile computed
    from the RSNA/BHSD training-pool distribution and held fixed across
    sites (see ich_gen.datasets.common.assign_size_tercile).

This is also, per Section IV-D, the site used for the size-stratified
external analysis at the center of hypothesis H1 -- so the correctness of
this file's connected-component logic matters more to the paper's central
claim than any other single loader, and it is exercised directly by
tests/test_bhsd_size_derivation.py against synthetic masks with known
ground-truth volumes.

SCHEMA ASSUMPTION -- VERIFY BEFORE USE
=======================================
BHSD's exact integer class-ID-to-subtype mapping (CLASS_ID_TO_SUBTYPE
below) has NOT been independently verified against the dataset's official
documentation/dataset card as part of preparing this manuscript, and
must be confirmed (e.g. via the dataset's GitHub README or metadata JSON)
before running this loader against real BHSD volumes -- see the note in
CLASS_ID_TO_SUBTYPE.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import label as cc_label, regionprops

from ich_gen.datasets.common import LABEL_COLUMNS, SUBTYPES, Sample

try:
    import nibabel as nib
    _HAS_NIBABEL = True
except ImportError:  # pragma: no cover
    _HAS_NIBABEL = False

# NOTE: placeholder mapping, ordered to match a plausible BHSD release
# (background=0, then the 5 subtypes) -- CONFIRM against BHSD's actual
# class definitions before use; if the real mapping differs, only this
# dict needs to change, nothing else in this file.
CLASS_ID_TO_SUBTYPE: dict[int, str] = {
    1: "epidural",
    2: "intraparenchymal",
    3: "intraventricular",
    4: "subarachnoid",
    5: "subdural",
}
assert set(CLASS_ID_TO_SUBTYPE.values()) == set(SUBTYPES)


def derive_slice_labels_and_sizes(mask_3d: np.ndarray,
                                   voxel_volume_mm3: float,
                                   min_component_voxels: int = 5,
                                   ) -> tuple[np.ndarray, list[float | None]]:
    """Core derivation logic (Section IV-D), separated from I/O so it can
    be unit-tested against synthetic arrays with known answers.

    Parameters
    ----------
    mask_3d : integer array (Z, H, W) of class IDs (0 = background).
    voxel_volume_mm3 : physical volume of one voxel, from the NIfTI affine.
    min_component_voxels : components smaller than this are treated as
        segmentation noise and ignored (both for labeling and sizing).

    Returns
    -------
    slice_labels : float32 array (Z, 6), columns = LABEL_COLUMNS order.
    slice_lesion_size_mm3 : list of length Z; for slices with more than
        one component/subtype present, the LARGEST component's volume is
        used, since size-tercile assignment (Section IV-E) is intended to
        reflect clinically dominant lesion burden, not an ambiguous
        multi-component average. `None` for slices with no hemorrhage.
    """
    z_dim = mask_3d.shape[0]
    slice_labels = np.zeros((z_dim, len(LABEL_COLUMNS)), dtype=np.float32)
    slice_lesion_size_mm3: list[float | None] = [None] * z_dim

    for class_id, subtype in CLASS_ID_TO_SUBTYPE.items():
        subtype_col = LABEL_COLUMNS.index(subtype)
        binary = (mask_3d == class_id)
        if not binary.any():
            continue

        components = cc_label(binary, connectivity=3)
        for region in regionprops(components):
            if region.area < min_component_voxels:
                continue  # treat as noise, per min_component_voxels
            volume_mm3 = float(region.area) * voxel_volume_mm3
            z_coords = region.coords[:, 0]
            for z in np.unique(z_coords):
                slice_labels[z, subtype_col] = 1.0
                slice_labels[z, 0] = 1.0  # "any"
                current = slice_lesion_size_mm3[z]
                if current is None or volume_mm3 > current:
                    slice_lesion_size_mm3[z] = volume_mm3

    return slice_labels, slice_lesion_size_mm3


def build_samples(root: Path,
                   manifest_csv_name: str = "bhsd_manifest.csv"
                   ) -> list[Sample]:
    """`manifest_csv_name` is expected to list one row per volume with
    columns [volume_id, ct_path, mask_path], both paths relative to
    `root`. Constructing this manifest from BHSD's actual release
    directory structure is left to a small site-specific script, since
    (per the dataset's GitHub release) the exact directory layout may
    differ between the pixel-level-annotated (192 volumes) and
    slice-level-annotated (2200 volumes) subsets described by Wu et al.;
    this loader only consumes the manifest, not the raw release tree.
    """
    if not _HAS_NIBABEL:
        raise ImportError("nibabel is required; `pip install nibabel`")
    root = Path(root)
    manifest = pd.read_csv(root / manifest_csv_name)
    from ich_gen.datasets.common import require_columns
    require_columns(manifest.columns, ["volume_id", "ct_path", "mask_path"],
                     "BHSD manifest CSV")

    samples: list[Sample] = []
    for _, row in manifest.iterrows():
        ct_path = root / row["ct_path"]
        mask_path = root / row["mask_path"]

        ct_img = nib.load(str(ct_path))
        mask_img = nib.load(str(mask_path))
        zooms = ct_img.header.get_zooms()[:3]
        voxel_volume_mm3 = float(np.prod(zooms))

        mask_3d = np.asarray(mask_img.dataobj).astype(np.int16)
        # BHSD volumes are stored (H, W, Z) or (Z, H, W) depending on
        # release orientation; this loader assumes the mask's LAST axis is
        # the through-plane (slice) axis and transposes to (Z, H, W) --
        # verify this matches your actual volumes (e.g. by checking
        # ct_img.header.get_data_shape() against known slice counts)
        # before trusting the derived per-slice labels.
        mask_3d = np.moveaxis(mask_3d, -1, 0)

        slice_labels, slice_sizes = derive_slice_labels_and_sizes(
            mask_3d, voxel_volume_mm3)

        ct_data = np.asarray(ct_img.dataobj).astype(np.float32)
        ct_data = np.moveaxis(ct_data, -1, 0)  # match mask orientation

        volume_id = row["volume_id"]
        for z in range(mask_3d.shape[0]):
            samples.append(Sample(
                sample_id=f"BHSD_{volume_id}_{z}",
                patient_id=f"BHSD_{volume_id}",  # one volume == one patient
                                                   # for BHSD; verify if a
                                                   # patient can contribute
                                                   # >1 volume in your
                                                   # release, and namespace
                                                   # accordingly
                site="bhsd",
                hu_loader=lambda arr=ct_data, zz=z: arr[zz],
                labels=slice_labels[z],
                lesion_size_mm3=slice_sizes[z],
            ))
    return samples
