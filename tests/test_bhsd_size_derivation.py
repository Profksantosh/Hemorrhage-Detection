"""Tests for the BHSD volumetric-to-slice-label derivation logic
(manuscript Section IV-D), the component the paper's central size-
stratified analysis (H1) depends on most directly."""
import numpy as np

from ich_gen.datasets.bhsd import derive_slice_labels_and_sizes
from ich_gen.datasets.common import LABEL_COLUMNS


def test_single_lesion_labels_and_volume():
    mask = np.zeros((10, 20, 20), dtype=np.int16)
    mask[2:6, 5:9, 5:9] = 2  # intraparenchymal, 4x4x4 = 64 voxels
    labels, sizes = derive_slice_labels_and_sizes(mask, voxel_volume_mm3=1.0,
                                                    min_component_voxels=5)
    ip_idx = LABEL_COLUMNS.index("intraparenchymal")
    any_idx = LABEL_COLUMNS.index("any")
    for z in range(2, 6):
        assert labels[z, ip_idx] == 1.0
        assert labels[z, any_idx] == 1.0
        assert sizes[z] == 64.0
    for z in [0, 1, 6, 7, 8, 9]:
        assert labels[z].sum() == 0.0
        assert sizes[z] is None


def test_small_component_filtered_as_noise():
    mask = np.zeros((5, 10, 10), dtype=np.int16)
    mask[2, 3, 3] = 2  # single voxel
    labels, sizes = derive_slice_labels_and_sizes(mask, voxel_volume_mm3=1.0,
                                                    min_component_voxels=5)
    assert labels.sum() == 0.0
    assert all(s is None for s in sizes)


def test_voxel_volume_scaling():
    mask = np.zeros((5, 10, 10), dtype=np.int16)
    mask[1:3, 2:4, 2:4] = 3  # intraventricular, 2x2x2 = 8 voxels
    labels, sizes = derive_slice_labels_and_sizes(mask, voxel_volume_mm3=2.5,
                                                    min_component_voxels=1)
    iv_idx = LABEL_COLUMNS.index("intraventricular")
    assert labels[1, iv_idx] == 1.0
    assert sizes[1] == 8 * 2.5


def test_largest_component_wins_when_multiple_overlap_a_slice():
    """If two components of the SAME subtype touch one slice, the manuscript
    specifies using the LARGEST component's volume (Section IV-D
    docstring), since size-tercile assignment should reflect clinically
    dominant lesion burden."""
    mask = np.zeros((5, 20, 20), dtype=np.int16)
    mask[2, 1:3, 1:3] = 5    # small subdural component, slice 2 only, 4 vox
    mask[1:4, 10:15, 10:15] = 5  # large subdural component spanning slice 2, 3*5*5=75 vox
    labels, sizes = derive_slice_labels_and_sizes(mask, voxel_volume_mm3=1.0,
                                                    min_component_voxels=1)
    assert sizes[2] == 75.0  # the larger of the two components at slice 2


def test_multiple_subtypes_in_same_slice():
    mask = np.zeros((3, 20, 20), dtype=np.int16)
    mask[1, 1:4, 1:4] = 2  # intraparenchymal
    mask[1, 10:13, 10:13] = 5  # subdural
    labels, sizes = derive_slice_labels_and_sizes(mask, voxel_volume_mm3=1.0,
                                                    min_component_voxels=1)
    ip_idx = LABEL_COLUMNS.index("intraparenchymal")
    sdh_idx = LABEL_COLUMNS.index("subdural")
    assert labels[1, ip_idx] == 1.0
    assert labels[1, sdh_idx] == 1.0
    assert labels[1, LABEL_COLUMNS.index("any")] == 1.0
