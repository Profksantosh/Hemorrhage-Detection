"""ich_gen: reference implementation for

Windowing-Invariant Representation Learning for Cross-Site Generalization
in Deep Learning-Based Intracranial Hemorrhage Detection and Subtyping

This package implements Sections IV-VII of the companion manuscript exactly
as specified: patient-grouped multi-site data loading, WICL, all five
required baseline families, the statistical analysis plan, the synthetic
HU-recalibration stress test, and the failure-mode / mechanism-attribution
analysis.

No experiment in this package has been run against real patient data as
part of preparing the manuscript. All code is smoke-tested against
synthetic tensors only (see tests/). Running it against real RSNA/CQ500/
PhysioNet-ICH/BHSD data, obtained by the user under each dataset's own
license, is required to produce the numbers in the manuscript's Tables
II-VII.
"""

__version__ = "0.1.0"
