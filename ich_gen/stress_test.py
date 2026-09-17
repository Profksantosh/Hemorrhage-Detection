"""Synthetic HU-recalibration robustness test (manuscript Section VI-F,
tests H4).

Applies a controlled affine HU transform x' = a*x + b to a held-out RSNA
subset never used in any other evaluation, simulating scanner
calibration drift never seen in any of the three REAL external sites
(CQ500, PhysioNet-ICH, BHSD). Because the only difference between the
perturbed and unperturbed versions of a given slice is this controlled
transform -- the anatomy, patient, and label are identical -- this
isolates robustness to HU/calibration shift specifically, cleanly
separated from the population/annotation differences that also vary
across the three real external sites (manuscript Section VI-F).
"""
from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class HUPerturbation:
    """x' = scale * x + offset, in Hounsfield units.

    Default bounds are a starting point representative of documented
    CT quality-assurance calibration-drift tolerances (typically a few
    HU of offset drift and a few percent of scale drift between
    QA-passing scanner states); the manuscript commits (Section VI-F) to
    citing a specific QA-literature reference for these bounds once
    selected -- treat the numbers below as provisional pending that
    citation, not as an established clinical tolerance.
    """
    scale: float
    offset: float

    def apply(self, hu: np.ndarray) -> np.ndarray:
        return self.scale * hu + self.offset


def make_perturbation_grid(scale_range: tuple[float, float] = (0.85, 1.15),
                            offset_range: tuple[float, float] = (-60.0, 60.0),
                            n_scale: int = 5, n_offset: int = 5
                            ) -> list[HUPerturbation]:
    """A grid of (scale, offset) perturbations spanning mild to severe
    simulated calibration drift, for the degradation-curve plot specified
    in manuscript Section VI-F / Table VII."""
    scales = np.linspace(*scale_range, n_scale)
    offsets = np.linspace(*offset_range, n_offset)
    return [HUPerturbation(float(s), float(o)) for s in scales for o in offsets]


def run_stress_test(model_predict_fn, samples, perturbations: list[HUPerturbation],
                     evaluate_fn) -> list[dict]:
    """Generic driver: for each perturbation, apply it to every sample's
    raw HU array before windowing/inference, run `model_predict_fn` to get
    predictions, and score with `evaluate_fn`.

    Parameters
    ----------
    model_predict_fn : callable(list[Sample_with_perturbed_hu]) -> (y_true, y_prob)
        Wraps a trained model's inference over a perturbed sample list;
        left generic so both a WICL model and any baseline can be plugged
        in via the same driver, with no perturbation-specific logic
        duplicated per condition.
    samples : list[ich_gen.datasets.common.Sample], a held-out RSNA subset
        NEVER used in training or in any other reported evaluation
        (manuscript Section VI-F).
    perturbations : from make_perturbation_grid().
    evaluate_fn : callable(y_true, y_prob) -> dict of scalar metrics,
        e.g. a thin wrapper around ich_gen.evaluate.evaluate_pooled_and_stratified.

    Returns
    -------
    list of dicts, one per perturbation, each with the perturbation's
    (scale, offset) plus whatever evaluate_fn returns -- ready to become
    manuscript Table VII / the degradation-curve figure.
    """
    results = []
    for pert in perturbations:
        perturbed_samples = _apply_perturbation_to_samples(samples, pert)
        y_true, y_prob = model_predict_fn(perturbed_samples)
        metrics = evaluate_fn(y_true, y_prob)
        results.append({"scale": pert.scale, "offset": pert.offset, **metrics})
    return results


def _apply_perturbation_to_samples(samples, pert: HUPerturbation):
    """Return new Sample objects whose hu_loader applies `pert` on top of
    the original loader, without mutating the originals (so the same
    `samples` list can be reused across every perturbation in the grid)."""
    from ich_gen.datasets.common import Sample

    def _wrap(original_loader):
        return lambda: pert.apply(original_loader())

    return [
        Sample(sample_id=s.sample_id, patient_id=s.patient_id, site=s.site,
               hu_loader=_wrap(s.hu_loader), labels=s.labels,
               lesion_size_mm3=s.lesion_size_mm3)
        for s in samples
    ]
