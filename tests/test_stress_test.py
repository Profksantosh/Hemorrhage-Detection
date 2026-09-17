import numpy as np

from ich_gen.datasets.common import Sample
from ich_gen.stress_test import HUPerturbation, make_perturbation_grid, run_stress_test


def test_hu_perturbation_applies_affine_transform():
    hu = np.array([100.0, 200.0, -50.0])
    pert = HUPerturbation(scale=1.1, offset=10.0)
    out = pert.apply(hu)
    assert np.allclose(out, hu * 1.1 + 10.0)


def test_perturbation_grid_size():
    grid = make_perturbation_grid(n_scale=3, n_offset=4)
    assert len(grid) == 12


def test_run_stress_test_does_not_mutate_original_samples():
    rng = np.random.RandomState(0)
    original_arr = rng.uniform(-200, 300, size=(16, 16)).astype(np.float32)
    samples = [Sample("s0", "p0", "rsna", lambda a=original_arr: a,
                       np.zeros(6, dtype=np.float32), None)]

    def predict_fn(perturbed_samples):
        y_true = np.stack([s.labels for s in perturbed_samples])
        y_prob = np.stack([[s.hu_loader().mean()] * 6 for s in perturbed_samples])
        return y_true, y_prob

    def eval_fn(y_true, y_prob):
        return {"mean_score": float(y_prob.mean())}

    grid = make_perturbation_grid(n_scale=2, n_offset=2)
    results = run_stress_test(predict_fn, samples, grid, eval_fn)

    assert len(results) == len(grid)
    # original sample's loader must be untouched
    assert np.allclose(samples[0].hu_loader(), original_arr)


def test_perturbation_actually_changes_downstream_scores():
    rng = np.random.RandomState(1)
    arr = rng.uniform(-200, 300, size=(16, 16)).astype(np.float32)
    samples = [Sample("s0", "p0", "rsna", lambda a=arr: a,
                       np.zeros(6, dtype=np.float32), None)]

    def predict_fn(perturbed_samples):
        y_true = np.stack([s.labels for s in perturbed_samples])
        y_prob = np.stack([[s.hu_loader().mean()] * 6 for s in perturbed_samples])
        return y_true, y_prob

    def eval_fn(y_true, y_prob):
        return {"mean_score": float(y_prob.mean())}

    grid = make_perturbation_grid(n_scale=3, n_offset=3)
    results = run_stress_test(predict_fn, samples, grid, eval_fn)
    scores = {round(r["mean_score"], 4) for r in results}
    assert len(scores) > 1, "perturbations should produce varying downstream scores"
