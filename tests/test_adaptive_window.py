"""Tests for baseline 3a/3b reimplementations (manuscript Section VI-A,
Section II-G): Songsaeng et al.'s HRT [14] and Karki et al.'s WEM [33]."""
import numpy as np
import torch

from ich_gen.models.adaptive_window import (
    select_hrt_window, hrt_composite, WindowEstimatorModule,
    decode_window_params, WEMClassifier,
)


def _random_hu(seed=0, shape=(96, 96)):
    rng = np.random.RandomState(seed)
    return rng.uniform(-1000, 3000, size=shape).astype(np.float32)


def test_select_hrt_window_returns_valid_window():
    hu = _random_hu()
    level, width = select_hrt_window(hu)
    assert width > 0
    assert isinstance(level, float) and isinstance(width, float)


def test_hrt_composite_shape_and_range():
    hu = _random_hu(shape=(64, 64))
    comp = hrt_composite(hu)
    assert comp.shape == (3, 64, 64)
    assert comp.min() >= 0.0 and comp.max() <= 255.0
    # all three channels must be identical (single selected window,
    # replicated -- unlike baseline 2's three DIFFERENT canonical windows)
    assert np.allclose(comp[0], comp[1]) and np.allclose(comp[1], comp[2])


def test_decode_window_params_never_produces_nonpositive_width():
    raw = torch.tensor([[-100.0, 100.0], [0.0, 0.0], [50.0, -50.0]])
    level, width = decode_window_params(raw)
    assert (width > 0).all()


def test_window_estimator_module_output_shape():
    wem = WindowEstimatorModule()
    x = torch.randn(5, 1, 128, 128)
    raw = wem(x)
    assert raw.shape == (5, 2)


def test_wem_classifier_forward_shape():
    model = WEMClassifier(pretrained=False)
    hu_for_wem = torch.randn(2, 1, 224, 224)
    hu_raw = torch.randn(2, 1, 224, 224) * 300
    logits = model(hu_for_wem, hu_raw)
    assert logits.shape == (2, 6)


def test_wem_classifier_gradients_reach_both_submodules():
    model = WEMClassifier(pretrained=False)
    hu_for_wem = torch.randn(2, 1, 224, 224)
    hu_raw = torch.randn(2, 1, 224, 224) * 300
    y = torch.randint(0, 2, (2, 6)).float()

    logits = model(hu_for_wem, hu_raw)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    wem_grads = [p.grad for p in model.wem.parameters()]
    clf_grads = [p.grad for p in model.classifier.parameters()]
    assert len(wem_grads) > 0 and all(g is not None and torch.isfinite(g).all()
                                        for g in wem_grads)
    assert len(clf_grads) > 0 and all(g is not None and torch.isfinite(g).all()
                                        for g in clf_grads)
