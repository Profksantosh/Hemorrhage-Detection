import numpy as np
import torch

from ich_gen.windowing import (
    window_hu, three_window_composite, sample_three_window_composite,
    two_independent_draws, to_tensor, DEFAULT_SAMPLING_RANGES,
    resize_composite, resize_hu, differentiable_window_hu,
)


def _random_hu(seed=0, shape=(64, 64)):
    rng = np.random.RandomState(seed)
    return rng.uniform(-1000, 3000, size=shape).astype(np.float32)


def test_window_hu_range_and_clipping():
    hu = _random_hu()
    out = window_hu(hu, level=40, width=80)
    assert out.shape == hu.shape
    assert out.min() >= 0.0 and out.max() <= 255.0
    # values far below the window should clip to 0, far above to 255
    assert window_hu(np.array([-2000.0]), 40, 80)[0] == 0.0
    assert window_hu(np.array([5000.0]), 40, 80)[0] == 255.0


def test_window_hu_rejects_nonpositive_width():
    import pytest
    with pytest.raises(ValueError):
        window_hu(_random_hu(), level=40, width=0)


def test_three_window_composite_shape():
    hu = _random_hu()
    comp = three_window_composite(hu)
    assert comp.shape == (3, 64, 64)
    assert comp.dtype == np.float32


def test_stochastic_draws_are_independent_and_bounded():
    hu = _random_hu()
    v1, v2 = two_independent_draws(hu, breadth="moderate")
    assert v1.shape == v2.shape == (3, 64, 64)
    assert not np.allclose(v1, v2), "two independent draws should (almost surely) differ"
    for v in (v1, v2):
        assert v.min() >= 0.0 and v.max() <= 255.0


def test_sampling_breadth_ordering_widens_variance():
    """Wider sampling breadths should produce more inter-draw variance --
    a basic sanity check on DEFAULT_SAMPLING_RANGES before the real
    delta_level/delta_width ablation (manuscript Section VI-D) is run."""
    hu = _random_hu(seed=1)
    rng = np.random.RandomState(42)

    def total_variation(breadth):
        draws = [sample_three_window_composite(hu, breadth, rng) for _ in range(20)]
        stacked = np.stack(draws)
        return stacked.std(axis=0).mean()

    narrow_var = total_variation("narrow")
    wide_var = total_variation("wide")
    assert wide_var > narrow_var


def test_to_tensor_normalization():
    hu = _random_hu()
    comp = three_window_composite(hu)
    t = to_tensor(comp)
    assert t.shape == (3, 64, 64)
    assert t.dtype.is_floating_point


def test_resize_composite_changes_spatial_shape_only():
    hu = _random_hu(shape=(80, 120))
    comp = three_window_composite(hu)
    resized = resize_composite(comp, size=224)
    assert resized.shape == (3, 224, 224)
    assert resized.dtype == np.float32
    assert resized.min() >= 0.0 and resized.max() <= 255.0


def test_resize_composite_is_a_noop_at_matching_size():
    hu = _random_hu(shape=(224, 224))
    comp = three_window_composite(hu)
    resized = resize_composite(comp, size=224)
    assert resized.shape == comp.shape
    assert np.allclose(resized, comp, atol=1.0)  # bilinear at identity scale ~ exact


def test_resize_hu_preserves_value_range_approximately():
    hu = _random_hu(shape=(64, 64), seed=3)
    resized = resize_hu(hu, size=128)
    assert resized.shape == (128, 128)
    # resizing (interpolating) must not invent values wildly outside the
    # original range
    assert resized.min() >= hu.min() - 1.0
    assert resized.max() <= hu.max() + 1.0


def test_differentiable_window_hu_matches_numpy_window_hu():
    hu = _random_hu(shape=(32, 32), seed=4)
    hu_t = torch.from_numpy(hu).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    level = torch.tensor([40.0])
    width = torch.tensor([80.0])
    torch_out = differentiable_window_hu(hu_t, level, width).squeeze().numpy()
    numpy_out = window_hu(hu, 40.0, 80.0)
    assert np.allclose(torch_out, numpy_out, atol=1e-3)


def test_differentiable_window_hu_gradients_flow_to_level_and_width():
    hu = _random_hu(shape=(32, 32), seed=5)
    hu_t = torch.from_numpy(hu).unsqueeze(0).unsqueeze(0)
    level = torch.tensor([40.0], requires_grad=True)
    width = torch.tensor([80.0], requires_grad=True)
    out = differentiable_window_hu(hu_t, level, width)
    out.mean().backward()
    assert level.grad is not None and torch.isfinite(level.grad)
    assert width.grad is not None and torch.isfinite(width.grad)


def test_differentiable_window_hu_output_bounded():
    hu = torch.tensor([[-5000.0, 5000.0, 40.0]]).view(1, 1, 1, 3)
    out = differentiable_window_hu(hu, torch.tensor([40.0]), torch.tensor([80.0]))
    assert out.min() >= 0.0 and out.max() <= 255.0


def test_differentiable_window_hu_batched_per_sample_params():
    """Different samples in a batch must be windowed with THEIR OWN
    predicted (level, width), not a shared value -- this is what
    WindowEstimatorModule/WEMClassifier relies on."""
    hu = torch.randn(3, 1, 16, 16) * 200
    level = torch.tensor([0.0, 40.0, 100.0])
    width = torch.tensor([50.0, 80.0, 200.0])
    out = differentiable_window_hu(hu, level, width)
    assert out.shape == (3, 1, 16, 16)
    # verify sample 0's output matches windowing with ONLY its own params
    expected_0 = differentiable_window_hu(hu[0:1], level[0:1], width[0:1])
    assert torch.allclose(out[0:1], expected_0)
